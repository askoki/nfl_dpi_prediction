import os
import sys
import pandas as pd
from datetime import datetime
from settings import RAW_DATA_DIR, DataV3, DATA_V3_SUBVERSION
from src.features.helpers.processing import add_missing_timestamp_values
from src.features.helpers.processing_v3 import get_closest_players, get_players_and_ball_indices, calculate_distance, \
    normalize_according_to_play_direction, check_group_event
from src.features.helpers.processing_v4 import home_has_possession, calculate_team_sitation

week_num = int(sys.argv[1])

data_v3 = DataV3(DATA_V3_SUBVERSION)
save_file_path = data_v3.get_step1_checkpoint_path(week_num)
try:
    clean_df = pd.read_csv(save_file_path)
    save_file_exists = True
except FileNotFoundError:
    save_file_exists = False

if not save_file_exists:
    print("Started loading data")
    play_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'plays.csv'))
    games_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'games.csv'))

    week_and_games = games_df[games_df.week == week_num]
    tracking_df = pd.read_csv(os.path.join(RAW_DATA_DIR, f'week{week_num}.csv'))
    print("Data loaded. Start processing timestamps")

    tracking_df = add_missing_timestamp_values(tracking_df)

    games_n_plays_df = play_df.merge(week_and_games, how='inner', on='gameId')
    m_grouped = games_n_plays_df.groupby(['gameId', 'playId'])

    df_t = tracking_df.merge(games_n_plays_df, how='left', on=['gameId', 'playId'])

    # Remove all events without 'pass_forward'
    df_t_grouped = df_t.groupby(['gameId', 'playId'])
    df_t_v3 = df_t.copy().sort_index()
    for name, group in df_t_grouped:
        game_id, play_id = name

        # if group does not contain pass forward, drop it
        if all(group.event != 'pass_forward'):
            df_t_v3 = df_t_v3[(df_t_v3.gameId != game_id) | (df_t_v3.playId != play_id)]

    df_t_v3_s = df_t_v3.sort_values(by=['gameId', 'playId', 'time', 'event'])
    df_t_v3_s = df_t_v3_s.reset_index(drop=True)

    df_t_grouped = df_t_v3_s.groupby(['gameId', 'playId'])

    # remove all values before 'pass_forward'
    print("Removing all values before pass forward event...")
    for name, group in df_t_grouped:
        game_id, play_id = name

        pass_forward_frame_id = group[group.event == 'pass_forward'].index.min() - 1
        remove_start = group.index.min()
        df_t_v3_s = df_t_v3_s.drop(df_t_v3_s.loc[remove_start:pass_forward_frame_id].index)

    pd.options.mode.chained_assignment = None
    gb = df_t_v3_s.groupby(['gameId', 'playId'])

    print('Getting closest players...')
    keep_indices = []
    for name, group in gb:
        game_id, play_id = name
        try:
            event_3rd = group.event.unique()[2]
        except IndexError:
            print('Number of events is < 3, skipping...')
            continue

        situation_df = group[group.event == event_3rd]

        # convert dataframe into series
        ball_row = situation_df[situation_df.team == 'football'].head(1)

        # remove ball
        player_situation_df = situation_df[situation_df.team != 'football']
        try:
            p1, p2 = get_closest_players(player_situation_df, ball_row.x.item(), ball_row.y.item())
        except ValueError:
            print('Value Error raised. This group will be skipped.')
            continue
        p_n_b_indices = get_players_and_ball_indices(group, p1, p2)
        if p_n_b_indices:
            keep_indices.extend(p_n_b_indices)

    clean_df = df_t_v3_s[df_t_v3_s.index.isin(keep_indices)]

    clean_df.to_csv(
        save_file_path,
        index=False
    )

print('Normalize...')
clean_df = normalize_according_to_play_direction(clean_df)

clean_df['homeHasPossession'] = clean_df.apply(
    lambda row: home_has_possession(row), axis=1
)

clean_df['teamSituation'] = clean_df.apply(
    lambda row: calculate_team_sitation(row), axis=1
)

print('Creating features...')
min_df = clean_df[[
    'time', 'x', 'y', 's', 'o', 'dir', 'event', 'team',
    'gameId', 'playId', 'frameId', 'isDefensivePI'
]]
gb_2 = clean_df.groupby(['gameId', 'playId', 'frameId'])

# ball direction and orientation are NaN
calc_df = pd.DataFrame(
    columns=[
        'time',
        'att_def_d', 'att_ball_d', 'def_ball_d',
        'att_s', 'def_s', 'ball_s',
        'att_o', 'def_o',
        'att_dir', 'def_dir',
        'event', 'gameId', 'playId', 'frameId', 'isDefensivePI'
    ]
)
GROUP_SIZE_MINIMUM = 3
for name, group in gb_2:
    game_id, play_id, frameId = name

    if len(group) < GROUP_SIZE_MINIMUM:
        continue

    ball = group[group.teamSituation == 'football'].head(1).squeeze()
    p_att = group[group.teamSituation == 'attacking'].head(1).squeeze()
    p_def = group[group.teamSituation == 'defending'].head(1).squeeze()

    group_row = group.head(1).squeeze()

    group_events = group.event.unique().tolist()

    dict_to_append = {
        'time': group_row.time,
        'att_def_d': calculate_distance(p_att.x, p_att.y, p_def.x, p_def.y),
        'att_ball_d': calculate_distance(p_att.x, p_att.y, ball.x, ball.y),
        'def_ball_d': calculate_distance(p_def.x, p_def.y, ball.x, ball.y),
        'att_s': p_att.s, 'def_s': p_def.s, 'ball_s': ball.s,
        'att_a': p_att.a, 'def_a': p_def.a, 'ball_a': ball.a,
        'att_o': p_att.o, 'def_o': p_def.o,
        'att_dir': p_att.dir, 'def_dir': p_def.dir,
        'event': group_row.event,
        'pass_arrived': check_group_event(group_events, 'pass_arrived'),
        'pass_outcome_caught': check_group_event(group_events, 'pass_outcome_caught'),
        'tackle': check_group_event(group_events, 'tackle'),
        'first_contact': check_group_event(group_events, 'first_contact'),
        'pass_outcome_incomplete': check_group_event(group_events, 'pass_outcome_incomplete'),
        'out_of_bounds': check_group_event(group_events, 'out_of_bounds'),
        'week': week_num,
        'gameId': group_row.gameId,
        'playId': group_row.playId,
        'frameId': group_row.frameId,
        'isDefensivePI': group_row.isDefensivePI
    }

    calc_df = calc_df.append(
        dict_to_append,
        ignore_index=True
    )

print("Saving data...")
calc_df.to_csv(
    data_v3.get_step1_end_path(week_num),
    index=False
)

print(f'End time: {datetime.now().strftime("%H:%M:%S")}')
