import os
import sys
import pandas as pd
from datetime import datetime
from settings import RAW_DATA_DIR, INTERIM_DATA_DIR
from src.features.helpers.processing import add_missing_timestamp_values, normalize_play_direction

WEEK_MIN = 1
WEEK_MAX = 17
if not sys.argv[1]:
    print(f'Enter week number [{WEEK_MIN}-{WEEK_MAX}]')
    sys.exit()

try:
    week_num = int(sys.argv[1])
    if not (WEEK_MIN <= week_num <= WEEK_MAX):
        print(F'Entered week number is not in accepted range {WEEK_MIN}-{WEEK_MAX}')
        sys.exit()
except ValueError:
    print("Input is not an integer")
    sys.exit()

print("Started loading data")
start_time = datetime.now()
print(f'Time: {start_time.strftime("%H:%M:%S")}')

play_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "plays.csv"))
games_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "games.csv"))
n_week_tracking_df = pd.read_csv(os.path.join(RAW_DATA_DIR, f'week{week_num}.csv'))

print("Data loaded. Start merging")

week1_games = games_df[games_df.week == week_num]
games_n_plays_df = play_df.merge(week1_games, how='inner', on='gameId')
group_t = n_week_tracking_df.groupby(['time', 'frameId'])

# ### Merge all 15 (or different count) features of one timestamp -> player positions and ball position
print("Dropping all groups that do not contain football...")
# ### Drop all records that doesnt have football
drop_t = group_t.filter(lambda row: (row['displayName'] == 'Football').any())
# ### Drop all values that have only football
drop_f = drop_t.groupby(['time', 'frameId']).filter(lambda row: (row['displayName'] != 'Football').any())

keep_f = drop_f.copy()

keep_f = add_missing_timestamp_values(keep_f)

# ###  Success now apply to whole week 1
# #### First normalize
print("Normalize data...")
normalize_tracking = keep_f.copy()

normalize_tracking = normalize_play_direction(normalize_tracking)

games_n_plays_df_cleaned = games_n_plays_df.drop(columns=[
    'playDescription',
    'penaltyJerseyNumbers', 'penaltyCodes', 'gameDate',
])


def home_has_possession(row):
    if row.possessionTeam == row.homeTeamAbbr:
        return 1
    return 0


games_n_plays_df_cleaned['homeHasPossesion'] = games_n_plays_df_cleaned.apply(
    lambda row: home_has_possession(row), axis=1
)

gnp_core_cols = games_n_plays_df_cleaned[
    [
        'gameId', 'playId', 'homeHasPossesion',
        'isDefensivePI'
    ]
]

normalize_tracking.to_csv(os.path.join(INTERIM_DATA_DIR, f'plot_tracking_data_week{week_num}.csv'), index=False)
