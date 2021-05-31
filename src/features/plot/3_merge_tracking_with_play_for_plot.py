import gc
import os
import pandas as pd
from datetime import datetime
from settings import RAW_DATA_DIR, INTERIM_DATA_DIR

WEEK_MIN = 1
WEEK_MAX = 17
play_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "plays.csv"))
games_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "games.csv"))

gc.collect()

print(f'Start time: {datetime.now().strftime("%H:%M:%S")}')
for i in range(WEEK_MAX):
    week_num = i + 1
    print(f'Started with week {week_num}: {datetime.now()}')
    week_and_games = games_df[games_df.week == week_num]
    print(f'Merging games and plays df...')
    games_and_plays_df = play_df.merge(week_and_games, how='inner', on='gameId')
    print(f'{datetime.now()} Load tracking data...')
    tracking_cleaned = pd.read_csv(os.path.join(INTERIM_DATA_DIR, f'plot_tracking_data_week{week_num}.csv'))
    print(f'Merge finished start with tracking data merge: {datetime.now()}')
    merge_weekn_plays_games = tracking_cleaned.merge(games_and_plays_df, how='left', on=['gameId', 'playId'])
    print(f'Save dpi only: {datetime.now()}')
    dpi_only = merge_weekn_plays_games[merge_weekn_plays_games.isDefensivePI == True]
    dpi_only.to_csv(os.path.join(INTERIM_DATA_DIR, f'plot_dpi_only_week{week_num}.csv'), index=False)
    print(f'Save non dpi: {datetime.now()}')
    non_dpi = merge_weekn_plays_games[merge_weekn_plays_games.isDefensivePI == False]
    non_dpi.to_csv(os.path.join(INTERIM_DATA_DIR, f'plot_non_dpi_week{week_num}.csv'), index=False)
