import pandas as pd
from datetime import datetime
from settings import DataV3, DATA_V3_SUBVERSION

WEEK_MIN = 2
WEEK_MAX = 18
print(f'Start time: {datetime.now().strftime("%H:%M:%S")}')
print('Loading week 1')

data_v3 = DataV3(DATA_V3_SUBVERSION)
df = pd.read_csv(data_v3.get_step1_end_path(1))
for i in range(WEEK_MIN, WEEK_MAX):
    print(f'Week {i}: {datetime.now().strftime("%H:%M:%S")}')
    df2 = pd.read_csv(data_v3.get_step1_end_path(i))
    print(f'Week {i} loaded: {datetime.now().strftime("%H:%M:%S")}')
    df = pd.concat([df, df2])
    print(f'Merged')
print(f'Saving')
df.to_csv(data_v3.get_step1_all_weeks_end_path(), index=False)
