import numpy as np
import pandas as pd
from settings import DataV3, DATA_V3_SUBVERSION
from src.features.helpers.processing_v3 import get_df_dpi_count, convert_features_to_numpy_array, \
    convert_labels_to_numpy_array, drop_records_below_att_def_d_max_threshold

data_v3 = DataV3(DATA_V3_SUBVERSION)
df = pd.read_csv(data_v3.get_step1_all_weeks_end_path(), decimal='.')

# 90% of dpi records fall in this range
ACCEPT_THRESHOLD = 5.56
df = drop_records_below_att_def_d_max_threshold(df, ACCEPT_THRESHOLD)

dpi_count = get_df_dpi_count(df)

# for now delete event
df2 = df.drop(columns=['event', 'week'])

cols_to_normalize = [
    'att_def_d', 'att_ball_d', 'def_ball_d',
    'att_s', 'def_s', 'ball_s',
    'att_a', 'def_a', 'ball_a',
    'att_o', 'def_o',
    'att_dir', 'def_dir',
]
print(f'Applying normalization for: {cols_to_normalize}')
# all min values of these columns are 0
df2[cols_to_normalize] = df2[cols_to_normalize].apply(lambda x: x / x.max())
df2.isDefensivePI = df2.isDefensivePI.astype(int)

df_labels = df2[['gameId', 'playId', 'time', 'isDefensivePI']]

cols_not_to_include = [
    'pass_arrived',
    'pass_outcome_caught',
    'tackle',
    'first_contact',
    'pass_outcome_incomplete',
    'out_of_bounds',
    'isDefensivePI'
]
df_features = df2.drop(columns=cols_not_to_include)

df_static_features = df2[[
    'time', 'gameId', 'playId', 'frameId',
    'pass_arrived',
    'pass_outcome_caught',
    'tackle',
    'first_contact',
    'pass_outcome_incomplete',
    'out_of_bounds'
]]
static_features_final = convert_features_to_numpy_array(df_static_features)
df_features_all = df2.drop(columns='isDefensivePI')

# convert features to numpy array
features_final = convert_features_to_numpy_array(df_features)
labels_final = convert_labels_to_numpy_array(df_labels)

features_all_final = convert_features_to_numpy_array(df_features_all)

labels_final = labels_final.astype(int)
labels_dpi = np.count_nonzero(labels_final == 1)

assert dpi_count == labels_dpi

np.save(data_v3.get_step2_features_path(), features_final)
np.save(data_v3.get_step2_all_features_path(), features_all_final)
np.save(data_v3.get_step2_labels_path(), labels_final)
np.save(data_v3.get_step2_static_features_path(), static_features_final)
