import os
import subprocess
from datetime import datetime

from settings import DataV3, DATA_V3_SUBVERSION, CURRENT_DATA_VERSION

data_v3 = DataV3(DATA_V3_SUBVERSION)
print(f'Start step 1 - preprocess and calculate distance all weeks, {datetime.now().strftime("%H:%M:%S")}')
p_level = subprocess.check_call(
    ['python', 'proc_v3_n1_all_weeks.py'],
    cwd=os.path.join('..', 'features', 'v3')
)
print(f'\n\nIs successful {p_level == 0}, {datetime.now().strftime("%H:%M:%S")}')

print(f'Start step 2 - merge all, {datetime .now().strftime("%H:%M:%S")}')
p_level = subprocess.check_call(
    ['python', 'proc_v3_n2_merge_all.py'],
    cwd=os.path.join('..', 'features', 'v3')
)
print(f'Is successful {p_level == 0}, {datetime.now().strftime("%H:%M:%S")}')

print(f'Start step 3 - split features and labels, {datetime.now().strftime("%H:%M:%S")}')
p_level = subprocess.check_call(
    ['python', 'proc_v3_n3_split_features_n_labels.py'],
    cwd=os.path.join('..', 'features', 'v3')
)
print(f'Is successful {p_level == 0}, {datetime.now().strftime("%H:%M:%S")}')

print(f'Start step 4 - train test val split, {datetime.now().strftime("%H:%M:%S")}')
p_level = subprocess.check_call(
    ['python', 'train_test_val_split.py', data_v3.get_all_features_short(), data_v3.get_labels_short(),
     f'{CURRENT_DATA_VERSION}_{DATA_V3_SUBVERSION}'],
    cwd=os.path.join('..', 'features', 'helpers')
)
print(f'Is successful {p_level == 0}, {datetime.now().strftime("%H:%M:%S")}')
