import os
import subprocess
from datetime import datetime

print(f'Start step 1 - preprocess for plot, {datetime.now().strftime("%H:%M:%S")}')
p_level = subprocess.check_call(
    ['python', '2_plot_preprocess_all.py'],
    cwd=os.path.join('..', 'features', 'plot')
)
print(f'\n\nIs successful {p_level == 0}, {datetime.now().strftime("%H:%M:%S")}')

print(f'Start step 2 - merge tracking with play, {datetime.now().strftime("%H:%M:%S")}')
p_level = subprocess.check_call(
    ['python', '3_merge_tracking_with_play_for_plot.py'],
    cwd=os.path.join('..', 'features', 'plot')
)
print(f'Is successful {p_level == 0}, {datetime.now().strftime("%H:%M:%S")}')
