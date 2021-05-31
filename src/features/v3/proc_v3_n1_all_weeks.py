import gc
import subprocess
from datetime import datetime

WEEK_MIN = 1
WEEK_MAX = 18
print(f'Start time: {datetime.now().strftime("%H:%M:%S")}')
for i in range(WEEK_MIN, WEEK_MAX):
    print(f'Starting week {i}: {datetime.now().strftime("%H:%M:%S")}')

    p_level = subprocess.check_call(
        ['python', 'proc_v3_n1_calc_distance.py', str(i)]
    )
    print(f'Week {i} is processed {p_level == 0}, {datetime.now().strftime("%H:%M:%S")}')
    # Clean memory
    gc.collect()
print(f'Total end time: {datetime.now().strftime("%H:%M:%S")}')
