import gc
import subprocess
from datetime import datetime

WEEK_MIN = 1
WEEK_MAX = 17
print(f'Start time: {datetime.now().strftime("%H:%M:%S")}')
for i in range(WEEK_MAX):
    print(f'Starting week {i + 1}: {datetime.now().strftime("%H:%M:%S")}')
    p_level = subprocess.check_call(
        ['python', '1_preprocess_for_plot.py', str(i + 1)]
    )
    print(f'Week {i + 1} is processed {p_level == 0}, {datetime.now().strftime("%H:%M:%S")}')
    # Clean memory
    gc.collect()
print(f'Total end time: {datetime.now().strftime("%H:%M:%S")}')
