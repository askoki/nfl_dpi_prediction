import os
import sys
import numpy as np
from datetime import datetime
from settings import CLEANED_DATA_DIR
from sklearn.model_selection import train_test_split

from src.features.helpers.data_load import save_train_val_test_data
from src.features.helpers.processing import format_x

x_name = sys.argv[1]
y_name = sys.argv[2]

try:
    version = sys.argv[3]
except IndexError:
    version = 'v1'
    print(f'### Setting default version: {version}')

X = np.load(os.path.join(CLEANED_DATA_DIR, f'{x_name}'), allow_pickle=True)
y = np.load(os.path.join(CLEANED_DATA_DIR, f'{y_name}'), allow_pickle=True)

# Convert labels to int
y_num = y.astype(int)

# num_positives = y_train_num.count(1)
num_positives = np.count_nonzero(y_num == 1)

# num_negatives = y_train.count(0)
num_negatives = np.count_nonzero(y_num == 0)

total = len(y_num)

print(
    f'Examples:\nTotal: {total}\n Positive: {num_positives} ({round(100 * num_positives / total, 2)}% of total)\n'
)

print(f'Formatting x train: {datetime.now().strftime("%H:%M:%S")}')
X = format_x(X)
print(f'{datetime.now().strftime("%H:%M:%S")} Split train and test set')
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y_num,
    test_size=0.3,
    stratify=y_num,
    random_state=42
)

print("Split train and validation set")
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

print("Shape train features", np.shape(x_train))
print("Shape val features", np.shape(x_val))
print("Shape test features", np.shape(x_test))

save_train_val_test_data(
    x_train, y_train,
    x_val, y_val,
    x_test, y_test,
    version
)
