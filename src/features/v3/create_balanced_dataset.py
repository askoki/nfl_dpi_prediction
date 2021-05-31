import os
import numpy as np
from datetime import datetime
from settings import PROCESSED_DATA_DIR, CURRENT_DATA_VERSION
from src.features.helpers.data_load import get_train_val_test_data
from src.features.helpers.processing import print_num_positives
from src.features.helpers.processing_v3 import split_binary_classes, balance_binary_dataset

x_train, y_train, x_val, y_val, ignore_x_test, ignore_y_test = get_train_val_test_data(version='v2')

print_num_positives(y_train)
print_num_positives(y_val)


def undersample_dataset(x_features, y_labels, name):
    x_dpi, x_non_dpi, y_dpi, y_non_dpi = split_binary_classes(x_features, y_labels)

    x_balanced, y_balanced = balance_binary_dataset(x_dpi, x_non_dpi, y_dpi, y_non_dpi)
    print('After balancing...')
    print_num_positives(y_balanced)
    np.save(os.path.join(PROCESSED_DATA_DIR, f'x_{name}_{CURRENT_DATA_VERSION}_balanced.npy'), x_balanced)
    np.save(os.path.join(PROCESSED_DATA_DIR, f'y_{name}_{CURRENT_DATA_VERSION}_balanced.npy'), np.array(y_balanced))


def oversample_dataset(x_features, y_labels, name, oversample_factor=6):
    x_dpi, x_non_dpi, y_dpi, y_non_dpi = split_binary_classes(x_features, y_labels)
    x_balanced, y_balanced = balance_binary_dataset(x_dpi, x_non_dpi, y_dpi, y_non_dpi, oversample_factor)
    print('After balancing...')
    print_num_positives(y_balanced)
    np.save(os.path.join(PROCESSED_DATA_DIR, f'x_{name}_{CURRENT_DATA_VERSION}_balanced.npy'), x_balanced)
    np.save(os.path.join(PROCESSED_DATA_DIR, f'y_{name}_{CURRENT_DATA_VERSION}_balanced.npy'), np.array(y_balanced))


# undersample_dataset(x_train, y_train, 'train')
# undersample_dataset(x_val, y_val, 'val')
oversample_dataset(x_train, y_train, 'train')
oversample_dataset(x_val, y_val, 'val')

print(f'{datetime.now().strftime("%H:%M:%S")} End data loading...')
