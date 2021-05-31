import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras_preprocessing.image import ImageDataGenerator

from settings import PROCESSED_DATA_DIR, PROCESSED_IMG_DATA_DIR


def save_train_val_test_data(x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array, x_test: np.array,
                             y_test: np.array, version='v1') -> None:
    print(f'{datetime.now().strftime("%H:%M:%S")} Saving training labels and features...')
    np.save(os.path.join(PROCESSED_DATA_DIR, f'x_train_{version}.npy'), x_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, f'y_train_{version}.npy'), y_train)

    print(f'{datetime.now().strftime("%H:%M:%S")} Saving validation labels and features...')
    np.save(os.path.join(PROCESSED_DATA_DIR, f'x_val_{version}.npy'), x_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, f'y_val_{version}.npy'), y_val)

    print(f'{datetime.now().strftime("%H:%M:%S")} Saving test labels and features...')
    np.save(os.path.join(PROCESSED_DATA_DIR, f'x_test_{version}.npy'), x_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, f'y_test_{version}.npy'), y_test)

    return


def get_and_print_distribution(labels: np.array) -> tuple:
    num_positives = np.count_nonzero(labels == 1)
    num_negatives = np.count_nonzero(labels == 0)
    total = len(labels)

    positives_percentage = round(100 * num_positives / total, 2)
    print(
        f'Examples:\n Total: {total}\n Positive: {num_positives} '
        f'({positives_percentage}% of total)\n'
    )
    return num_negatives, num_positives


def get_img_train_val_test_data():
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
    )

    # load and iterate training dataset
    train_it = datagen.flow_from_directory(
        os.path.join(PROCESSED_IMG_DATA_DIR, 'train'),
        classes=['non_dpi', 'dpi'],
        color_mode='rgb',
        class_mode='binary',
        batch_size=32
    )
    # load and iterate validation dataset
    val_it = datagen.flow_from_directory(
        os.path.join(PROCESSED_IMG_DATA_DIR, 'val'),
        classes=['non_dpi', 'dpi'],
        color_mode='rgb',
        class_mode='binary',
        batch_size=16
    )
    # load and iterate test dataset
    test_it = datagen.flow_from_directory(
        os.path.join(PROCESSED_IMG_DATA_DIR, 'test'),
        classes=['non_dpi', 'dpi'],
        color_mode='rgb',
        class_mode='binary',
        batch_size=16
    )

    return train_it, train_it.classes, val_it, val_it.classes, test_it, test_it.classes


def get_train_val_test_data(train_source='', version='v1', subversion='3'):
    print(f'{datetime.now().strftime("%H:%M:%S")} Start data loading...')
    subversion = '' if not subversion else f'_{subversion}'
    x_train = np.load(
        os.path.join(PROCESSED_DATA_DIR, f'x_train_{version}{subversion}{train_source}.npy'), allow_pickle=True
    )
    y_train = np.load(
        os.path.join(PROCESSED_DATA_DIR, f'y_train_{version}{subversion}{train_source}.npy'), allow_pickle=True
    )

    x_val = np.load(
        os.path.join(PROCESSED_DATA_DIR, f'x_val_{version}{subversion}{train_source}.npy'), allow_pickle=True
    )
    y_val = np.load(
        os.path.join(PROCESSED_DATA_DIR, f'y_val_{version}{subversion}{train_source}.npy'), allow_pickle=True
    )

    x_test = np.load(os.path.join(PROCESSED_DATA_DIR, f'x_test_{version}{subversion}.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, f'y_test_{version}{subversion}.npy'), allow_pickle=True)
    print(f'{datetime.now().strftime("%H:%M:%S")} End data loading...')

    # UNITS = 2
    # y_train[1] = 1
    # y_val[1] = 1
    # y_test[1] = 1
    # return x_train[:UNITS], y_train[:UNITS], x_val[:UNITS], y_val[:UNITS], x_test[:UNITS], y_test[:UNITS]
    return x_train, y_train, x_val, y_val, x_test, y_test


def string_to_list(string_list: str) -> list or tuple:
    """
    string_list -> is a string in a list shape
    e.g. '[1,2,False,String]'
    returns list of parameters
    """
    parameters = []
    string_parameters = string_list.strip('[]').split(',')
    for param in string_parameters:
        try:
            parameters.append(int(param))
        except ValueError:
            # If object is not integer try to convert it to boolean.
            if param == 'True' or param == 'False':
                parameters.append(param == 'True')
            else:
                # If object is not boolean then append string.
                parameters.append(param)
    return parameters


def list_to_string(list_to_convert: list or tuple) -> str:
    """
    list_to_convert -> List that should be converted into string
    e.g. '[1,2,False,String]'
    returns string
    """
    string_list = '['
    for j in range(len(list_to_convert)):
        string_list += str(list_to_convert[j]) + ','
    # remove last ','
    string_list = string_list[:-1]
    string_list += ']'
    return string_list


def get_model_name() -> str:
    return get_n_cmd_arg('Enter model name!', 1)


def get_n_cmd_arg(error_message: str, arg_index: int) -> str:
    try:
        model_name = str(sys.argv[arg_index])
    except IndexError:
        print(error_message)
        sys.exit()
    return model_name


def get_model_args():
    try:
        return string_to_list(sys.argv[2])
    except IndexError:
        print('Enter model parameters!')
        sys.exit()


def get_model_type() -> str:
    return get_n_cmd_arg('Enter model type!', 2)


def get_train_source(arg_num=3):
    try:
        bool(sys.argv[arg_num])
        source_balanced = True
    except IndexError:
        print('Using imbalanced dataset')
        source_balanced = False

    train_source = '_balanced' if source_balanced else ''
    print(f'\n### Using option: {train_source}\n\n')
    return train_source
