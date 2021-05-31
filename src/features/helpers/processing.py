import pandas as pd
import numpy as np
from operator import itemgetter
from datetime import datetime
from keras_preprocessing.sequence import pad_sequences


def format_x(input_format: np.array) -> np.array:
    for i in range(input_format.shape[0]):
        input_format[i] = np.array(np.nan_to_num(input_format[i].astype(np.float32)))
    input_format = pad_sequences(input_format, dtype='float32', padding='pre', value=0.0)
    print(np.shape(input_format))
    return input_format


def add_missing_timestamp_values(df: pd.DataFrame) -> pd.DataFrame:
    frame2time = df.groupby(['gameId', 'time', 'frameId'])
    prev_frame_id = 0
    coef = 1
    prev_timestamp = pd.Timestamp('2018-01-01T00:00:00.000Z')
    timestamp_delta = pd.Timedelta(np.timedelta64(10, 'ms'))

    print('Start record processing. Differentiate timestamps that have multiple records...')
    print(f'Time: {datetime.now().strftime("%H:%M:%S")}')
    for name, group in frame2time:
        game_id, timestamp, frame_id = name
        timestamp = pd.Timestamp(timestamp)

        if frame_id >= prev_frame_id and timestamp == prev_timestamp:
            prev_frame_id = frame_id
            new_timestamp = timestamp + (timestamp_delta * coef)
            coef += 1
            df['time'].mask(
                (df['gameId'] == game_id) &
                (df['time'] == timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z') &
                (df['frameId'] == frame_id),
                new_timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
                inplace=True
            )
        else:
            # new play
            coef = 1
            prev_frame_id = frame_id
            prev_timestamp = timestamp
    print(f'End record processing: {datetime.now().strftime("%H:%M:%S")}')
    return df


def normalize_play_direction(df: pd.DataFrame) -> pd.DataFrame:
    # normalize coordinates
    def normalize_x(row):
        if row.playDirection == 'left':
            return 120 - row.x
        return row.x

    def normalize_y(row):
        if row.playDirection == 'left':
            return 160 / 3 - row.y
        return row.y

    df.x = df.apply(lambda row: normalize_x(row), axis=1)
    df.y = df.apply(lambda row: normalize_y(row), axis=1)
    return df


SPEED_MAX_THRESHOLD = 43
ACCELERATION_MAX_THRESHOLD = 71
DISTANCE_MAX_THRESHOLD = 13


def normalize_and_discard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numeric values between 0 and 1 and discard records that are out of bounds.
    """
    # ## 2. Discard values out of range of x and y
    df_cleaned = df[(df.x >= 0) & (df.x <= 120) & (df.y >= 0) & (df.y <= (160 / 3))]
    print(f'Shape difference {df.shape[0] - df_cleaned.shape[0]}')

    # ## 3. Normalize x, y , s, a, dis, o, dir on scale 0-1
    # thresholds are determined by examining data from all weeks

    df_cleaned.x = df_cleaned.x / df.x.max()
    df_cleaned.y = df_cleaned.y / df.y.max()
    df_cleaned.s = df_cleaned.s / SPEED_MAX_THRESHOLD
    df_cleaned.a = df_cleaned.a / ACCELERATION_MAX_THRESHOLD
    df_cleaned.dis = df_cleaned.dis / DISTANCE_MAX_THRESHOLD
    df_cleaned.o = df_cleaned.o / 360
    df_cleaned.dir = df_cleaned.dir / 360

    df_n2 = df_cleaned[[
        'time', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'event', 'frameId', 'team', 'gameId',
        'playId', 'quarter', 'homeHasPossession',
        'down', 'playType', 'defendersInTheBox',
        'numberOfPassRushers', 'passResult', 'isDefensivePI'
    ]]

    df_n2.quarter /= 5.0  # max quarters
    df_n2.down /= 4.0  # max quarters
    df_n2.defendersInTheBox /= 11.0
    df_n2.numberOfPassRushers /= 11.0

    return df_n2


def convert_df_rows2columns(df: pd.DataFrame) -> pd.DataFrame:
    print('Transforming rows to cols...')

    def preprocess_df_columns() -> pd.Series:
        occurrence_list = df.sort_values(['gameId', 'time', 'frameId']).groupby(
            ['gameId', 'time', 'frameId', 'team']).team.cumcount().add(1)
        labels_list = df.sort_values(['gameId', 'time', 'frameId']).groupby(
            ['gameId', 'time', 'frameId']).team.apply(list)
        flat_list = np.array([item for sublist in labels_list for item in sublist])
        merge_list = np.column_stack((occurrence_list, flat_list))
        col_list = [f'{row[1]}_{row[0]}' for row in merge_list]
        return pd.Series(col_list)

    df_converted = pd.pivot_table(
        df,
        index=['time', 'playId', 'gameId'],
        columns=preprocess_df_columns(),
        values=['x', 'y', 's', 'a', 'dis', 'o', 'dir'],
        aggfunc='sum',
        fill_value=0.0
    )

    df_converted.columns = df_converted.columns.map('{0[0]}_{0[1]}'.format)
    return df_converted


def convert_df_to_numpy_array(df: pd.DataFrame, groupby_id: list, skip_last_list_wrapper=False) -> np.array:
    group = df.groupby(groupby_id).cumcount()

    if skip_last_list_wrapper:
        return (
            df.set_index([*groupby_id, group])
                .unstack(fill_value=None)
                .stack().groupby(level=0).agg({'time': 'first', 'isDefensivePI': 'first'})
                .apply(lambda x: x.values.tolist())
                .values.tolist()
        )

    return (
        df.set_index([*groupby_id, group])
            .unstack(fill_value=None)
            .stack().groupby(level=0)
            .apply(lambda x: x.values.tolist())
            .values.tolist()
    )


def remove_col_from_inner_list(input_list: np.array, column_index: int) -> np.array:
    for i, play in enumerate(input_list):
        input_list[i] = np.delete(input_list[i], column_index, axis=1)
    return input_list


def sort_by_timestamp_and_remove_timestamp(features_list: np.array, labels_list: np.array) -> (np.array, np.array):
    def sort_list_by_col_index(list2sort: np.array, column_index: int, deep_sort=True) -> np.array:
        if deep_sort:
            for i, play in enumerate(list2sort):
                list2sort[i] = sorted(play, key=itemgetter(column_index))
            return list2sort
        return sorted(list2sort, key=itemgetter(column_index))

    def promote_list_item(input_list: np.array, element2keep_index: int) -> np.array:
        """
        Keep only one element from the list and remove list wrapper.
        Usefull when using list of lists
        """
        # remove list wrapper and col (keep only label)
        new_list = []
        for i, row in enumerate(input_list):
            new_list.append(input_list[i][element2keep_index])
        return new_list

    print("Sort data by timestamp...")
    features_sorted = sort_list_by_col_index(features_list, 0)
    # when it is sorted remove timestamp from array
    features_final = remove_col_from_inner_list(features_sorted, 0)

    labels_sorted = sort_list_by_col_index(labels_list, 0, deep_sort=False)
    # when it is sorted remove timestamp from array
    labels_final = promote_list_item(labels_sorted, 1)
    return features_final, labels_final


def print_num_positives(y_labels: np.array) -> None:
    y_train_num = y_labels.astype(int)
    num_positives = np.count_nonzero(y_train_num == 1)

    total = len(y_labels)
    print(
        f'Examples:\nTotal: {total}\n Positive: {num_positives} ({round(100 * num_positives / total, 2)}% of total)\n'
    )
