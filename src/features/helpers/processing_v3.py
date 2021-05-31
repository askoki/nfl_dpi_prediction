import pandas as pd
import numpy as np


def get_closest_players(situation_df: pd.DataFrame, ball_x: int, ball_y: int) -> tuple:
    home_df = situation_df[situation_df.team == 'home'].copy()
    away_df = situation_df[situation_df.team == 'away'].copy()
    home_df.loc[:, 'distance'] = np.sqrt(np.square(home_df.x - ball_x) + np.square(home_df.y - ball_y))
    away_df.loc[:, 'distance'] = np.sqrt(np.square(away_df.x - ball_x) + np.square(away_df.y - ball_y))

    p1_p = home_df.distance.min()
    p2_p = away_df.distance.min()
    p1 = home_df[home_df.distance == p1_p]
    p2 = away_df[away_df.distance == p2_p]
    return p1, p2


def get_players_and_ball_indices(df: pd.DataFrame, p1: pd.Series, p2: pd.Series) -> list or None:
    try:
        indices = df[(df.team == 'football') | (df.nflId == p1.nflId.item()) | (df.nflId == p2.nflId.item())].index
    except ValueError:
        print('get_players_and_ball_indices: no item indices found')
        return None
    return indices.tolist()


def normalize_according_to_play_direction(df: pd.DataFrame) -> pd.DataFrame:
    df_norm = df.copy()

    # normalize coordinates
    def normalize_x(row):
        if row.playDirection == 'left':
            return 120 - row.x
        return row.x

    def normalize_y(row):
        if row.playDirection == 'left':
            return 160 / 3 - row.y
        return row.y

    def normalize_degrees(play_direction: str, degree_value: int) -> int:
        if play_direction == 'left':
            opposite_value = 180 if 0 <= degree_value < 180 else -180
            return degree_value + opposite_value
        return degree_value

    def normalize_dir(row):
        return normalize_degrees(row.playDirection, row.dir)

    def normalize_o(row):
        return normalize_degrees(row.playDirection, row.o)

    df_norm.x = df_norm.apply(lambda row: normalize_x(row), axis=1)
    df_norm.y = df_norm.apply(lambda row: normalize_y(row), axis=1)
    df_norm.o = df_norm.apply(lambda row: normalize_o(row), axis=1)
    df_norm.dir = df_norm.apply(lambda row: normalize_dir(row), axis=1)
    return df_norm


def calculate_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    return round(np.sqrt(np.square(x1 - x2) + np.square(y1 - y2)), 2)


def split_binary_classes(x_features: np.array, y_labels: np.array, positive_class_label=1):
    x_positive = []
    x_negative = []
    y_positive = []
    y_negative = []
    for x, y in zip(x_features, y_labels):
        if y == positive_class_label:
            x_positive.append(x)
            y_positive.append(y)
        else:
            x_negative.append(x)
            y_negative.append(y)

    x_positive = np.asarray(x_positive)
    x_negative = np.asarray(x_negative)
    y_positive = np.asarray(y_positive)
    y_negative = np.asarray(y_negative)
    return x_positive, x_negative, y_positive, y_negative


def deep_multiplicate_np_array(input_array: np.array, factor: int) -> np.array:
    result = input_array.copy()
    for i in range(factor - 1):
        result = np.concatenate((result, input_array))
    return result


def balance_binary_dataset(x_positive: np.array, x_negative: np.array, y_positive: np.array, y_negative: np.array,
                           multiplication_factor=1):
    """
    x_positive: DPI class features
    x_negative: non-DPI class features
    y_positive: DPI class labels
    y_negative: non-DPI class labels
    """
    num_positive = x_positive.shape[0] * multiplication_factor
    num_negative = x_negative.shape[0]
    random_negative_samples = np.random.choice(num_negative, size=num_positive, replace=True)
    random_negative_samples = np.sort(random_negative_samples)

    x_negative_rand = []
    y_negative_rand = []
    for index in random_negative_samples:
        x_negative_rand.append(x_negative[index])
        y_negative_rand.append(y_negative[index])
    x_negative_rand = np.asarray(x_negative_rand)
    y_negative_rand = np.asarray(y_negative_rand)

    x_positive_multi = deep_multiplicate_np_array(x_positive, multiplication_factor)
    y_positive_multi = deep_multiplicate_np_array(y_positive, multiplication_factor)
    x_balanced = np.concatenate((x_positive_multi, x_negative_rand))
    y_balanced = np.concatenate((y_positive_multi, y_negative_rand))
    return x_balanced, y_balanced


def convert_features_to_numpy_array(df_features: pd.DataFrame) -> np.array:
    grouped = df_features.groupby(['gameId', 'playId'])
    np_array = []
    for name, group in grouped:
        clean_g = group.drop(columns=['time', 'gameId', 'playId', 'frameId'])
        np_array.append(clean_g.values)
    return np.asarray(np_array, dtype=object)


def convert_labels_to_numpy_array(df_labels: pd.DataFrame) -> np.array:
    grouped = df_labels.groupby(['gameId', 'playId'])
    return grouped.isDefensivePI.agg('first').values


def get_df_dpi_count(df: pd.DataFrame) -> int:
    df_grouped = df.groupby(['gameId', 'playId'])
    dpi_count = 0
    print(f'Number of groups/plays {len(df_grouped)}')
    for name, group in df_grouped:
        if any(group.isDefensivePI == True):
            dpi_count += 1
    print(f'Cumulative dpi: {dpi_count}')
    return dpi_count


def drop_records_below_att_def_d_max_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    gb = df.groupby(['gameId', 'playId'])
    num_records = len(gb)
    new_df = pd.DataFrame()
    current_record = 0
    for name, group in gb:
        current_record += 1
        print(f'Creating dataset week {current_record} / {num_records}', end='\r')

        if group.att_def_d.max() < threshold:
            new_df = new_df.append(group, ignore_index=True)
    return new_df


def check_group_event(event_list: list, event_name: str) -> bool:
    return event_name in event_list
