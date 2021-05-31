import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()
CURRENT_DATA_VERSION = 'v3'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
INTERIM_IMG_DATA_DIR = os.path.join(INTERIM_DATA_DIR, 'images')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
PROCESSED_IMG_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'images')
CLEANED_DATA_DIR = os.path.join(DATA_DIR, 'cleaned')
FEATURES_DATA_DIR = os.path.join(DATA_DIR, 'src', 'features')
FIGURES_DIR = os.path.join(ROOT_DIR, 'reports', 'figures')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 128

STATISTICS_FILENAME = 'Test_model_statistics.xlsx'


class Models(Enum):
    LSTM = 'lstm'
    ANN = 'ann'
    GRU = 'gru'
    MLSTM = 'mlstm'

    @classmethod
    def get_models_list(cls):
        return list(map(lambda c: c.value, cls))


# add new features in subversion 4 a, and events
DATA_V3_SUBVERSION = 4


class DataHelper:

    def __init__(self, version: int, subversion: int):
        self.version = version
        self.subversion = subversion

    def get_step_1_save_path(self, week_num):
        return os.path.join(
            INTERIM_DATA_DIR,
            f'processing_v{self.version}_{self.subversion}_clean_dataframe_week{week_num}.csv'
        )

    def get_step_2_save_path(self):
        return os.path.join(
            INTERIM_DATA_DIR,
            f'processing_v{self.version}_{self.subversion}_merge_all_df.csv'
        )


class DataV3:

    def __init__(self, version: str):
        self.subversion = version

    def get_step1_checkpoint_path(self, week_num: int) -> str:
        return os.path.join(INTERIM_DATA_DIR, f'processing_v3_2_clean_dataframe_week{week_num}.csv')

    def get_step1_end_path(self, week_num: int) -> str:
        return os.path.join(INTERIM_DATA_DIR, f'processing_v3_{self.subversion}_dataframe_week{week_num}.csv')

    def get_step1_all_weeks_end_path(self) -> str:
        return os.path.join(INTERIM_DATA_DIR, f'processing_v3_{self.subversion}_step_1_all_weeks.csv')

    def get_step2_features_path(self) -> str:
        return os.path.join(CLEANED_DATA_DIR, self.get_features_short())

    def get_step2_labels_path(self) -> str:
        return os.path.join(CLEANED_DATA_DIR, self.get_labels_short())

    def get_step2_static_features_path(self) -> str:
        return os.path.join(CLEANED_DATA_DIR, f'features_static_v3_{self.subversion}_all_weeks.npy')

    def get_step2_all_features_path(self) -> str:
        return os.path.join(CLEANED_DATA_DIR, self.get_all_features_short())

    def get_all_features_short(self) -> str:
        return f'all_features_v3_{self.subversion}_all_weeks.npy'

    def get_features_short(self) -> str:
        return f'features_v3_{self.subversion}_all_weeks.npy'

    def get_labels_short(self) -> str:
        return f'labels_v3_{self.subversion}_all_weeks.npy'
