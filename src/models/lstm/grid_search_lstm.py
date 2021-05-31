import subprocess
from itertools import product

from settings import CURRENT_DATA_VERSION
from src.features.helpers.data_load import list_to_string

minority_class_division_factor = [1, 2, 3]
lstm_layers = [1, 2, 3]
num_cells = [64, 128, 256]

param_cartesian_product = product(
    minority_class_division_factor,
    lstm_layers,
    num_cells,
)

for i, parameters in enumerate(param_cartesian_product):
    print(parameters)
    model_name = f'lstm__weight_{parameters[0]}_layers_{parameters[1]}_cells_{parameters[2]}_data_{CURRENT_DATA_VERSION}'
    p = subprocess.check_call(
        ['python', 'train_rnn.py', model_name, list_to_string(parameters)]
    )
    print(p)
