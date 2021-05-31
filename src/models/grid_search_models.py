import os
import subprocess
from itertools import product
from settings import CURRENT_DATA_VERSION, Models, DEFAULT_EPOCHS, DATA_V3_SUBVERSION
from src.features.helpers.data_load import list_to_string, get_n_cmd_arg
from datetime import datetime

# val_f1_score, val_recall
monitor = get_n_cmd_arg('Enter monitor value!', 1)

minority_class_division_factor = [1]
lstm_layers = [1]
num_cells = [8, 64, 128]
monitor = [monitor]
models_list = Models.get_models_list()

param_cartesian_product = product(
    minority_class_division_factor,
    lstm_layers,
    num_cells,
    monitor,
    models_list
)

REPEAT_FACTOR = 5

for i, parameters in enumerate(param_cartesian_product):
    print(parameters)
    model_type = parameters[-1]
    parameters = parameters[:-1]
    for step in range(REPEAT_FACTOR):
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        model_name = f'{timestamp}_weight_{parameters[0]}_layers_{parameters[1]}_cells_{parameters[2]}_data_{CURRENT_DATA_VERSION}_{DATA_V3_SUBVERSION}_{DEFAULT_EPOCHS}_epoch_step{step + 1}'
        is_train_successful = subprocess.check_call(
            ['python', os.path.join(model_type, 'train.py'), model_name, list_to_string(parameters)]
        )
        print(f'Training is successful: {is_train_successful == 0}')
        is_evaluation_successful = subprocess.check_call(
            ['python', 'evaluate_model.py', model_name, model_type],
            cwd=os.path.join('..', 'features', 'helpers')
        )
        print(f'Evaluation is successful: {is_evaluation_successful == 0}')
