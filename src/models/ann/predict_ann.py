import os
import subprocess
from settings import Models
from src.features.helpers.data_load import get_model_name

ML_MODEL = get_model_name()

is_evaluation_successful = subprocess.check_call(
    ['python', 'evaluate_model.py', ML_MODEL, Models.ANN.value],
    cwd=os.path.join('..', '..', 'features', 'helpers')
)
print(f'Evaluation is successful: {is_evaluation_successful == 0}')
