import os
import re

import pandas as pd
from os import listdir

from settings import Models, FIGURES_DIR, STATISTICS_FILENAME
from src.features.helpers.data_load import get_n_cmd_arg

folder_result_name = get_n_cmd_arg('Enter results folder!', 1)
models_list = Models.get_models_list()


def find_csv_filenames(path_to_dir: str, suffix='.csv'):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def extract_model_info_from_filename(filename: str, pattern: str) -> str:
    try:
        return re.search(f'{pattern}(\d+)', filename).group(1)
    except AttributeError:
        pass

    try:
        return re.search(f'(\d+){pattern}', filename).group(1)
    except AttributeError:
        pass

    return ''


def get_model_result(results, dataset: str, filename: str):
    result = results[results.name == dataset]
    filename_stripped = filename.rstrip('.csv')
    weight = extract_model_info_from_filename(filename_stripped, 'weight_')
    layers = extract_model_info_from_filename(filename_stripped, 'layers_')
    cells = extract_model_info_from_filename(filename_stripped, 'cells_')
    epoch = extract_model_info_from_filename(filename_stripped, '_epoch')
    result['weight'] = weight
    result['layers'] = layers
    result['cells'] = cells
    result['epoch'] = epoch
    result.loc[:, 'name'] = filename_stripped
    return result


def merge_model_statistics(model_folder: str, model_csv: []):
    test_dataset_results = pd.DataFrame()
    val_dataset_results = pd.DataFrame()
    # Disable pandas SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    for csv_file in model_csv:
        if csv_file == STATISTICS_FILENAME:
            continue

        model_result = pd.read_csv(os.path.join(model_folder, csv_file))
        test_result = get_model_result(model_result, 'Test', csv_file)
        val_result = get_model_result(model_result, 'Validation', csv_file)
        test_dataset_results = test_dataset_results.append(test_result, ignore_index=True)
        val_dataset_results = val_dataset_results.append(val_result, ignore_index=True)
    save_statistics_path = os.path.join(model_folder, STATISTICS_FILENAME)
    with pd.ExcelWriter(save_statistics_path) as writer:
        test_dataset_results.to_excel(writer, sheet_name='Test', index=False)
        val_dataset_results.to_excel(writer, sheet_name='Validation', index=False)


for model_type in models_list:
    model_folder = os.path.join(FIGURES_DIR, model_type, folder_result_name)
    csv_files = find_csv_filenames(model_folder)
    merge_model_statistics(model_folder, csv_files)
