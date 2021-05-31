import os
import pandas as pd
from settings import Models, FIGURES_DIR, STATISTICS_FILENAME
from src.features.helpers.data_load import get_n_cmd_arg

folder_result_name = get_n_cmd_arg('Enter results folder!', 1)
models_list = Models.get_models_list()

cumulative_results = pd.DataFrame()
# Disable pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None
cell_options = [8, 64, 128]
for model_type in models_list:
    try:
        test_statistics = pd.read_excel(
            os.path.join(FIGURES_DIR, model_type, folder_result_name, STATISTICS_FILENAME),
            sheet_name='Test',
            engine='openpyxl',
        )
        val_statistics = pd.read_excel(
            os.path.join(FIGURES_DIR, model_type, folder_result_name, STATISTICS_FILENAME),
            sheet_name='Validation',
            engine='openpyxl',
        )
    except FileNotFoundError:
        continue

    for cell in cell_options:
        cell_result = val_statistics[val_statistics.cells == cell]
        test_cells = test_statistics[test_statistics.cells == cell]

        # calculate best on validation set
        max_recall = cell_result.recall.max()
        cell_result = cell_result[cell_result.recall == max_recall]
        # if multiple models have the same recall then pick with highest precision
        if cell_result.shape[0] > 1:
            max_precision = cell_result.precision.max()
            # if recall and precision are the same pick the first one
            cell_result = cell_result[cell_result.precision == max_precision].head(1)
        cell_result['model'] = model_type
        cell_result['cells'] = cell

        # add test data
        test_result = test_cells[test_cells.name == cell_result.name.unique()[0]]
        cell_result['test_f1'] = test_result.f1
        cell_result['test_accuracy'] = test_result.accuracy
        cell_result['test_auc'] = test_result.auc
        cell_result['test_precision'] = test_result.precision
        cell_result['test_recall'] = test_result.recall
        cell_result = cell_result[[
            'model', 'cells', 'f1', 'test_f1',
            'accuracy', 'test_accuracy', 'auc', 'test_auc', 'precision', 'test_precision',
            'recall', 'test_recall', 'name']]
        cumulative_results = cumulative_results.append(cell_result, ignore_index=True)

cumulative_results.to_csv(os.path.join(FIGURES_DIR, f'Cumulative_statistics_{folder_result_name}.csv'), index=False)
