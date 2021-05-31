import os
import tensorflow_addons as tfa
from tensorflow.python.keras.metrics import PrecisionAtRecall

from settings import FIGURES_DIR
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def get_f1() -> tfa.metrics.F1Score:
    return tfa.metrics.F1Score(
        num_classes=1,
        average='macro',
        threshold=0.5
    )


def get_model_metrics() -> list:
    return [
        PrecisionAtRecall(0.9),
        get_f1(),
        'BinaryAccuracy',
        'Precision',
        'Recall',
        'AUC'
    ]


def get_model_prediction_results(predicted_classes, true_classes, name=None) -> dict:
    if name:
        return {
            'name': name,
            'accuracy': accuracy_score(predicted_classes, true_classes),
            'f1': f1_score(predicted_classes, true_classes),
            'recall': recall_score(predicted_classes, true_classes),
            'precision': precision_score(predicted_classes, true_classes)
        }
    return {
        'accuracy': accuracy_score(predicted_classes, true_classes),
        'f1': f1_score(predicted_classes, true_classes),
        'recall': recall_score(predicted_classes, true_classes),
        'precision': precision_score(predicted_classes, true_classes)
    }


def write_train_val_test_metrics(model_name: str, model_dir: str, train: dict, val: dict, test: dict) -> None:
    try:
        f = open(os.path.join(FIGURES_DIR, model_dir, f'{model_name}__metrics.txt'), 'a+')
    except FileNotFoundError:
        os.mkdir(os.path.join(FIGURES_DIR, model_dir))
        f = open(os.path.join(FIGURES_DIR, model_dir, f'{model_name}__metrics.txt'), 'a+')
    for result in [train, val, test]:
        for label, value in result.items():
            f.write(f'{label}: {value}\n')
        f.write('\n\n')
    f.close()
