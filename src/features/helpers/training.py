import os
import csv
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import ceil
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Masking, TimeDistributed, RepeatVector
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from settings import FIGURES_DIR, MODELS_DIR
from src.features.helpers.evaluation import ResultMetric


def get_early_stopping_definition(monitor='loss', mode='max', patience='3'):
    return EarlyStopping(
        monitor=monitor,
        min_delta=0.01,
        verbose=1,
        patience=patience,
        mode=mode,
        restore_best_weights=True
    )


def check_and_create_dir_path(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)
    return


def get_model_path(model_folder: str, model_name: str):
    check_and_create_dir_path(os.path.join(MODELS_DIR, model_folder))
    return os.path.join(MODELS_DIR, model_folder, f"{model_name}_{model_folder}.h5")


def get_save_best_model(model_folder: str, model_name: str, mode='max', monitor='val_f1_score'):
    return ModelCheckpoint(
        get_model_path(model_folder, model_name),
        verbose=1,
        mode=mode,
        monitor=monitor,
        save_best_only=True
    )


def make_3_layer_autoencoder(input_shape=None):
    auto_encoder = Sequential()
    auto_encoder.add(Masking(mask_value=0.0, input_shape=(input_shape[1], input_shape[2])))
    auto_encoder.add(
        LSTM(256, return_sequences=True)
    )
    auto_encoder.add(
        LSTM(128, return_sequences=True)
    )
    auto_encoder.add(LSTM(64, return_sequences=False))
    auto_encoder.add(RepeatVector(n=input_shape[1]))
    auto_encoder.add(LSTM(64, return_sequences=True))
    auto_encoder.add(LSTM(128, return_sequences=True))
    auto_encoder.add(
        LSTM(256, return_sequences=True)
    )
    auto_encoder.add(TimeDistributed(Dense(units=input_shape[2])))
    auto_encoder.compile(
        loss='mse',
        optimizer='adam'
    )

    return auto_encoder


def make_lstm_model(input_dim=None, use_focal_loss=True):
    model = Sequential()

    model.add(Masking(mask_value=0.0, input_shape=input_dim))
    model.add(
        Bidirectional(
            LSTM(128, return_sequences=True)
        )
    )
    model.add(
        Bidirectional(
            LSTM(128, return_sequences=False)
        )
    )

    model.add(
        Dense(1, activation='sigmoid')
    )

    adam = tf.optimizers.Adam(clipvalue=0.5, learning_rate=0.0001)
    model.compile(
        optimizer=adam,
        loss=focal_loss if use_focal_loss else 'binary_crossentropy',
        metrics='accuracy'
    )

    return model


def focal_loss(y_true, y_pred):
    # made according to: https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758
    gamma = 2.0
    alpha = 1.0
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def plot_training_validation_loss(keras_history, save_filename: str, model_type: str):
    plt.plot(keras_history.history['loss'])
    plt.plot(keras_history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    dir_path = os.path.join(FIGURES_DIR, model_type)
    check_and_create_dir_path(dir_path)
    plt.savefig(os.path.join(dir_path, f'{save_filename}_train_val_loss.png'))


def plot_confusion_matrix(predicted_classes: [], true_classes: [], classes, matrix_name: str,
                          model_name: str, model_type: str, normalize=False, title='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    model_name -> name of the folder where statistics will be saved
    (defined according to ML_MODELS in settings.py)
    Code is taken from scikit-learn.org example.
    """
    cm = confusion_matrix(true_classes, predicted_classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig = plt.figure("Confusion matrix")
    fig.set_size_inches(12, 12)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    dir_path = os.path.join(FIGURES_DIR, model_type)
    check_and_create_dir_path(dir_path)
    plt.savefig(
        os.path.join(dir_path, f'{model_name}_confusion_matrix_{matrix_name}.png'),
        format='png',
        dpi=500
    )
    plt.close()


def write_to_csv(train_val_test_dict: list, dir_path: str, model_name: str):
    headers = ['name', 'f1', 'accuracy', 'auc', 'precision', 'recall']

    f1_name = ceil(train_val_test_dict[2]['f1'] * 100)
    csv_path = os.path.join(dir_path, f'{model_name}_metrics_test_f1_{f1_name}.csv')
    try:
        with open(csv_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            for data in train_val_test_dict:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def plot_prediction_results(
        train_pred: [], train_true: [], val_pred: [], val_true: [],
        test_pred: [], test_true: [], model_name, model_type
) -> None:
    evaluation_train = ResultMetric(train_pred, train_true)
    evaluation_train.print_metrics(f'Training_{model_name}')
    train_dict = evaluation_train.get_metrics_dict('Training')

    evaluation_val = ResultMetric(val_pred, val_true)
    evaluation_val.print_metrics(f'Validation_{model_name}')
    val_dict = evaluation_val.get_metrics_dict('Validation')

    evaluation_test = ResultMetric(test_pred, test_true)
    evaluation_test.print_metrics(f'Test_{model_name}')
    test_dict = evaluation_test.get_metrics_dict('Test')

    dir_path = os.path.join(FIGURES_DIR, model_type)
    check_and_create_dir_path(dir_path)

    write_to_csv([train_dict, val_dict, test_dict], dir_path, model_name)

    return
