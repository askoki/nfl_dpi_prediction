import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Masking, GRU
from tensorflow.python.keras.losses import BinaryCrossentropy

from src.features.helpers.metrics import get_model_metrics


def make_gru_model(input_dim: None, layers: int, cells: int):
    model = Sequential()

    model.add(Masking(mask_value=0.0, input_shape=input_dim))
    if layers > 1:
        for i in range(layers - 1):
            model.add(GRU(cells, return_sequences=True))
    model.add(GRU(cells, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    adam = tf.optimizers.Adam(clipvalue=0.5, learning_rate=0.0001)
    model.compile(
        optimizer=adam,
        loss=BinaryCrossentropy(),
        metrics=get_model_metrics()
    )
    return model

