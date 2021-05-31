import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Masking, Bidirectional, LSTM, Dense
from tensorflow.python.keras.losses import BinaryCrossentropy

from src.features.helpers.metrics import get_model_metrics


def make_lstm_model_v2(input_dim: None, layers: int, cells: int):
    model = Sequential()

    model.add(Masking(mask_value=0.0, input_shape=input_dim))
    if layers > 1:
        for i in range(layers - 1):
            model.add(
                Bidirectional(
                    LSTM(cells, return_sequences=True)
                )
            )
    model.add(
        Bidirectional(
            LSTM(cells, return_sequences=False)
        )
    )
    model.add(
        Dense(1, activation='sigmoid')
    )

    adam = tf.optimizers.Adam(clipvalue=0.5, learning_rate=0.0001)
    model.compile(
        optimizer=adam,
        loss=BinaryCrossentropy(),
        metrics=get_model_metrics()
    )
    return model
