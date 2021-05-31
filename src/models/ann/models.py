import tensorflow as tf
from attention import Attention
from keras.layers import Dense, LSTM, Bidirectional, Input, Masking
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.models import Model

from src.features.helpers.metrics import get_model_metrics


def make_ann_model_v2(input_dim: None, cells: int):
    model_input = Input(shape=input_dim)
    x = Masking(mask_value=0.0, input_shape=input_dim)(model_input)

    x = Bidirectional(LSTM(cells, input_shape=input_dim, return_sequences=True))(x)
    x = Attention(cells)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(model_input, x)

    adam = tf.optimizers.Adam(clipvalue=0.5, learning_rate=0.0001)
    model.compile(
        optimizer=adam,
        loss=BinaryCrossentropy(),
        metrics=get_model_metrics()
    )

    return model
