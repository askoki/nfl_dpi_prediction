from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Masking, LSTM, Dense, Dropout, Permute, Conv1D, \
    BatchNormalization, Activation, GlobalAveragePooling1D, Reshape, concatenate, multiply
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from src.features.helpers.metrics import get_model_metrics


def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input.shape[-1]  # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


def generate_model_mlstm(input_dim: None, cells: int, learning_rate=1e-3):
    ip = Input(shape=input_dim)

    x = Masking(mask_value=0.0, input_shape=input_dim)(ip)
    x = LSTM(cells)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(1, activation='sigmoid')(x)
    optm = Adam(lr=learning_rate)

    model = Model(ip, out)
    model.compile(
        optimizer=optm,
        loss=BinaryCrossentropy(),
        metrics=get_model_metrics()
    )

    return model
