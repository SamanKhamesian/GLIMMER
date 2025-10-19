from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.activations import gelu
from config import TransformerConfig as Config

def transformer_block(x, num_heads, ff_dim, dropout_rate, key_dim):
    input_dim = x.shape[-1]
    x_norm = LayerNormalization()(x)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_norm, x_norm)
    x = Add()([x, Dropout(dropout_rate)(attn_output)])
    x_norm = LayerNormalization()(x)
    ff = Dense(ff_dim, activation=gelu)(x_norm)
    ff = Dropout(dropout_rate)(ff)
    ff = Dense(ff_dim // 2, activation=gelu)(ff)
    ff = Dense(input_dim)(ff)
    return Add()([x, ff])

def build_model(input_shape, output_dim, config):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, 4, activation=Config.ACTIVATION)(inputs)
    x = Dropout(0.1)(x)
    x = Conv1D(16, 4, activation=Config.ACTIVATION)(x)
    x = Dropout(0.1)(x)
    x = Conv1D(8, 4, activation=Config.ACTIVATION)(x)
    x = Dropout(0.4)(x)

    for _ in range(config.NUM_BLOCKS):
        x = transformer_block(x, config.NUM_HEADS, config.FF_DIM, config.DROPOUT, config.KEY_DIM)

    x = Flatten()(x)
    x = Dense(output_dim, activation=config.ACTIVATION)(x)
    outputs = Dense(output_dim)(x)

    return Model(inputs, outputs)
