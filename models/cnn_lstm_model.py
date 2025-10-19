from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten

def build_model(input_shape, output_dim, config):
    model = Sequential([
        Conv1D(32, 4, input_shape=input_shape),
        Dropout(0.1),
        Conv1D(16, 4),
        Dropout(0.1),
        Conv1D(8, 4),
        Dropout(0.4),
        LSTM(8, return_sequences=True),
        Dropout(0.1),
        Flatten(),
        Dense(output_dim, activation=config.ACTIVATION),
        Dense(output_dim)
    ])
    return model
