import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import CustomLSTMConfig as Config
from config import DATASET_NAME


def calculate_moving_average(data, window_size):
    window_size = int(window_size)
    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return np.concatenate((np.zeros(window_size - 1), moving_avg), axis=0)


def create_sequences(X, y, time_steps, prediction_horizon):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps - prediction_horizon + 1):
        X_seq.append(X[i:i + time_steps])
        y_temp = y[i + time_steps:i + time_steps + prediction_horizon]
        y_seq.append(y_temp)
    return np.array(X_seq), np.array(y_seq)


def create_train_test_data(_X_train_val_, _y_train_val_, _X_test_, _y_test_):
    _X_train_, _X_val_, _y_train_, _y_val_ = train_test_split(_X_train_val_,
                                                              _y_train_val_,
                                                              test_size=Config.SPLIT_RATIO,
                                                              shuffle=False)

    scaler = StandardScaler()
    _X_train_ = scaler.fit_transform(_X_train_)
    _X_val_ = scaler.transform(_X_val_)
    _X_test_ = scaler.transform(_X_test_)

    _X_train_seq_, _y_train_seq_ = create_sequences(_X_train_, _y_train_, Config.TRAIN_WINDOW_SIZE, Config.N_PREDICTION)
    _X_val_seq_, _y_val_seq_ = create_sequences(_X_val_, _y_val_, Config.TRAIN_WINDOW_SIZE, Config.N_PREDICTION)
    _X_test_seq_, _y_test_seq_ = create_sequences(_X_test_, _y_test_, Config.TRAIN_WINDOW_SIZE, Config.N_PREDICTION)

    return _X_train_seq_, _y_train_seq_, _X_val_seq_, _y_val_seq_, _X_test_seq_, _y_test_seq_


def create_input_features(patient_id):
    df_train = pd.read_csv('./dataset/' + DATASET_NAME + '/' + patient_id + '_train.csv')
    df_test = pd.read_csv('./dataset/' + DATASET_NAME + '/' + patient_id + '_test.csv')

    df_train.replace(-1, np.nan, inplace=True)
    df_test.replace(-1, np.nan, inplace=True)

    df_train = df_train.dropna(subset=['glucose'])
    df_test = df_test.dropna(subset=['glucose'])

    df_train.fillna(0, inplace=True)
    df_test.fillna(0, inplace=True)

    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    relevant_features = ['basal', 'carbs', 'bolus', 'glucose']

    return (df_train[relevant_features].values,
            df_train['glucose'].values,
            df_test[relevant_features].values,
            df_test['glucose'].values)
