import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Dropout

from config import CustomLSTMConfig as Config
from config import PATIENT_ID_LIST, Threshold
from models.custom_lstm.preprocess import create_train_test_data, create_input_features
from postprocess import (calculate_scores, calculate_error_metrics, plot_prediction_results, plot_train_val_history,
                         calculate_correlation_coefficient, calculate_total_error_metrics_and_scores, clarke_error_grid)

folder_path = str(Config.FOLDER_PATH + 'test/')


def prepare_clarke_error_output():
    zones = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}
    for i in range(Config.N_PREDICTION - 1, -1, -1):
        zones_temp = clarke_error_grid(save_to=patient_path,
                                       patient_id=patient_id,
                                       y_true=y_test_seq[:, i],
                                       y_pred=y_pred[:, i],
                                       show=False)
        zones_total_temp = zones_temp['A'] + zones_temp['B'] + zones_temp['C'] + zones_temp['D'] + zones_temp['E']

        zones['A'].append(zones_temp['A'] / zones_total_temp * 100)
        zones['B'].append(zones_temp['B'] / zones_total_temp * 100)
        zones['C'].append(zones_temp['C'] / zones_total_temp * 100)
        zones['D'].append(zones_temp['D'] / zones_total_temp * 100)
        zones['E'].append(zones_temp['E'] / zones_total_temp * 100)

    return zones


def create_loss_function(w_normal, w_hypo, w_hyper):
    def custom_loss(y_true, y_pred):
        mae = K.abs(y_pred - y_true)

        normal_abs_error = mae * tf.cast(tf.logical_and(y_true >= Threshold.HYPOGLYCEMIA,
                                                        y_true <= Threshold.HYPERGLYCEMIA), K.floatx()) * w_normal
        penalty_lower = K.cast(y_true < Threshold.HYPOGLYCEMIA, K.floatx()) * mae * w_hypo
        penalty_upper = K.cast(y_true > Threshold.HYPERGLYCEMIA, K.floatx()) * mae * w_hyper

        return K.mean(normal_abs_error + penalty_lower + penalty_upper)

    return custom_loss


if __name__ == '__main__':
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    list_clarke_error_zones = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}
    list_y_pred, list_y_true = [], []
    list_rmse_total, list_mse_total, list_mae_total, list_mape_total = [], [], [], []

    list_f1_score_normal, list_f1_score_dysglycemic, list_f1_score_hypo, list_f1_score_hyper = [], [], [], []
    list_precision_normal, list_precision_dysglycemic, list_precision_hypo, list_precision_hyper = [], [], [], []
    list_recall_normal, list_recall_dysglycemic, list_recall_hypo, list_recall_hyper = [], [], [], []

    list_rmse_normal, list_mse_normal, list_mae_normal, list_mape_normal = [], [], [], []
    list_rmse_hyperglycemia, list_mse_hyperglycemia, list_mae_hyperglycemia, list_mape_hyperglycemia = [], [], [], []
    list_rmse_hypoglycemia, list_mse_hypoglycemia, list_mae_hypoglycemia, list_mape_hypoglycemia = [], [], [], []
    list_rmse_dysglycemic, list_mse_dysglycemic, list_mae_dysglycemic, list_mape_dysglycemic = [], [], [], []

    list_corr = []

    for patient_id in PATIENT_ID_LIST:
        patient_id = str(patient_id)
        patient_path = str(folder_path + patient_id + '/')

        if not os.path.exists(patient_path):
            os.makedirs(patient_path)

        history, y_pred = None, None

        X_train, y_train, X_test, y_test = create_input_features(patient_id=patient_id)
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = create_train_test_data(X_train,
                                                                                                        y_train,
                                                                                                        X_test,
                                                                                                        y_test)

        for i in range(Config.REPEAT):

            random.seed(int(i + 100))
            np.random.seed(int(i + 100))
            tf.random.set_seed(int(i + 100))

            model = Sequential([
                Conv1D(filters=32, kernel_size=4, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                Dropout(0.1), Conv1D(filters=16, kernel_size=4), Dropout(0.1), Conv1D(filters=8, kernel_size=4),
                Dropout(0.4), LSTM(units=8, return_sequences=True), Dropout(0.1), Flatten(),
                Dense(Config.N_PREDICTION, activation=Config.ACTIVATION), Dense(units=Config.N_PREDICTION)])

            weights = Config.WEIGHTS
            loss_func = create_loss_function(*weights)
            model.compile(optimizer=Config.OPTIMIZER, loss=loss_func)

            model.summary()

            history = model.fit(X_train_seq,
                                y_train_seq,
                                epochs=Config.EPOCHS,
                                batch_size=Config.BATCH_SIZE,
                                validation_data=(X_val_seq, y_val_seq))

            y_pred_temp = model.predict(X_test_seq)

            if y_pred is None:
                y_pred = y_pred_temp
            else:
                y_pred += y_pred_temp

        y_pred = y_pred / Config.REPEAT

        list_y_pred.append(y_pred[:, 0])
        list_y_true.append(y_test_seq[:, 0])

        zones = prepare_clarke_error_output()

        list_clarke_error_zones['A'].append(np.mean(zones['A']))
        list_clarke_error_zones['B'].append(np.mean(zones['B']))
        list_clarke_error_zones['C'].append(np.mean(zones['C']))
        list_clarke_error_zones['D'].append(np.mean(zones['D']))
        list_clarke_error_zones['E'].append(np.mean(zones['E']))

        (rmse_total, mse_total, mae_total, mape_total), (rmse_normal, mse_normal, mae_normal, mape_normal), (
            rmse_dysglycemic, mse_dysglycemic, mae_dysglycemic, mape_dysglycemic), (
            rmse_hyperglycemia, mse_hyperglycemia, mae_hyperglycemia, mape_hyperglycemia), (
            rmse_hypoglycemia, mse_hypoglycemia, mae_hypoglycemia,
            mape_hypoglycemia) = calculate_error_metrics(save_to=patient_path,
                                                         patient_id=patient_id,
                                                         y_true=y_test_seq,
                                                         y_pred=y_pred,
                                                         zones=zones)

        (f1_score_hyper, f1_score_hypo, f1_score_normal, f1_score_dysglycemic, precision_hyper, precision_hypo,
         precision_normal, precision_dysglycemic, recall_hyper, recall_hypo, recall_normal,
         recall_dysglycemic) = calculate_scores(save_to=patient_path, y_true=y_test_seq, y_pred=y_pred)

        list_precision_hyper.append(precision_hyper)
        list_precision_hypo.append(precision_hypo)
        list_precision_normal.append(precision_normal)
        list_precision_dysglycemic.append(precision_dysglycemic)

        list_recall_hyper.append(recall_hyper)
        list_recall_hypo.append(recall_hypo)
        list_recall_normal.append(recall_normal)
        list_recall_dysglycemic.append(recall_dysglycemic)

        list_f1_score_hyper.append(f1_score_hyper)
        list_f1_score_hypo.append(f1_score_hypo)
        list_f1_score_normal.append(f1_score_normal)
        list_f1_score_dysglycemic.append(f1_score_dysglycemic)

        list_rmse_normal.append(rmse_normal)
        list_mse_normal.append(mse_normal)
        list_mae_normal.append(mae_normal)
        list_mape_normal.append(mape_normal)

        list_rmse_dysglycemic.append(rmse_dysglycemic)
        list_mse_dysglycemic.append(mse_dysglycemic)
        list_mae_dysglycemic.append(mae_dysglycemic)
        list_mape_dysglycemic.append(mape_dysglycemic)

        list_rmse_hyperglycemia.append(rmse_hyperglycemia)
        list_mse_hyperglycemia.append(mse_hyperglycemia)
        list_mae_hyperglycemia.append(mae_hyperglycemia)
        list_mape_hyperglycemia.append(mape_hyperglycemia)

        list_rmse_hypoglycemia.append(rmse_hypoglycemia)
        list_mse_hypoglycemia.append(mse_hypoglycemia)
        list_mae_hypoglycemia.append(mae_hypoglycemia)
        list_mape_hypoglycemia.append(mape_hypoglycemia)

        correlation_matrix = calculate_correlation_coefficient(save_to=patient_path,
                                                               patient_id=patient_id,
                                                               y_true=y_test_seq,
                                                               y_pred=y_pred,
                                                               length=Config.N_PREDICTION,
                                                               show=False)

        plot_train_val_history(save_to=patient_path, patient_id=patient_id, history=history, show=False)

        plot_prediction_results(save_to=patient_path,
                                patient_id=patient_id,
                                y_true=y_test_seq,
                                y_pred=y_pred,
                                rmse=rmse_total,
                                mse=mse_total,
                                mae=mae_total,
                                mape=mape_total,
                                show=False)

        list_rmse_total.append(rmse_total)
        list_mse_total.append(mse_total)
        list_mae_total.append(mae_total)
        list_mape_total.append(mape_total)
        list_corr.append(correlation_matrix)

    list_y_pred = [item for sublist in list_y_pred for item in sublist]
    list_y_true = [item for sublist in list_y_true for item in sublist]

    clarke_error_grid(save_to=folder_path, patient_id="None", y_true=list_y_true, y_pred=list_y_pred, show=True)

    calculate_total_error_metrics_and_scores(save_to=folder_path,
                                             list_rmse_total=list_rmse_total,
                                             list_mse_total=list_mse_total,
                                             list_mae_total=list_mae_total,
                                             list_mape_total=list_mape_total,
                                             list_f1_score_hypo=list_f1_score_hypo,
                                             list_f1_score_hyper=list_f1_score_hyper,
                                             list_f1_score_normal=list_f1_score_normal,
                                             list_f1_score_dysglycemic=list_f1_score_dysglycemic,
                                             list_precision_hyper=list_precision_hyper,
                                             list_precision_hypo=list_precision_hypo,
                                             list_precision_normal=list_precision_normal,
                                             list_precision_dysglycemic=list_precision_dysglycemic,
                                             list_recall_hyper=list_recall_hyper,
                                             list_recall_hypo=list_recall_hypo,
                                             list_recall_normal=list_recall_normal,
                                             list_recall_dysglycemic=list_recall_dysglycemic,
                                             list_rmse_normal=list_rmse_normal,
                                             list_mse_normal=list_mse_normal,
                                             list_mae_normal=list_mae_normal,
                                             list_mape_normal=list_mape_normal,
                                             list_rmse_dysglycemic=list_rmse_dysglycemic,
                                             list_mse_dysglycemic=list_mse_dysglycemic,
                                             list_mae_dysglycemic=list_mae_dysglycemic,
                                             list_mape_dysglycemic=list_mape_dysglycemic,
                                             list_rmse_hyperglycemia=list_rmse_hyperglycemia,
                                             list_mse_hyperglycemia=list_mse_hyperglycemia,
                                             list_mae_hyperglycemia=list_mae_hyperglycemia,
                                             list_mape_hyperglycemia=list_mape_hyperglycemia,
                                             list_rmse_hypoglycemia=list_rmse_hypoglycemia,
                                             list_mse_hypoglycemia=list_mse_hypoglycemia,
                                             list_mae_hypoglycemia=list_mae_hypoglycemia,
                                             list_mape_hypoglycemia=list_mape_hypoglycemia,
                                             list_clarke_error_zones=list_clarke_error_zones,
                                             list_corr=list_corr)
