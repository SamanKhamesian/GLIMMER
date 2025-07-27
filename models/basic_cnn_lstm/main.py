import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Dropout

from config import CustomLSTMConfig as Config
from config import PATIENT_ID_LIST
from models.basic_cnn_lstm.preprocess import create_train_test_data, create_input_features
from postprocess import (calculate_scores, calculate_error_metrics, plot_prediction_results, plot_train_val_history,
                         calculate_correlation_coefficient, calculate_total_error_metrics_and_scores, clarke_error_grid)

folder_path = str(Config.FOLDER_PATH + 'journal_results/')

if __name__ == '__main__':
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    list_clarke_error_zones = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}
    list_y_pred, list_y_true = [], []
    list_rmse_total, list_mse_total, list_mae_total, list_mape_total = [], [], [], []

    list_f1_score_normal, list_f1_score_glycemic, list_f1_score_hypo, list_f1_score_hyper = [], [], [], []
    list_precision_normal, list_precision_glycemic, list_precision_hypo, list_precision_hyper = [], [], [], []
    list_recall_normal, list_recall_glycemic, list_recall_hypo, list_recall_hyper = [], [], [], []

    list_rmse_normal, list_mse_normal, list_mae_normal, list_mape_normal = [], [], [], []
    list_rmse_hyperglycemia, list_mse_hyperglycemia, list_mae_hyperglycemia, list_mape_hyperglycemia = [], [], [], []
    list_rmse_hypoglycemia, list_mse_hypoglycemia, list_mae_hypoglycemia, list_mape_hypoglycemia = [], [], [], []
    list_rmse_glycemic, list_mse_glycemic, list_mae_glycemic, list_mape_glycemic = [], [], [], []

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

            model.compile(optimizer=Config.OPTIMIZER, loss='mean_absolute_error')

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

        zones = clarke_error_grid(save_to=patient_path,
                                  patient_id=patient_id,
                                  y_true=y_test_seq[:, 0],
                                  y_pred=y_pred[:, 0],
                                  show=True)

        list_clarke_error_zones['A'].append(zones['A'])
        list_clarke_error_zones['B'].append(zones['B'])
        list_clarke_error_zones['C'].append(zones['C'])
        list_clarke_error_zones['D'].append(zones['D'])
        list_clarke_error_zones['E'].append(zones['E'])

        (rmse_total, mse_total, mae_total, mape_total), (rmse_normal, mse_normal, mae_normal, mape_normal), (
        rmse_glycemic, mse_glycemic, mae_glycemic, mape_glycemic), (
        rmse_hyperglycemia, mse_hyperglycemia, mae_hyperglycemia, mape_hyperglycemia), (
        rmse_hypoglycemia, mse_hypoglycemia, mae_hypoglycemia,
        mape_hypoglycemia) = calculate_error_metrics(save_to=patient_path,
                                                     patient_id=patient_id,
                                                     y_true=y_test_seq,
                                                     y_pred=y_pred,
                                                     zones=zones)

        (f1_score_hyper, f1_score_hypo, f1_score_normal, f1_score_glycemic, precision_hyper, precision_hypo,
         precision_normal, precision_glycemic, recall_hyper, recall_hypo, recall_normal,
         recall_glycemic) = calculate_scores(save_to=patient_path, y_true=y_test_seq, y_pred=y_pred)

        list_precision_hyper.append(precision_hyper)
        list_precision_hypo.append(precision_hypo)
        list_precision_normal.append(precision_normal)
        list_precision_glycemic.append(precision_glycemic)

        list_recall_hyper.append(recall_hyper)
        list_recall_hypo.append(recall_hypo)
        list_recall_normal.append(recall_normal)
        list_recall_glycemic.append(recall_glycemic)

        list_f1_score_hyper.append(f1_score_hyper)
        list_f1_score_hypo.append(f1_score_hypo)
        list_f1_score_normal.append(f1_score_normal)
        list_f1_score_glycemic.append(f1_score_glycemic)

        list_rmse_normal.append(rmse_normal)
        list_mse_normal.append(mse_normal)
        list_mae_normal.append(mae_normal)
        list_mape_normal.append(mape_normal)

        list_rmse_glycemic.append(rmse_glycemic)
        list_mse_glycemic.append(mse_glycemic)
        list_mae_glycemic.append(mae_glycemic)
        list_mape_glycemic.append(mape_glycemic)

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
                                             list_f1_score_glycemic=list_f1_score_glycemic,
                                             list_precision_hyper=list_precision_hyper,
                                             list_precision_hypo=list_precision_hypo,
                                             list_precision_normal=list_precision_normal,
                                             list_precision_glycemic=list_precision_glycemic,
                                             list_recall_hyper=list_recall_hyper,
                                             list_recall_hypo=list_recall_hypo,
                                             list_recall_normal=list_recall_normal,
                                             list_recall_glycemic=list_recall_glycemic,
                                             list_rmse_normal=list_rmse_normal,
                                             list_mse_normal=list_mse_normal,
                                             list_mae_normal=list_mae_normal,
                                             list_mape_normal=list_mape_normal,
                                             list_rmse_glycemic=list_rmse_glycemic,
                                             list_mse_glycemic=list_mse_glycemic,
                                             list_mae_glycemic=list_mae_glycemic,
                                             list_mape_glycemic=list_mape_glycemic,
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
