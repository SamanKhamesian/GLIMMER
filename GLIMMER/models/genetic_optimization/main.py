import os
import random
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Dropout

from config import CustomLSTMConfig as Config
from config import GeneticAlgorithmConfig as GenConfig
from config import PATIENT_ID_LIST, Threshold
from models.custom_lstm.preprocess import create_train_test_data, create_input_features
from postprocess import (calculate_error_metrics, plot_fitness_evolution, plot_contour)

warnings.filterwarnings('ignore')

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

folder_path = str(GenConfig.FOLDER_PATH + 'test/')


def create_loss_function(w_hypo, w_hyper):
    w_normal = 1

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

    for patient_id in PATIENT_ID_LIST:
        patient_id = str(patient_id)
        patient_path = str(folder_path + patient_id + '/')

        if not os.path.exists(patient_path):
            os.makedirs(patient_path)

        y_pred, history = None, None

        X_train, y_train, X_test, y_test = create_input_features(patient_id=patient_id)
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = create_train_test_data(X_train,
                                                                                                        y_train,
                                                                                                        X_test,
                                                                                                        y_test)
        pop_size = GenConfig.POPULATION_SIZE
        population = np.random.uniform(1, 10, (pop_size, 2))
        best_scores = []
        all_fitness_scores = []

        # Train and Val
        for g in range(GenConfig.N_GENERATION):
            fitness_scores = []

            for p, weights in enumerate(population):
                print("--------------------- Patient ID: {} ---------------------".format(patient_id))
                print('----------------- Generation {0} is started -----------------'.format(g))
                print('----------------- Population {0} is running -----------------'.format(p))

                model = Sequential([
                    Conv1D(filters=32, kernel_size=4, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                    Dropout(0.1), Conv1D(filters=16, kernel_size=4), Dropout(0.1), Conv1D(filters=8, kernel_size=4),
                    Dropout(0.4), LSTM(units=8, return_sequences=True), Dropout(0.4), Flatten(),
                    Dense(Config.N_PREDICTION, activation=Config.ACTIVATION), Dense(units=Config.N_PREDICTION)])

                loss_func = create_loss_function(*weights)
                model.compile(optimizer=Config.OPTIMIZER, loss=loss_func)
                model.fit(X_train_seq,
                          y_train_seq,
                          epochs=Config.EPOCHS,
                          batch_size=Config.BATCH_SIZE,
                          validation_data=(X_val_seq, y_val_seq))

                y_pred_temp = model.predict(X_val_seq)
                (rmse, mse, mae, mape), (rmse_normal, mse_normal, mae_normal, mape_normal), (
                    rmse_dysglycemic, mse_dysglycemic, mae_dysglycemic, mape_dysglycemic), (
                    rmse_hyperglycemia, mse_hyperglycemia, mae_hyperglycemia, mape_hyperglycemia), (
                    rmse_hypoglycemia, mse_hypoglycemia, mae_hypoglycemia, mape_hypoglycemia) = calculate_error_metrics(
                    save_to=patient_path,
                    patient_id=patient_id,
                    y_true=y_val_seq,
                    y_pred=y_pred_temp)

                fitness_scores.append(rmse)

            best_scores.append(min(fitness_scores))
            best_individuals = population[np.argsort(fitness_scores)[:pop_size // 2]]
            all_fitness_scores.append(fitness_scores)

            # Crossover and mutation
            offspring = []
            while len(offspring) < pop_size // 2:
                parent1, parent2 = np.random.choice(len(best_individuals), 2, replace=False)
                child = np.mean([best_individuals[parent1], best_individuals[parent2]], axis=0)
                # Mutation
                mutation = np.random.normal(0, 0.5, 2)
                child = np.clip(child + mutation, 1, 10)
                offspring.append(child)

            population = np.vstack((best_individuals, offspring))

        final_fitness_scores = all_fitness_scores[-1]
        best_index = np.argmin(final_fitness_scores)
        best_weights = population[best_index]

        plot_contour(save_to=patient_path,
                     patient_id=patient_id,
                     population=population,
                     fitness_scores=final_fitness_scores,
                     best_individuals=best_weights,
                     show=False)

        plot_fitness_evolution(save_to=patient_path,
                               patient_id=patient_id,
                               all_fitness_scores=all_fitness_scores,
                               show=False)

        weights = best_weights
        print('Selected weights (w_hypo, w_hyper): ', weights)
