import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from config import DATASET_NAME, Threshold


# Clarke Error Grid function with heatmap and zone counts
def clarke_error_grid(save_to, patient_id, y_true, y_pred, show=False):
    assert (len(y_true) == len(y_pred)), (
        "Unequal number of values (reference : {}) (prediction : {}).".format(len(y_true), len(y_pred)))

    # Set up plot
    plt.figure(figsize=(8, 8))
    plt.title(f"Clarke Error Grid" + ' / ' + DATASET_NAME + ' dataset' + ' / ' + 'Patient ID: ' + patient_id + '\n',
              fontsize=14)

    plt.xlabel("Reference Concentration (mg/dl)", fontsize=24, labelpad=14)
    plt.ylabel("Prediction Concentration (mg/dl)", fontsize=24, labelpad=14)
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.gca().set_facecolor('white')

    # Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect(400 / 400)

    # Plot zone lines
    plt.plot([0, 400], [0, 400], ':', c='black')
    plt.plot([0, 175 / 3], [70, 70], '-', c='black')
    plt.plot([175 / 3, 400 / 1.2], [70, 400], '-', c='black')
    plt.plot([70, 70], [84, 400], '-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290], [180, 400], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')
    plt.plot([70, 400], [56, 320], '-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180], '-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    # Add zone titles
    plt.text(30, 15, "A", fontsize=20)
    plt.text(370, 260, "B", fontsize=20)
    plt.text(280, 370, "B", fontsize=20)
    plt.text(160, 370, "C", fontsize=20)
    plt.text(160, 15, "C", fontsize=20)
    plt.text(30, 140, "D", fontsize=20)
    plt.text(370, 120, "D", fontsize=20)
    plt.text(30, 370, "E", fontsize=20)
    plt.text(370, 15, "E", fontsize=20)

    # Assign colors for each zone and count points per zone
    zone_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    for i in range(len(y_true)):
        # Zone A
        if (y_true[i] <= 70 and y_pred[i] <= 70) or (1.2 * y_true[i] >= y_pred[i] >= 0.8 * y_true[i]):
            zone_counts['A'] += 1
        # Zone E
        elif (y_true[i] >= 180 and y_pred[i] <= 70) or (y_true[i] <= 70 and y_pred[i] >= 180):
            zone_counts['E'] += 1
        # Zone C
        elif ((70 <= y_true[i] <= 290) and y_pred[i] >= y_true[i] + 110) or (
                (130 <= y_true[i] <= 180) and (y_pred[i] <= (7 / 5) * y_true[i] - 182)):
            zone_counts['C'] += 1
        # Zone D
        elif (y_true[i] >= 240 and (70 <= y_pred[i] <= 180)) or (y_true[i] <= 175 / 3 and 180 >= y_pred[i] >= 70) or (
                (175 / 3 <= y_true[i] <= 70) and y_pred[i] >= (6 / 5) * y_true[i]):
            zone_counts['D'] += 1
        # Zone B
        else:
            zone_counts['B'] += 1

    # Plot colorful dots for each zone
    plt.scatter(y_true, y_pred, c='c', s=50, marker='o', edgecolor='black')

    plt.tight_layout()
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + 'clarke_error_grid' + '.png'), dpi=400)
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + 'clarke_error_grid' + '.pdf'), dpi=400)

    if show is True:
        plt.show()

    plt.close()
    return zone_counts


def calculate_error_metrics(save_to,
                            patient_id,
                            y_true,
                            y_pred,
                            zones=None,
                            hyperglycemia_th=180,
                            hypoglycemia_th=70,
                            file_name='error_metrics.txt'):
    # Overall error metrics
    if len(y_true) > 0 and len(y_pred) > 0:
        rmse_total = np.sqrt(mean_squared_error(y_true, y_pred))
        mse_total = mean_squared_error(y_true, y_pred)
        mae_total = mean_absolute_error(y_true, y_pred)
        mape_total = mean_absolute_percentage_error(y_true, y_pred)
    else:
        rmse_total = mse_total = mae_total = mape_total = np.nan

    # Normal range mask (70 <= y_true <= 180)
    normal_mask = (y_true >= hypoglycemia_th) & (y_true <= hyperglycemia_th)
    y_true_normal = y_true[normal_mask]
    y_pred_normal = y_pred[normal_mask]

    if len(y_true_normal) > 0 and len(y_pred_normal) > 0:
        rmse_normal = np.sqrt(mean_squared_error(y_true_normal, y_pred_normal))
        mse_normal = mean_squared_error(y_true_normal, y_pred_normal)
        mae_normal = mean_absolute_error(y_true_normal, y_pred_normal)
        mape_normal = mean_absolute_percentage_error(y_true_normal, y_pred_normal)
    else:
        rmse_normal = mse_normal = mae_normal = mape_normal = np.nan

    # Dysdysglycemic regions mask (y_true < 70 or y_true > 180)
    dysglycemic_mask = (y_true < hypoglycemia_th) | (y_true > hyperglycemia_th)
    y_true_dysglycemic = y_true[dysglycemic_mask]
    y_pred_dysglycemic = y_pred[dysglycemic_mask]

    if len(y_true_dysglycemic) > 0 and len(y_pred_dysglycemic) > 0:
        rmse_dysglycemic = np.sqrt(mean_squared_error(y_true_dysglycemic, y_pred_dysglycemic))
        mse_dysglycemic = mean_squared_error(y_true_dysglycemic, y_pred_dysglycemic)
        mae_dysglycemic = mean_absolute_error(y_true_dysglycemic, y_pred_dysglycemic)
        mape_dysglycemic = mean_absolute_percentage_error(y_true_dysglycemic, y_pred_dysglycemic)
    else:
        rmse_dysglycemic = mse_dysglycemic = mae_dysglycemic = mape_dysglycemic = np.nan

    # Hyperglycemia region mask (y_true > 180)
    hyperglycemia_mask = y_true > hyperglycemia_th
    y_true_hyperglycemia = y_true[hyperglycemia_mask]
    y_pred_hyperglycemia = y_pred[hyperglycemia_mask]

    if len(y_true_hyperglycemia) > 0 and len(y_pred_hyperglycemia) > 0:
        rmse_hyperglycemia = np.sqrt(mean_squared_error(y_true_hyperglycemia, y_pred_hyperglycemia))
        mse_hyperglycemia = mean_squared_error(y_true_hyperglycemia, y_pred_hyperglycemia)
        mae_hyperglycemia = mean_absolute_error(y_true_hyperglycemia, y_pred_hyperglycemia)
        mape_hyperglycemia = mean_absolute_percentage_error(y_true_hyperglycemia, y_pred_hyperglycemia)
    else:
        rmse_hyperglycemia = mse_hyperglycemia = mae_hyperglycemia = mape_hyperglycemia = np.nan

    # Hypoglycemia region mask (y_true < 70)
    hypoglycemia_mask = y_true < hypoglycemia_th
    y_true_hypoglycemia = y_true[hypoglycemia_mask]
    y_pred_hypoglycemia = y_pred[hypoglycemia_mask]

    if len(y_true_hypoglycemia) > 0 and len(y_pred_hypoglycemia) > 0:
        rmse_hypoglycemia = np.sqrt(mean_squared_error(y_true_hypoglycemia, y_pred_hypoglycemia))
        mse_hypoglycemia = mean_squared_error(y_true_hypoglycemia, y_pred_hypoglycemia)
        mae_hypoglycemia = mean_absolute_error(y_true_hypoglycemia, y_pred_hypoglycemia)
        mape_hypoglycemia = mean_absolute_percentage_error(y_true_hypoglycemia, y_pred_hypoglycemia)
    else:
        rmse_hypoglycemia = mse_hypoglycemia = mae_hypoglycemia = mape_hypoglycemia = np.nan

    if zones is not None:
        zone_A_mean, zone_A_std = np.mean(zones['A']), np.std(zones['A'])
        zone_B_mean, zone_B_std = np.mean(zones['B']), np.std(zones['B'])
        zone_C_mean, zone_C_std = np.mean(zones['C']), np.std(zones['C'])
        zone_D_mean, zone_D_std = np.mean(zones['D']), np.std(zones['D'])
        zone_E_mean, zone_E_std = np.mean(zones['E']), np.std(zones['E'])

    # Print results
    print('\n')
    print("--------------------- Patient ID: {} ---------------------".format(patient_id))
    print('Overall Error Metrics:')
    print('Root Mean Squared Error (RMSE): {:.2f}'.format(rmse_total))
    print('Mean Squared Error (MSE): {:.2f}'.format(mse_total))
    print('Mean Absolute Error (MAE): {:.2f}'.format(mae_total))
    print('Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(mape_total))

    print('\nNormal Region Error Metrics:')
    print('Root Mean Squared Error (RMSE): {:.2f}'.format(rmse_normal))
    print('Mean Squared Error (MSE): {:.2f}'.format(mse_normal))
    print('Mean Absolute Error (MAE): {:.2f}'.format(mae_normal))
    print('Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(mape_normal))

    print('\ndysglycemic Region Error Metrics:')
    print('Root Mean Squared Error (RMSE): {:.2f}'.format(rmse_dysglycemic))
    print('Mean Squared Error (MSE): {:.2f}'.format(mse_dysglycemic))
    print('Mean Absolute Error (MAE): {:.2f}'.format(mae_dysglycemic))
    print('Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(mape_dysglycemic))

    print('\nHyperglycemia Region Error Metrics:')
    print('Root Mean Squared Error (RMSE): {:.2f}'.format(rmse_hyperglycemia))
    print('Mean Squared Error (MSE): {:.2f}'.format(mse_hyperglycemia))
    print('Mean Absolute Error (MAE): {:.2f}'.format(mae_hyperglycemia))
    print('Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(mape_hyperglycemia))

    print('\nHypoglycemia Region Error Metrics:')
    print('Root Mean Squared Error (RMSE): {:.2f}'.format(rmse_hypoglycemia))
    print('Mean Squared Error (MSE): {:.2f}'.format(mse_hypoglycemia))
    print('Mean Absolute Error (MAE): {:.2f}'.format(mae_hypoglycemia))
    print('Mean Absolute Percentage Error (MAPE): {:.2f}%\n'.format(mape_hypoglycemia))

    if zones is not None:
        # Print zone distribution
        print('\nClark Error Zone Distribution:')
        print('Zone A: {:.2f}% (STD: {:.2f}%)'.format(zone_A_mean, zone_A_std))
        print('Zone B: {:.2f}% (STD: {:.2f}%)'.format(zone_B_mean, zone_B_std))
        print('Zone C: {:.2f}% (STD: {:.2f}%)'.format(zone_C_mean, zone_C_std))
        print('Zone D: {:.2f}% (STD: {:.2f}%)'.format(zone_D_mean, zone_D_std))
        print('Zone E: {:.2f}% (STD: {:.2f}%)'.format(zone_E_mean, zone_E_std))

    # Save to file
    with open(save_to + file_name, 'w') as f:
        f.write('Overall Error Metrics:\n')
        f.write('Root Mean Squared Error (RMSE): {:.2f}\n'.format(rmse_total))
        f.write('Mean Squared Error (MSE): {:.2f}\n'.format(mse_total))
        f.write('Mean Absolute Error (MAE): {:.2f}\n'.format(mae_total))
        f.write('Mean Absolute Percentage Error (MAPE): {:.2f}%\n'.format(mape_total))

        f.write('\nNormal Region Error Metrics:\n')
        f.write('Root Mean Squared Error (RMSE): {:.2f}\n'.format(rmse_normal))
        f.write('Mean Squared Error (MSE): {:.2f}\n'.format(mse_normal))
        f.write('Mean Absolute Error (MAE): {:.2f}\n'.format(mae_normal))
        f.write('Mean Absolute Percentage Error (MAPE): {:.2f}%\n'.format(mape_normal))

        f.write('\ndysglycemic Region Error Metrics:\n')
        f.write('Root Mean Squared Error (RMSE): {:.2f}\n'.format(rmse_dysglycemic))
        f.write('Mean Squared Error (MSE): {:.2f}\n'.format(mse_dysglycemic))
        f.write('Mean Absolute Error (MAE): {:.2f}\n'.format(mae_dysglycemic))
        f.write('Mean Absolute Percentage Error (MAPE): {:.2f}%\n'.format(mape_dysglycemic))

        f.write('\nHyperglycemia Region Error Metrics:\n')
        f.write('Root Mean Squared Error (RMSE): {:.2f}\n'.format(rmse_hyperglycemia))
        f.write('Mean Squared Error (MSE): {:.2f}\n'.format(mse_hyperglycemia))
        f.write('Mean Absolute Error (MAE): {:.2f}\n'.format(mae_hyperglycemia))
        f.write('Mean Absolute Percentage Error (MAPE): {:.2f}%\n'.format(mape_hyperglycemia))

        f.write('\nHypoglycemia Region Error Metrics:\n')
        f.write('Root Mean Squared Error (RMSE): {:.2f}\n'.format(rmse_hypoglycemia))
        f.write('Mean Squared Error (MSE): {:.2f}\n'.format(mse_hypoglycemia))
        f.write('Mean Absolute Error (MAE): {:.2f}\n'.format(mae_hypoglycemia))
        f.write('Mean Absolute Percentage Error (MAPE): {:.2f}%\n'.format(mape_hypoglycemia))

        # Add zone distribution percentages
        if zones is not None:
            f.write('\nClark Error Zone Distribution:\n')
            f.write('Zone A: {:.2f}% (STD: {:.2f}%)\n'.format(zone_A_mean, zone_A_std))
            f.write('Zone B: {:.2f}% (STD: {:.2f}%)\n'.format(zone_B_mean, zone_B_std))
            f.write('Zone C: {:.2f}% (STD: {:.2f}%)\n'.format(zone_C_mean, zone_C_std))
            f.write('Zone D: {:.2f}% (STD: {:.2f}%)\n'.format(zone_D_mean, zone_D_std))
            f.write('Zone E: {:.2f}% (STD: {:.2f}%)\n'.format(zone_E_mean, zone_E_std))

    return (rmse_total, mse_total, mae_total, mape_total), (rmse_normal, mse_normal, mae_normal, mape_normal), (
        rmse_dysglycemic, mse_dysglycemic, mae_dysglycemic, mape_dysglycemic), (
        rmse_hyperglycemia, mse_hyperglycemia, mae_hyperglycemia, mape_hyperglycemia), (
        rmse_hypoglycemia, mse_hypoglycemia, mae_hypoglycemia, mape_hypoglycemia)


def calculate_scores(save_to, y_pred, y_true, file_name='scores.txt'):
    # Flatten the data to make it easier to work with
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    # Convert predictions and actual values to binary indicators of states
    predicted_hyperglycemia = y_pred_flat > Threshold.HYPERGLYCEMIA
    actual_hyperglycemia = y_true_flat > Threshold.HYPERGLYCEMIA

    predicted_hypoglycemia = y_pred_flat < Threshold.HYPOGLYCEMIA
    actual_hypoglycemia = y_true_flat < Threshold.HYPOGLYCEMIA

    # dysglycemic region indicators
    dysglycemic_mask = (y_true_flat < Threshold.HYPOGLYCEMIA) | (y_true_flat > Threshold.HYPERGLYCEMIA)
    normal_mask = (y_true_flat >= Threshold.HYPOGLYCEMIA) & (y_true_flat <= Threshold.HYPERGLYCEMIA)

    # Calculate True Positives, False Positives, and False Negatives for hyperglycemia
    true_positives_hyper = np.sum(predicted_hyperglycemia & actual_hyperglycemia)
    false_positives_hyper = np.sum(predicted_hyperglycemia & ~actual_hyperglycemia)
    false_negatives_hyper = np.sum(~predicted_hyperglycemia & actual_hyperglycemia)

    # Calculate precision, recall, and F1 for hyperglycemia
    precision_hyper = true_positives_hyper / (true_positives_hyper + false_positives_hyper) if (
                                                                                                       true_positives_hyper + false_positives_hyper) > 0 else np.nan
    recall_hyper = true_positives_hyper / (true_positives_hyper + false_negatives_hyper) if (
                                                                                                    true_positives_hyper + false_negatives_hyper) > 0 else np.nan
    f1_score_hyper = 2 * (precision_hyper * recall_hyper) / (precision_hyper + recall_hyper) if (
                                                                                                        precision_hyper + recall_hyper) > 0 else np.nan

    # Calculate True Positives, False Positives, and False Negatives for hypoglycemia
    true_positives_hypo = np.sum(predicted_hypoglycemia & actual_hypoglycemia)
    false_positives_hypo = np.sum(predicted_hypoglycemia & ~actual_hypoglycemia)
    false_negatives_hypo = np.sum(~predicted_hypoglycemia & actual_hypoglycemia)

    # Calculate precision, recall, and F1 for hypoglycemia
    precision_hypo = true_positives_hypo / (true_positives_hypo + false_positives_hypo) if (
                                                                                                   true_positives_hypo + false_positives_hypo) > 0 else np.nan
    recall_hypo = true_positives_hypo / (true_positives_hypo + false_negatives_hypo) if (
                                                                                                true_positives_hypo + false_negatives_hypo) > 0 else np.nan
    f1_score_hypo = 2 * (precision_hypo * recall_hypo) / (precision_hypo + recall_hypo) if (
                                                                                                   precision_hypo + recall_hypo) > 0 else np.nan

    # Calculate metrics for the normal region
    y_true_normal = y_true_flat[normal_mask]
    y_pred_normal = y_pred_flat[normal_mask]

    true_positives_normal = np.sum((y_pred_normal >= Threshold.HYPOGLYCEMIA) & (
            y_pred_normal <= Threshold.HYPERGLYCEMIA))
    false_positives_normal = np.sum((y_pred_normal > Threshold.HYPERGLYCEMIA) | (
            y_pred_normal < Threshold.HYPOGLYCEMIA))
    false_negatives_normal = np.sum((y_true_normal > Threshold.HYPERGLYCEMIA) | (
            y_true_normal < Threshold.HYPOGLYCEMIA))

    precision_normal = true_positives_normal / (true_positives_normal + false_positives_normal) if (
                                                                                                           true_positives_normal + false_positives_normal) > 0 else np.nan
    recall_normal = true_positives_normal / (true_positives_normal + false_negatives_normal) if (
                                                                                                        true_positives_normal + false_negatives_normal) > 0 else np.nan
    f1_score_normal = 2 * (precision_normal * recall_normal) / (precision_normal + recall_normal) if (
                                                                                                             precision_normal + recall_normal) > 0 else np.nan

    # Calculate metrics for the dysglycemic regions
    y_true_dysglycemic = y_true_flat[dysglycemic_mask]
    y_pred_dysglycemic = y_pred_flat[dysglycemic_mask]

    true_positives_dysglycemic = np.sum((y_pred_dysglycemic < Threshold.HYPOGLYCEMIA) | (
            y_pred_dysglycemic > Threshold.HYPERGLYCEMIA))
    false_positives_dysglycemic = np.sum((y_pred_dysglycemic >= Threshold.HYPOGLYCEMIA) & (
            y_pred_dysglycemic <= Threshold.HYPERGLYCEMIA))
    false_negatives_dysglycemic = np.sum((y_true_dysglycemic >= Threshold.HYPOGLYCEMIA) & (
            y_true_dysglycemic <= Threshold.HYPERGLYCEMIA))

    precision_dysglycemic = true_positives_dysglycemic / (true_positives_dysglycemic + false_positives_dysglycemic) if (
                                                                                                                   true_positives_dysglycemic + false_positives_dysglycemic) > 0 else np.nan
    recall_dysglycemic = true_positives_dysglycemic / (true_positives_dysglycemic + false_negatives_dysglycemic) if (
                                                                                                                true_positives_dysglycemic + false_negatives_dysglycemic) > 0 else np.nan
    f1_score_dysglycemic = 2 * (precision_dysglycemic * recall_dysglycemic) / (precision_dysglycemic + recall_dysglycemic) if (
                                                                                                                       precision_dysglycemic + recall_dysglycemic) > 0 else np.nan

    print(f"\nF1 score for normal region: {f1_score_normal:.2f}")
    print(f"Precision for normal region: {precision_normal:.2f}")
    print(f"Recall for normal region: {recall_normal:.2f}")

    print(f"\nF1 score for dysglycemic regions: {f1_score_dysglycemic:.2f}")
    print(f"Precision for dysglycemic regions: {precision_dysglycemic:.2f}")
    print(f"Recall for dysglycemic regions: {recall_dysglycemic:.2f}")

    print(f"\nF1 score for hyperglycemia: {f1_score_hyper:.2f}")
    print(f"Precision for hyperglycemia: {precision_hyper:.2f}")
    print(f"Recall for hyperglycemia: {recall_hyper:.2f}")

    print(f"\nF1 score for hypoglycemia: {f1_score_hypo:.2f}")
    print(f"Precision for hypoglycemia: {precision_hypo:.2f}")
    print(f"Recall for hypoglycemia: {recall_hypo:.2f}")

    with open(save_to + file_name, "w") as file:
        file.write(f"F1 score for normal region: {f1_score_normal:.2f}\n")
        file.write(f"Precision for normal region: {precision_normal:.2f}\n")
        file.write(f"Recall for normal region: {recall_normal:.2f}\n")

        file.write(f"\nF1 score for dysglycemic regions: {f1_score_dysglycemic:.2f}\n")
        file.write(f"Precision for dysglycemic regions: {precision_dysglycemic:.2f}\n")
        file.write(f"Recall for dysglycemic regions: {recall_dysglycemic:.2f}\n")

        file.write(f"\nF1 score for hyperglycemia: {f1_score_hyper:.2f}\n")
        file.write(f"Precision for hyperglycemia: {precision_hyper:.2f}\n")
        file.write(f"Recall for hyperglycemia: {recall_hyper:.2f}\n")

        file.write(f"\nF1 score for hypoglycemia: {f1_score_hypo:.2f}\n")
        file.write(f"Precision for hypoglycemia: {precision_hypo:.2f}\n")
        file.write(f"Recall for hypoglycemia: {recall_hypo:.2f}\n")

    return (f1_score_hyper, f1_score_hypo, f1_score_normal, f1_score_dysglycemic, precision_hyper, precision_hypo,
            precision_normal, precision_dysglycemic, recall_hyper, recall_hypo, recall_normal, recall_dysglycemic)


def calculate_correlation_coefficient(save_to, patient_id, y_true, y_pred, length, show=False):
    correlations = [pearsonr(x=y_pred[:, i], y=y_true[:, i])[0] for i in range(length)]
    correlation_matrix = np.array(correlations).reshape(1, length)

    time_labels = [f'{5 * (i + 1)}' for i in range(length)]

    plt.figure(figsize=(10.5, 1.75))
    ax = sns.heatmap(correlation_matrix,
                     annot=True,
                     cmap='coolwarm',
                     cbar=True,
                     xticklabels=time_labels,
                     yticklabels=['Correlation'])

    plt.title('Correlations Between Real and Predicted Values' + ' / ' + DATASET_NAME + ' dataset' + ' / ' + 'Patient ID: ' + patient_id + '\n',
              fontsize=12)
    ax.set_xlabel('Prediction Horizon (min)', fontsize=12, labelpad=10)
    ax.set_ylabel('', fontsize=12, labelpad=10)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    for text in ax.texts:
        text.set_fontsize(12)
    plt.tight_layout()

    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + 'correlation' + '.png'), dpi=450)
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + 'correlation' + '.pdf'), dpi=450)

    if show is True:
        plt.show()
    plt.close()

    return correlation_matrix


def calculate_total_error_metrics_and_scores(save_to,
                                             list_rmse_total,
                                             list_mse_total,
                                             list_mae_total,
                                             list_mape_total,
                                             list_f1_score_hypo,
                                             list_f1_score_hyper,
                                             list_f1_score_normal,
                                             list_f1_score_dysglycemic,
                                             list_precision_hyper,
                                             list_precision_hypo,
                                             list_precision_normal,
                                             list_precision_dysglycemic,
                                             list_recall_hyper,
                                             list_recall_hypo,
                                             list_recall_normal,
                                             list_recall_dysglycemic,
                                             list_rmse_normal,
                                             list_mse_normal,
                                             list_mae_normal,
                                             list_mape_normal,
                                             list_rmse_dysglycemic,
                                             list_mse_dysglycemic,
                                             list_mae_dysglycemic,
                                             list_mape_dysglycemic,
                                             list_rmse_hyperglycemia,
                                             list_mse_hyperglycemia,
                                             list_mae_hyperglycemia,
                                             list_mape_hyperglycemia,
                                             list_rmse_hypoglycemia,
                                             list_mse_hypoglycemia,
                                             list_mae_hypoglycemia,
                                             list_mape_hypoglycemia,
                                             list_corr,
                                             list_clarke_error_zones,
                                             file_name='total_error_metrics.txt'):
    def calculate_mean_std(data):
        data = [x for x in data if not np.isnan(x)]
        return np.mean(data), np.std(data)

    mean_rmse_total, std_rmse_total = calculate_mean_std(list_rmse_total)
    mean_mse_total, std_mse_total = calculate_mean_std(list_mse_total)
    mean_mae_total, std_mae_total = calculate_mean_std(list_mae_total)
    mean_mape_total, std_mape_total = calculate_mean_std(list_mape_total)

    mean_f1_score_hypo, std_f1_score_hypo = calculate_mean_std(list_f1_score_hypo)
    mean_f1_score_hyper, std_f1_score_hyper = calculate_mean_std(list_f1_score_hyper)
    mean_f1_score_normal, std_f1_score_normal = calculate_mean_std(list_f1_score_normal)
    mean_f1_score_dysglycemic, std_f1_score_dysglycemic = calculate_mean_std(list_f1_score_dysglycemic)

    mean_precision_hyper, std_precision_hyper = calculate_mean_std(list_precision_hyper)
    mean_precision_hypo, std_precision_hypo = calculate_mean_std(list_precision_hypo)
    mean_precision_normal, std_precision_normal = calculate_mean_std(list_precision_normal)
    mean_precision_dysglycemic, std_precision_dysglycemic = calculate_mean_std(list_precision_dysglycemic)

    mean_recall_hyper, std_recall_hyper = calculate_mean_std(list_recall_hyper)
    mean_recall_hypo, std_recall_hypo = calculate_mean_std(list_recall_hypo)
    mean_recall_normal, std_recall_normal = calculate_mean_std(list_recall_normal)
    mean_recall_dysglycemic, std_recall_dysglycemic = calculate_mean_std(list_recall_dysglycemic)

    mean_rmse_normal, std_rmse_normal = calculate_mean_std(list_rmse_normal)
    mean_mse_normal, std_mse_normal = calculate_mean_std(list_mse_normal)
    mean_mae_normal, std_mae_normal = calculate_mean_std(list_mae_normal)
    mean_mape_normal, std_mape_normal = calculate_mean_std(list_mape_normal)

    mean_rmse_dysglycemic, std_rmse_dysglycemic = calculate_mean_std(list_rmse_dysglycemic)
    mean_mse_dysglycemic, std_mse_dysglycemic = calculate_mean_std(list_mse_dysglycemic)
    mean_mae_dysglycemic, std_mae_dysglycemic = calculate_mean_std(list_mae_dysglycemic)
    mean_mape_dysglycemic, std_mape_dysglycemic = calculate_mean_std(list_mape_dysglycemic)

    mean_rmse_hyperglycemia, std_rmse_hyperglycemia = calculate_mean_std(list_rmse_hyperglycemia)
    mean_mse_hyperglycemia, std_mse_hyperglycemia = calculate_mean_std(list_mse_hyperglycemia)
    mean_mae_hyperglycemia, std_mae_hyperglycemia = calculate_mean_std(list_mae_hyperglycemia)
    mean_mape_hyperglycemia, std_mape_hyperglycemia = calculate_mean_std(list_mape_hyperglycemia)

    mean_rmse_hypoglycemia, std_rmse_hypoglycemia = calculate_mean_std(list_rmse_hypoglycemia)
    mean_mse_hypoglycemia, std_mse_hypoglycemia = calculate_mean_std(list_mse_hypoglycemia)
    mean_mae_hypoglycemia, std_mae_hypoglycemia = calculate_mean_std(list_mae_hypoglycemia)
    mean_mape_hypoglycemia, std_mape_hypoglycemia = calculate_mean_std(list_mape_hypoglycemia)

    mean_corr = np.nanmean(list_corr)
    std_corr = np.nanstd(list_corr)

    zone_A_mean, zone_A_std = calculate_mean_std(list_clarke_error_zones['A'])
    zone_B_mean, zone_B_std = calculate_mean_std(list_clarke_error_zones['B'])
    zone_C_mean, zone_C_std = calculate_mean_std(list_clarke_error_zones['C'])
    zone_D_mean, zone_D_std = calculate_mean_std(list_clarke_error_zones['D'])
    zone_E_mean, zone_E_std = calculate_mean_std(list_clarke_error_zones['E'])

    print("\n----------------------- Avg. Metrics for All Patients -----------------------")
    print('Average RMSE Total: {:.2f} (STD: {:.2f})'.format(mean_rmse_total, std_rmse_total))
    print('Average MSE Total: {:.2f} (STD: {:.2f})'.format(mean_mse_total, std_mse_total))
    print('Average MAE Total: {:.2f} (STD: {:.2f})'.format(mean_mae_total, std_mae_total))
    print('Average MAPE Total: {:.2f}% (STD: {:.2f}%)'.format(mean_mape_total, std_mape_total))

    print('\nAverage RMSE Normal: {:.2f} (STD: {:.2f})'.format(mean_rmse_normal, std_rmse_normal))
    print('Average MSE Normal: {:.2f} (STD: {:.2f})'.format(mean_mse_normal, std_mse_normal))
    print('Average MAE Normal: {:.2f} (STD: {:.2f})'.format(mean_mae_normal, std_mae_normal))
    print('Average MAPE Normal: {:.2f}% (STD: {:.2f}%)'.format(mean_mape_normal, std_mape_normal))

    print('\nAverage RMSE dysglycemic: {:.2f} (STD: {:.2f})'.format(mean_rmse_dysglycemic, std_rmse_dysglycemic))
    print('Average MSE dysglycemic: {:.2f} (STD: {:.2f})'.format(mean_mse_dysglycemic, std_mse_dysglycemic))
    print('Average MAE dysglycemic: {:.2f} (STD: {:.2f})'.format(mean_mae_dysglycemic, std_mae_dysglycemic))
    print('Average MAPE dysglycemic: {:.2f}% (STD: {:.2f}%)'.format(mean_mape_dysglycemic, std_mape_dysglycemic))

    print('\nAverage RMSE Hyperglycemia: {:.2f} (STD: {:.2f})'.format(mean_rmse_hyperglycemia, std_rmse_hyperglycemia))
    print('Average MSE Hyperglycemia: {:.2f} (STD: {:.2f})'.format(mean_mse_hyperglycemia, std_mse_hyperglycemia))
    print('Average MAE Hyperglycemia: {:.2f} (STD: {:.2f})'.format(mean_mae_hyperglycemia, std_mae_hyperglycemia))
    print('Average MAPE Hyperglycemia: {:.2f}% (STD: {:.2f}%)'.format(mean_mape_hyperglycemia, std_mape_hyperglycemia))

    print('\nAverage RMSE Hypoglycemia: {:.2f} (STD: {:.2f})'.format(mean_rmse_hypoglycemia, std_rmse_hypoglycemia))
    print('Average MSE Hypoglycemia: {:.2f} (STD: {:.2f})'.format(mean_mse_hypoglycemia, std_mse_hypoglycemia))
    print('Average MAE Hypoglycemia: {:.2f} (STD: {:.2f})'.format(mean_mae_hypoglycemia, std_mae_hypoglycemia))
    print('Average MAPE Hypoglycemia: {:.2f}% (STD: {:.2f}%)'.format(mean_mape_hypoglycemia, std_mape_hypoglycemia))

    print('\nAverage F1 Score for Normal: {:.2f} (STD: {:.2f})'.format(mean_f1_score_normal, std_f1_score_normal))
    print('Average Precision for Normal: {:.2f} (STD: {:.2f})'.format(mean_precision_normal, std_precision_normal))
    print('Average Recall for Normal: {:.2f} (STD: {:.2f})'.format(mean_recall_normal, std_recall_normal))

    print('\nAverage F1 Score for dysglycemic: {:.2f} (STD: {:.2f})'.format(mean_f1_score_dysglycemic, std_f1_score_dysglycemic))
    print('Average Precision for dysglycemic: {:.2f} (STD: {:.2f})'.format(mean_precision_dysglycemic,
                                                                        std_precision_dysglycemic))
    print('Average Recall for dysglycemic: {:.2f} (STD: {:.2f})'.format(mean_recall_dysglycemic, std_recall_dysglycemic))

    print('\nAverage F1 Score for Hypoglycemia: {:.2f} (STD: {:.2f})'.format(mean_f1_score_hypo, std_f1_score_hypo))
    print('Average Precision for Hyperglycemia: {:.2f} (STD: {:.2f})'.format(mean_precision_hyper, std_precision_hyper))
    print('Average Recall for Hyperglycemia: {:.2f} (STD: {:.2f})'.format(mean_recall_hyper, std_recall_hyper))

    print('\nAverage F1 Score for Hyperglycemia: {:.2f} (STD: {:.2f})'.format(mean_f1_score_hyper, std_f1_score_hyper))
    print('Average Precision for Hypoglycemia: {:.2f} (STD: {:.2f})'.format(mean_precision_hypo, std_precision_hypo))
    print('Average Recall for Hypoglycemia: {:.2f} (STD: {:.2f})'.format(mean_recall_hypo, std_recall_hypo))

    print('\nAverage Correlation Between Real and Predicted Values: {:.2f}'.format(mean_corr))
    print('STD Correlation Between Real and Predicted Values: {:.2f}'.format(std_corr))

    # Print zone distribution
    print('\nClark Error Zone Distribution:')
    print('Zone A: {:.2f}% (STD: {:.2f}%)'.format(zone_A_mean, zone_A_std))
    print('Zone B: {:.2f}% (STD: {:.2f}%)'.format(zone_B_mean, zone_B_std))
    print('Zone C: {:.2f}% (STD: {:.2f}%)'.format(zone_C_mean, zone_C_std))
    print('Zone D: {:.2f}% (STD: {:.2f}%)'.format(zone_D_mean, zone_D_std))
    print('Zone E: {:.2f}% (STD: {:.2f}%)'.format(zone_E_mean, zone_E_std))

    with open(save_to + file_name, 'w') as f:
        f.write('----------------------- Avg. Metrics for All Patients -----------------------\n')
        f.write('Average RMSE Total: {:.2f} (STD: {:.2f})\n'.format(mean_rmse_total, std_rmse_total))
        f.write('Average MSE Total: {:.2f} (STD: {:.2f})\n'.format(mean_mse_total, std_mse_total))
        f.write('Average MAE Total: {:.2f} (STD: {:.2f})\n'.format(mean_mae_total, std_mae_total))
        f.write('Average MAPE Total: {:.2f}% (STD: {:.2f}%)\n'.format(mean_mape_total, std_mape_total))
        f.write('\n')

        f.write('Average RMSE Normal: {:.2f} (STD: {:.2f})\n'.format(mean_rmse_normal, std_rmse_normal))
        f.write('Average MSE Normal: {:.2f} (STD: {:.2f})\n'.format(mean_mse_normal, std_mse_normal))
        f.write('Average MAE Normal: {:.2f} (STD: {:.2f})\n'.format(mean_mae_normal, std_mae_normal))
        f.write('Average MAPE Normal: {:.2f}% (STD: {:.2f}%)\n'.format(mean_mape_normal, std_mape_normal))
        f.write('\n')

        f.write('Average RMSE dysglycemic: {:.2f} (STD: {:.2f})\n'.format(mean_rmse_dysglycemic, std_rmse_dysglycemic))
        f.write('Average MSE dysglycemic: {:.2f} (STD: {:.2f})\n'.format(mean_mse_dysglycemic, std_mse_dysglycemic))
        f.write('Average MAE dysglycemic: {:.2f} (STD: {:.2f})\n'.format(mean_mae_dysglycemic, std_mae_dysglycemic))
        f.write('Average MAPE dysglycemic: {:.2f}% (STD: {:.2f}%)\n'.format(mean_mape_dysglycemic, std_mape_dysglycemic))
        f.write('\n')

        f.write('Average RMSE Hyperglycemia: {:.2f} (STD: {:.2f})\n'.format(mean_rmse_hyperglycemia,
                                                                            std_rmse_hyperglycemia))
        f.write('Average MSE Hyperglycemia: {:.2f} (STD: {:.2f})\n'.format(mean_mse_hyperglycemia,
                                                                           std_mse_hyperglycemia))
        f.write('Average MAE Hyperglycemia: {:.2f} (STD: {:.2f})\n'.format(mean_mae_hyperglycemia,
                                                                           std_mae_hyperglycemia))
        f.write('Average MAPE Hyperglycemia: {:.2f}% (STD: {:.2f}%)\n'.format(mean_mape_hyperglycemia,
                                                                              std_mape_hyperglycemia))
        f.write('\n')

        f.write('Average RMSE Hypoglycemia: {:.2f} (STD: {:.2f})\n'.format(mean_rmse_hypoglycemia,
                                                                           std_rmse_hypoglycemia))
        f.write('Average MSE Hypoglycemia: {:.2f} (STD: {:.2f})\n'.format(mean_mse_hypoglycemia, std_mse_hypoglycemia))
        f.write('Average MAE Hypoglycemia: {:.2f} (STD: {:.2f})\n'.format(mean_mae_hypoglycemia, std_mae_hypoglycemia))
        f.write('Average MAPE Hypoglycemia: {:.2f}% (STD: {:.2f}%)\n'.format(mean_mape_hypoglycemia,
                                                                             std_mape_hypoglycemia))
        f.write('\n')

        f.write('Average F1 Score for Normal: {:.2f} (STD: {:.2f})\n'.format(mean_f1_score_normal, std_f1_score_normal))
        f.write('Average Precision for Normal: {:.2f} (STD: {:.2f})\n'.format(mean_precision_normal,
                                                                              std_precision_normal))
        f.write('Average Recall for Normal: {:.2f} (STD: {:.2f})\n'.format(mean_recall_normal, std_recall_normal))
        f.write('\n')

        f.write('Average F1 Score for dysglycemic: {:.2f} (STD: {:.2f})\n'.format(mean_f1_score_dysglycemic,
                                                                               std_f1_score_dysglycemic))
        f.write('Average Precision for dysglycemic: {:.2f} (STD: {:.2f})\n'.format(mean_precision_dysglycemic,
                                                                                std_precision_dysglycemic))
        f.write('Average Recall for dysglycemic: {:.2f} (STD: {:.2f})\n'.format(mean_recall_dysglycemic, std_recall_dysglycemic))
        f.write('\n')

        f.write('Average F1 Score for Hypoglycemia: {:.2f} (STD: {:.2f})\n'.format(mean_f1_score_hypo,
                                                                                   std_f1_score_hypo))
        f.write('Average Precision for Hypoglycemia: {:.2f} (STD: {:.2f})\n'.format(mean_precision_hypo,
                                                                                    std_precision_hypo))
        f.write('Average Recall for Hypoglycemia: {:.2f} (STD: {:.2f})\n'.format(mean_recall_hypo, std_recall_hypo))
        f.write('\n')

        f.write('Average F1 Score for Hyperglycemia: {:.2f} (STD: {:.2f})\n'.format(mean_f1_score_hyper,
                                                                                    std_f1_score_hyper))
        f.write('Average Precision for Hyperglycemia: {:.2f} (STD: {:.2f})\n'.format(mean_precision_hyper,
                                                                                     std_precision_hyper))
        f.write('Average Recall for Hyperglycemia: {:.2f} (STD: {:.2f})\n'.format(mean_recall_hyper, std_recall_hyper))
        f.write('\n')

        f.write('Average Correlation Between Real and Predicted Values: {:.2f}\n'.format(mean_corr))
        f.write('STD Correlation Between Real and Predicted Values: {:.2f}\n'.format(std_corr))
        f.write('\n')

        # Add zone distribution percentages
        f.write('Clark Error Zone Distribution:\n')
        f.write('Zone A: {:.2f}% (STD: {:.2f}%)\n'.format(zone_A_mean, zone_A_std))
        f.write('Zone B: {:.2f}% (STD: {:.2f}%)\n'.format(zone_B_mean, zone_B_std))
        f.write('Zone C: {:.2f}% (STD: {:.2f}%)\n'.format(zone_C_mean, zone_C_std))
        f.write('Zone D: {:.2f}% (STD: {:.2f}%)\n'.format(zone_D_mean, zone_D_std))
        f.write('Zone E: {:.2f}% (STD: {:.2f}%)\n'.format(zone_E_mean, zone_E_std))


def plot_train_val_history(save_to, patient_id, history, show=False):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss', color='c')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='m')
    plt.title('Train VS. Validation Loss' + ' / ' + DATASET_NAME + ' dataset' + ' / ' + 'Patient ID: ' + patient_id + '\n',
              fontsize=12)
    plt.xlabel('Epochs', fontsize=12, labelpad=10)
    plt.ylabel('Loss', fontsize=12, labelpad=10)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + 'loss' + '.png'), dpi=450)
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + 'loss' + '.pdf'), dpi=450)

    if show is True:
        plt.show()
    plt.close()


def plot_prediction_results(save_to, patient_id, y_true, y_pred, rmse, mse, mae, mape, show=False):
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.1])

    # Main plot
    ax_main = plt.subplot(gs[0])
    ax_main.plot(y_true[:, 0], label='Real', color='c')
    ax_main.plot(y_pred[:, 0], label='Predicted', color='m')
    ax_main.axhline(Threshold.HYPERGLYCEMIA, color='black', linestyle='--', label='Hyperglycemia Threshold')
    ax_main.axhline(Threshold.HYPOGLYCEMIA, color='gray', linestyle='--', label='Hypoglycemia Threshold')
    ax_main.set_xlabel('Time Samples (Number of Data Points)', fontsize=14, labelpad=10)
    ax_main.set_ylabel('Glucose Level (mg/dl)', fontsize=14, labelpad=10)
    ax_main.tick_params(axis='x', labelsize=14)
    ax_main.tick_params(axis='y', labelsize=14)
    ax_main.set_title('Real VS. Predicted Glucose Level Values During the Time' + ' / ' + DATASET_NAME + ' dataset' + ' / ' + 'Patient ID: ' + patient_id + '\n',
                      fontsize=14)
    ax_main.legend(fontsize=14, loc='upper right')

    # Text box
    ax_text = plt.subplot(gs[1])
    ax_text.axis('off')

    error_text = f'RMSE: {rmse:.2f}\nMSE: {mse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%'
    ax_text.text(0, 0, error_text, fontsize=12, bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + 'prediction' + '.png'), dpi=450)
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + 'prediction' + '.pdf'), dpi=450)

    if show is True:
        plt.show()
    plt.close()


def plot_fitness_evolution(save_to, patient_id, all_fitness_scores, show=False):
    mean_fitness = [np.mean(scores) for scores in all_fitness_scores]
    min_fitness = [np.min(scores) for scores in all_fitness_scores]
    max_fitness = [np.max(scores) for scores in all_fitness_scores]
    std_fitness = [np.std(scores) for scores in all_fitness_scores]
    generations = range(1, len(all_fitness_scores) + 1)

    plt.figure(figsize=(12, 8))
    plt.fill_between(generations,
                     np.array(mean_fitness) - np.array(std_fitness),
                     np.array(mean_fitness) + np.array(std_fitness),
                     color='gray',
                     alpha=0.5,
                     label='STD Range')
    plt.plot(generations, mean_fitness, 'k-', label='Average RMSE')
    plt.plot(generations, max_fitness, 'c--', label='Max RMSE')
    plt.plot(generations, min_fitness, 'm--', label='Min RMSE')
    plt.title('Root Mean Squared Error (RMSE) per Generation' + ' / ' + DATASET_NAME + ' dataset' + ' / ' + 'Patient ID: ' + patient_id + '\n',
              fontsize=14)
    plt.xlabel('Generation', fontsize=14, labelpad=10)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=14, labelpad=10)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + 'evolution' + '.png'), dpi=450)
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + 'evolution' + '.pdf'), dpi=450)

    if show is True:
        plt.show()
    plt.close()


def plot_contour(save_to, patient_id, population, fitness_scores, best_individuals, show=False):
    # Create a grid of values
    xi = np.linspace(population[:, 0].min(), population[:, 0].max(), 100)
    yi = np.linspace(population[:, 1].min(), population[:, 1].max(), 100)

    # Interpolate the RMSE values
    zi = scipy.interpolate.griddata(population, fitness_scores, (xi[None, :], yi[:, None]), method='cubic')

    # Handling NaN values in the interpolated results
    zi[np.isnan(zi)] = np.nanmax(zi)  # Replace NaNs with the maximum of non-NaN values

    # First plot: Scatter with color coding
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(population[:, 0], population[:, 1], c=fitness_scores, cmap='rainbow', s=50)

    cbar = plt.colorbar(scatter)
    cbar.ax.set_ylabel('RMSE', fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    plt.title(f'Best Weights Pair: [{best_individuals[0]:.2f}, {best_individuals[1]:.2f}]\n\n2D Scatter of Error Weights with RMSE' + ' / ' + DATASET_NAME + ' dataset' + ' / ' + 'Patient ID: ' + patient_id + '\n',
              fontsize=14)

    plt.xlabel('w_hypo', fontsize=14, labelpad=20)
    plt.ylabel('w_hyper', fontsize=14, labelpad=20)
    plt.tick_params(labelsize=14)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + '2d_scatter' + '.png'), dpi=450)
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + '2d_scatter' + '.pdf'), dpi=450)

    if show is True:
        plt.show()
    plt.close()

    # Second plot: Contour plot of interpolated RMSE
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xi, yi, zi, levels=50, cmap='viridis')
    cbar = plt.colorbar(contour)
    cbar.ax.set_ylabel('RMSE', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    plt.scatter(population[:, 0], population[:, 1], color='red', marker='o', s=50)
    plt.title(f'Best Weights Pair: [{best_individuals[0]:.2f}, {best_individuals[1]:.2f}]\n\nContour Plot of Error Weights with RMSE' + ' / ' + DATASET_NAME + ' dataset' + ' / ' + 'Patient ID: ' + patient_id + '\n',
              fontsize=14)
    plt.xlabel('w_hypo', fontsize=14, labelpad=20)
    plt.ylabel('w_hyper', fontsize=14, labelpad=20)
    plt.tick_params(labelsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + 'contour_plot' + '.png'), dpi=450)
    plt.savefig(str(save_to + DATASET_NAME + '_' + patient_id + '_' + 'contour_plot' + '.pdf'), dpi=450)

    if show is True:
        plt.show()
    plt.close()
