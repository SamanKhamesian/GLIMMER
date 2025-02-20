# Extract and Link all Data to a dataframe.
# Chirath Hettiarachchi
# ANU - OHIOH Group.
# October 2020

import os
from loader import load_data
from options import Options
import numpy as np
np.random.seed(7)

import warnings
warnings.simplefilter('ignore', Warning)


def convert(o):
    if isinstance(o, np.int64): return int(o)


def extract_data(args, mode='train', files='', version='new', patient_index=None):
    print('\nPreparing to extract data...')
    sample_rate = 5
    count = 0
    for patient in files:
        patient_id = patient_index[count]
        print('\nData exporting for subject {}'.format(patient_id))
        print('Dataset version: ' + version + ', Mode: ' + mode)
        print('Data folder: ' + patient)
        df = load_data(patient)

        # Data pre-processing will be carried out.
        df['glucose'] = df['glucose'].fillna(-1)
        df['gsr'] = df['gsr'].fillna(-1)
        df['hr'] = df['hr'].fillna(-1)
        df['st'] = df['st'].fillna(-1)
        df['basal'] = df['basal'].fillna(method='ffill')

        df['bolus'] = df['bolus'].fillna(-1)
        df['bolus_dur'] = df['bolus_dur'].fillna(-1)
        df['bolus_end'] = df['bolus_end'].fillna(-1)
        df['temp_basal'] = df['temp_basal'].fillna(-1)
        df['basal_end'] = df['basal_end'].fillna(-1)
        df['carbs'] = df['carbs'].fillna(-1)
        df['meal_type'] = df['meal_type'].fillna(-1)

        df['sleep'] = df['sleep'].fillna(-1)
        df['sleep_dur'] = df['sleep_dur'].fillna(-1)
        df['sleep_end'] = df['sleep_end'].fillna(-1)
        df['work'] = df['work'].fillna(-1)
        df['work_dur'] = df['work_dur'].fillna(-1)
        df['work_end'] = df['work_end'].fillna(-1)
        df['exercise_intensity'] = df['exercise_intensity'].fillna(-1)
        df['exer_dur'] = df['exer_dur'].fillna(-1)

        df = df.dropna()

        df['index'] = df.index
        df['index_new'] = df.index
        df['temp_bolus'] = df['bolus']
        df['temp_sleep'] = df['sleep']
        df['temp_work'] = df['work']
        df['temp_exercise_intensity'] = df['exercise_intensity']
        df['missing'] = -1

        # remove the duplicate rows by replacing max
        df = df.groupby(df['index_new']).max()
        # df.to_csv('extracted_data_Oct28/' + str(patient_id) + 'debug_' + mode + '.csv')

        # Overwrite basal with OhioT1DM_CSV basal values.
        for i in range(0, len(df)):
            # overwrite basal with OhioT1DM_CSV basal
            if df['temp_basal'][i] != -1:
                basal_timeflag = df['basal_end'][i]
                basal_loop = True
                j = i
                while (basal_loop):
                    if df['index'][j] == basal_timeflag:
                        df['basal'][j] = df['temp_basal'][i]
                        basal_loop = False
                    elif df['index'][j] < basal_timeflag:
                        df['basal'][j] = df['temp_basal'][i]
                        j = j + 1
                        if j > len(df) - 1:
                            basal_loop = False
                    elif df['index'][j] > basal_timeflag:
                        basal_loop = False

        # Check how bolus is applied.
        for i in range(0, len(df)):
            if df['temp_bolus'][i] != -1:
                bolus_timeflag = df['bolus_end'][i]
                bolus_loop = True
                j = i
                while (bolus_loop):
                    if df['index'][j] == bolus_timeflag:
                        df['bolus'][j] = df['bolus'][i]
                        bolus_loop = False
                    elif df['index'][j] < bolus_timeflag:
                        df['bolus'][j] = df['bolus'][i]
                        j = j + 1
                    elif df['index'][j] > bolus_timeflag:
                        bolus_loop = False

        # Check sleep
        for i in range(0, len(df)):
            if df['temp_sleep'][i] != -1:
                sleep_timeflag = df['sleep_end'][i]
                sleep_loop = True
                j = i
                while (sleep_loop and (j < len(df))):
                    if df['index'][j] == sleep_timeflag:
                        df['sleep'][j] = df['sleep'][i]
                        sleep_loop = False
                    elif df['index'][j] < sleep_timeflag:
                        df['sleep'][j] = df['sleep'][i]
                        j = j + 1
                    elif df['index'][j] > sleep_timeflag:
                        sleep_loop = False

        # Check work
        for i in range(0, len(df)):
            if df['temp_work'][i] != -1:
                work_timeflag = df['work_end'][i]
                work_loop = True
                j = i
                while (work_loop and (j < len(df))):
                    if df['index'][j] == work_timeflag:
                        df['work'][j] = df['work'][i]
                        work_loop = False
                    elif df['index'][j] < work_timeflag:
                        df['work'][j] = df['work'][i]
                        j = j + 1
                    elif df['index'][j] > work_timeflag:
                        work_loop = False

        # Check the exercise.
        for i in range(0, len(df)):
            if df['temp_exercise_intensity'][i] != -1:
                exercise_dur = df['exer_dur'][i]
                exercise_start = df['index'][i]
                exercise_loop = True
                j = i
                while (exercise_loop and (j < len(df))):
                    minutes_diff = (df['index'][j] - exercise_start).total_seconds() / 60.0
                    if float(exercise_dur) < minutes_diff:
                        exercise_loop = False
                    else:
                        df['exercise_intensity'][j] = df['exercise_intensity'][i]
                        j = j + 1

        # check if duplicates and also missing timestep rows.
        for i in range(1, len(df)):
            gap = (df['index'][i] - df['index'][i - 1]).total_seconds() / 60.0
            if gap != sample_rate:
                df['missing'][i] = gap



        df.to_csv(os.path.join(args.extract_folder_path, str(patient_id) + '_' + mode + '.csv'))
        # df.to_csv(args.extract_folder_path + str(patient_id) + '_' + mode + '.csv')
        print('Data export completed for subject {}'.format(patient_id))
        count = count + 1


def main():
    args = Options().parse()
    modes_arr = ['train', 'test']
    versions_arr = ['2020', '2018']
    for m in modes_arr:
        for v in versions_arr:
            if v == '2018':
                patient_index = [559, 563, 570, 575, 588, 591]
                train_files = ['train/559-ws-training.xml', 'train/563-ws-training.xml',
                               'train/570-ws-training.xml', 'train/575-ws-training.xml',
                               'train/588-ws-training.xml', 'train/591-ws-training.xml']
                file_train = [os.path.join(args.data_folder_path, v, s) for s in train_files]
                test_files = ['test/559-ws-testing.xml', 'test/563-ws-testing.xml',
                              'test/570-ws-testing.xml', 'test/575-ws-testing.xml',
                              'test/588-ws-testing.xml', 'test/591-ws-testing.xml']
                file_test = [os.path.join(args.data_folder_path, v, s) for s in test_files]
            elif v == '2020':
                patient_index = [540, 544, 552, 567, 584, 596]
                train_files = ['train/540-ws-training.xml', 'train/544-ws-training.xml',
                               'train/552-ws-training.xml', 'train/567-ws-training.xml',
                               'train/584-ws-training.xml', 'train/596-ws-training.xml']
                file_train = [os.path.join(args.data_folder_path, v, s) for s in train_files]
                test_files = ['test/540-ws-testing.xml', 'test/544-ws-testing.xml',
                              'test/552-ws-testing.xml', 'test/567-ws-testing.xml',
                              'test/584-ws-testing.xml', 'test/596-ws-testing.xml']
                file_test = [os.path.join(args.data_folder_path, v, s) for s in test_files]

            if m == 'train':
                files = file_train
            elif m == 'test':
                files = file_test

            extract_data(args, mode=m, files=files, version=v, patient_index=patient_index)


if __name__ == '__main__':
    main()
