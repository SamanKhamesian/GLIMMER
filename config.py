DATASET_NAME = 'OhioT1DM'
# PATIENT_ID_LIST = [559, 563, 570, 575, 588, 591, 540, 544, 552, 567, 584, 596]
PATIENT_ID_LIST = [559, 563]

class Threshold:
    HYPOGLYCEMIA = 70
    HYPERGLYCEMIA = 180


class Mode:
    HYPER = 'Hyperglycemia'
    HYPO = 'Hypoglycemia'
    TOTAL = 'Total'
    NORMAL = 'Normal'


class CustomLSTMConfig:
    FOLDER_PATH = './models/glimmer/'
    ACTIVATION = 'relu'
    OPTIMIZER = 'adam'
    WEIGHTS = [1, 3.296363582, 2.382397706]
    MA_WINDOW_SIZE = 200
    SPLIT_RATIO = 0.2
    TRAIN_WINDOW_SIZE = 72
    N_PREDICTION = 12
    BATCH_SIZE = 48
    EPOCHS = 4
    REPEAT = 1


class GeneticAlgorithmConfig:
    FOLDER_PATH = './models/genetic_optimization/'
    POPULATION_SIZE = 6
    N_GENERATION = 2
