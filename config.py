DATASET_NAME = 'OhioT1DM'
PATIENT_ID_LIST = [559, 563, 570, 575, 588, 591, 540, 544, 552, 567, 584, 596]

class Threshold:
    HYPOGLYCEMIA = 70
    HYPERGLYCEMIA = 180


class Mode:
    HYPER = 'Hyperglycemia'
    HYPO = 'Hypoglycemia'
    TOTAL = 'Total'
    NORMAL = 'Normal'


class GlimmerLSTMConfig:
    ACTIVATION = 'relu'
    OPTIMIZER = 'adam'
    WEIGHTS = [1, 3.296363582, 2.382397706]
    MA_WINDOW_SIZE = 200
    SPLIT_RATIO = 0.2
    TRAIN_WINDOW_SIZE = 72
    N_PREDICTION = 12
    BATCH_SIZE = 48
    EPOCHS = 30
    REPEAT = 3


class GeneticAlgorithmConfig:
    POPULATION_SIZE = 20
    N_GENERATION = 25


class TransformerConfig:
    ACTIVATION = 'relu'
    OPTIMIZER = 'adam'
    NUM_BLOCKS = 1
    NUM_HEADS = 8
    FF_DIM = 256
    DROPOUT = 0.1
    KEY_DIM = 16
    WEIGHTS = [1, 4.0, 2.5]
    MA_WINDOW_SIZE = 200
    SPLIT_RATIO = 0.2
    TRAIN_WINDOW_SIZE = 72
    N_PREDICTION = 12
    BATCH_SIZE = 48
    EPOCHS = 30
    REPEAT = 3