
import os

LABEL_TYPE = 'mixed'

# base data dir
DATA_DIR = os.path.join(os.getcwd(), 'exps-data/data')

# source train, dev and test docs
TRAINING_DOCS = os.path.join(DATA_DIR, 'docs-training')
TESTING_DOCS = os.path.join(DATA_DIR, 'nts-test/docs')

# Gold Labels for respective train, dev and test sets
ANNS_TRAIN_DEV = os.path.join(DATA_DIR, 'anns_train_dev.txt')
ANNS_TEST = os.path.join(DATA_DIR, 'anns_test.txt')

# ids for train, dev and test sets
IDS_TRAINING = os.path.join(DATA_DIR, 'ids_training.txt')
IDS_DEVELOPMENT = os.path.join(DATA_DIR, 'ids_development.txt')
IDS_TESTING = os.path.join(DATA_DIR, 'ids_testing.txt')

# create a directory to hold processed file if not existing
if not (os.path.exists(f"{DATA_DIR}/result/raw") and
        os.path.exists(f"{DATA_DIR}/result/mlb")):
    os.makedirs(f"{DATA_DIR}/result/raw")
    os.makedirs(f"{DATA_DIR}/result/mlb")

# Raw train, dev and test files
RAW_TRAIN_FILE = os.path.join(
    DATA_DIR,
    'result/raw',
    f"train_data_raw_{LABEL_TYPE}.pkl"
)
RAW_DEV_FILE = os.path.join(
    DATA_DIR,
    'result/raw',
    f"dev_data_raw_{LABEL_TYPE}.pkl"
)
RAW_TEST_FILE = os.path.join(
    DATA_DIR,
    'result/raw',
    f"test_data_raw_{LABEL_TYPE}.pkl"
)

# MLB train, dev and test files
MLB_FILE = os.path.join(
    DATA_DIR,
    'result/mlb',
    f'mlb_{LABEL_TYPE}.pkl'
)
DISCARDED_FILE = os.path.join(
    DATA_DIR,
    'result/mlb',
    f'discarded_{LABEL_TYPE}.pkl'
)
MLB_TRAIN_FILE = os.path.join(
    DATA_DIR,
    'result/mlb',
    f"train_data_{LABEL_TYPE}.pkl"
)
MLB_DEV_FILE = os.path.join(
    DATA_DIR,
    'result/mlb',
    f"dev_data_{LABEL_TYPE}.pkl"
)
MLB_TEST_FILE = os.path.join(
    DATA_DIR,
    'result/mlb',
    f"test_data_{LABEL_TYPE}.pkl"
)