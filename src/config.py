
import yaml

# Load the configuration parameters from config.yaml
with open('config.yaml', 'r') as yaml_file:
    global default_config
    default_config = yaml.safe_load(yaml_file)

# DATASET_PATH = 'data/datasets.pkl'
# LEARNING_RATE = 0.001
# BATCH_SIZE = 64
# VAL_SPLIT = 0.2
# NUM_EPOCHS = 100
# NUM_WORKERS = 4
# WINDOW_SIZE = 1000
# OVERLAP = 0.75
# BOTTLENECK_DIM = 256
# DROPOUT = 0.28
