# ========================= #
# Training of network model #
# ========================= #

# Usage: python <path-to-this-file>/train.py --config <your_config>.ini
# Default train config file is e3Aij/e3Aij/default_configs/train_default.ini


import argparse
from e3Aij import e3AijKernel

parser = argparse.ArgumentParser(description='Train e3Aij network')
parser.add_argument('--config', type=str, help='Config file for training')
args = parser.parse_args()

kernel = e3AijKernel()
kernel.train(args.config)