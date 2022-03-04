# ===================== #
# Preprocess of dataset #
# ===================== #

# Usage: python <path-to-this-file>/preprocess.py --config <your_config>.ini
# Default config for preprocessing data is e3Aij/e3Aij/default_configs/base_default.ini
# You only have to fill in fields in [data] section


import argparse
from e3Aij import e3AijKernel

parser = argparse.ArgumentParser(description='Preprocess e3Aij data')
parser.add_argument('--config', type=str, help='Config file for preprocessing')
args = parser.parse_args()

kernel = e3AijKernel()
kernel.preprocess(args.config)