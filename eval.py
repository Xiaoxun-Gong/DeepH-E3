# ============================================ #
# Getting hamiltonians.h5 using trained models #
# ============================================ #

# Usage: python <path-to-this-file>/eval.py --config <your_config>.ini
# Default config file for evaluation is e3Aij/e3Aij/default_configs/eval_default.ini


import argparse
from e3Aij import e3AijKernel

parser = argparse.ArgumentParser(description='Evaluate e3Aij model to get hamiltonians_pred.h5')
parser.add_argument('--config', type=str, help='Config file for evaluation')
args = parser.parse_args()

kernel = e3AijKernel()
kernel.eval(args.config)