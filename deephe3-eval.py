#!/usr/bin/env python
# ============================================ #
# Getting hamiltonians.h5 using trained models #
# ============================================ #

# Usage: python <path-to-this-file>/eval.py <your_config>.ini [-n NUM_THREADS] [--debug]
# Default config file for evaluation is DeepH-E3/deephe3/default_configs/eval_default.ini

import os
import argparse

parser = argparse.ArgumentParser(description='Evaluate DeepH-E3 model to get hamiltonians_pred.h5')
parser.add_argument('config', type=str, metavar='CONFIG', help='Config file for evaluation')
parser.add_argument('-n', type=int, default=None, help='Maximum number of threads')
parser.add_argument('--debug', action='store_true', help='Fill unpredicted matrix elements with 0 instead of throwing error.')
args = parser.parse_args()

if args.n is not None:
    os.environ["OMP_NUM_THREADS"] = f"{args.n}"
    os.environ["MKL_NUM_THREADS"] = f"{args.n}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{args.n}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{args.n}"
    os.environ["VECLIB_MAXIMUM_THREADS"] = f"{args.n}"
    import torch
    torch.set_num_threads(args.n)

from deephe3 import DeepHE3Kernel
kernel = DeepHE3Kernel()
kernel.eval(args.config, debug=args.debug)
