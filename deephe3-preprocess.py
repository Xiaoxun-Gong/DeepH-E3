#!/usr/bin/env python
# ===================== #
# Preprocess of dataset #
# ===================== #

# Usage: python <path-to-this-file>/preprocess.py <your_config>.ini [-n NUM_THREADS]
# Default config for preprocessing data is DeepH-E3/deephe3/default_configs/base_default.ini
# You only have to fill in fields in [data] section

import os
import argparse

parser = argparse.ArgumentParser(description='Preprocess DeepH-E3 data')
parser.add_argument('config', type=str, metavar='CONFIG', help='Config file for preprocessing')
parser.add_argument('-n', type=int, default=None, help='Maximum number of threads')
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
kernel.preprocess(args.config)
