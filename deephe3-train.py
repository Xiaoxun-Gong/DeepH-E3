#!/usr/bin/env python
# ========================= #
# Training of network model #
# ========================= #

# Usage: python <path-to-this-file>/train.py <your_config>.ini [-n NUM_THREADS]
# Default train config file is DeepH-E3/deephe3/default_configs/train_default.ini


import os
import argparse

parser = argparse.ArgumentParser(description='Train DeepH-E3 network')
parser.add_argument('config', type=str, metavar='CONFIG', help='Config file for training')
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
kernel.train(args.config)
