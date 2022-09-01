#!/usr/bin/env python
# config: same with train config, but only needs data section and target section
import argparse

parser = argparse.ArgumentParser(description='Analyze the dataset and give the detailed analysis of prediction target')
parser.add_argument('config', type=str, metavar='CONFIG', help='Config file for training')
parser.add_argument('output', type=str, metavar='OUTPUT', help='Output target.txt')
args = parser.parse_args()

from deephe3 import DeepHE3Kernel
kernel = DeepHE3Kernel()
kernel.load_config(train_config_path=args.config)
config = kernel.train_config

if config.tbt0 == 's':
    print('\nWARNING: Does not need target analysis because target_blocks_type=specify')
else:
    dataset = kernel.get_graph(config)
    kernel.config_set_target(verbose=args.output)
    print('Target analysis complete')
