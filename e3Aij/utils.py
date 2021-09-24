import os
import shutil
import sys
from configparser import ConfigParser

import numpy as np
import scipy
import torch
from torch import nn
import h5py


def print_args(args):
    for k, v in args._get_kwargs():
        print('{} = {}'.format(k, v))
    print('')


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class MaskMSELoss(nn.Module):
    def __init__(self) -> None:
        super(MaskMSELoss, self).__init__()
    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert input.shape == target.shape == mask.shape
        mse = torch.pow(input - target, 2)
        mse = torch.masked_select(mse, mask).mean()

        return mse


class MaskMAELoss(nn.Module):
    def __init__(self) -> None:
        super(MaskMAELoss, self).__init__()
    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert input.shape == target.shape == mask.shape
        mae = torch.abs(input - target)
        mae = torch.masked_select(mae, mask).mean()

        return mae


class LossRecord:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        self.last_val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def save_model(state, path, is_best):
    torch.save(state, os.path.join(path, 'model.pkl'))
    if is_best:
        shutil.copyfile(os.path.join(path, 'model.pkl'), os.path.join(path, 'best_model.pkl'))


def write_ham_h5(hoppings_dict, path):
    fid = h5py.File(path, "w")
    for k, v in hoppings_dict.items():
        fid[k] = v
    fid.close()


def get_config(*args):
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'default.ini'))
    for config_file in args:
        config.read(config_file)
    assert config['basic']['interface'] in ['h5', 'npz', 'h5_rc_only', 'npz_rc_only', 'dftb']
    assert config['network']['aggr'] in ['add', 'mean', 'max']
    assert config['network']['distance_expansion'] in ['GaussianBasis', 'BesselBasis', 'ExpBernsteinBasis']
    assert config['network']['normalization'] in ['BatchNorm', 'LayerNorm', 'PairNorm', 'InstanceNorm', 'GraphNorm', 'DiffGroupNorm']
    assert config['hyperparameter']['optimizer'] in ['sgd', 'sgdm', 'adam', 'adamW', 'adagrad', 'RMSprop', 'lbfgs']
    assert config['hyperparameter']['lr_scheduler'] in ['', 'MultiStepLR', 'ReduceLROnPlateau', 'CyclicLR']

    return config


def get_inference_config(*args):
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'inference_tools', 'inference_default.ini'))
    for config_file in args:
        config.read(config_file)

    return config


def get_process_data_config(*args):
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'process_data_tools', 'process_data_default.ini'))
    for config_file in args:
        config.read(config_file)
    assert config['basic']['target'] in ['hamiltonian', 'density_matrix']

    return config



