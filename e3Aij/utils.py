import os
import shutil
import sys
from configparser import ConfigParser
import warnings

import numpy as np
import torch
from torch import nn
import h5py
from e3nn.o3 import Irrep, Irreps, wigner_3j
from e3nn.nn import Extract


def print_args(args):
    for k, v in args._get_kwargs():
        print('{} = {}'.format(k, v))
    print('')


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=-1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


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


class RevertDecayLR:
    def __init__(self, model, optimizer, save_model_dir, decay_epoch=[], decay_gamma=[]):
        self.model = model
        self.optimizer = optimizer
        
        os.makedirs(save_model_dir, exist_ok=True)
        self.save_model_dir = save_model_dir
        
        self.best_epoch = 0
        self.best_loss = 1e10
        
        assert len(decay_epoch) == len(decay_gamma)
        self.decay_epoch = decay_epoch
        self.decay_gamma = decay_gamma
        self.decay_step = 0
        
    def load_state_dict(self, state_dict):
        self.best_epoch = state_dict['best_epoch']
        self.best_loss = state_dict['best_loss']
        self.decay_epoch = state_dict['decay_epoch']
        self.decay_gamma = state_dict['decay_gamma']
        assert len(self.decay_epoch) == len(self.decay_gamma)
        self.decay_step = state_dict['decay_step']
    
    def state_dict(self):
        state_dict = {
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
            'decay_epoch': self.decay_epoch,
            'decay_gamma': self.decay_gamma,
            'decay_step': self.decay_step
        }
        return state_dict
    
    def step(self, epoch, val_loss):
        # = revert to best model =
        if val_loss > 30 * self.best_loss:            
            print(f'Validation Loss {val_loss} at epoch {epoch} is larger than 30 times of best validation loss {self.best_loss} at epoch {self.best_epoch}. Reverting to the state there.')
            
            best_checkpoint = torch.load(os.path.join(self.save_model_dir, 'best_model.pkl'))
            # best_model should contain: state_dict, optimizer_state_dict, scheduler_state_dict

            self.model.load_state_dict(best_checkpoint['state_dict'])
            self.optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
            self.load_state_dict(best_checkpoint['scheduler_state_dict'])
            
            self.decay()
        
        # = decay learning rate when epoch reaches decay_epoch =
        if self.decay_step < len(self.decay_epoch):
            if epoch >= self.decay_epoch[self.decay_step]:
                self.decay()
        
        # = check is best =
        is_best = val_loss < self.best_loss
        if is_best:
            self.best_loss = val_loss
            self.best_epoch = epoch

        # = save model =
        self.save_model(epoch, val_loss, is_best=is_best)
    
    def decay(self):
        if self.decay_step > len(self.decay_gamma) - 1:
            warnings.warn(f'Learning rate decay step reaches {self.decay_step} which exceeds maximum {len(self.decay_gamma) - 1}. Learning rate is unchanged.')
        else:
            num_lr = 0
            for param_group in self.optimizer.param_groups:
                last_lr = param_group['lr']
                param_group['lr'] *= self.decay_gamma[self.decay_step]
                new_lr = param_group['lr'] 
                print(f'Learning rate {num_lr} is decayed from {last_lr} to {new_lr}.')
        self.decay_step += 1
    
    def save_model(self, epoch, val_loss, is_best=False, **kwargs):
        state = {
            'epoch': epoch,
            'val_loss': val_loss,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.state_dict(),
        }
        state.update(kwargs)
        torch.save(state, os.path.join(self.save_model_dir, 'model.pkl'))
        if is_best:
            shutil.copyfile(os.path.join(self.save_model_dir, 'model.pkl'), 
                            os.path.join(self.save_model_dir, 'best_model.pkl'))
            

class sort_irreps(torch.nn.Module):
    def __init__(self, irreps_in):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        sorted_irreps = irreps_in.sort()
        irreps_out_list = [((mul, ir),) for mul, ir in sorted_irreps.irreps]
        instructions = [(i,) for i in sorted_irreps.inv]
        self.extr = Extract(irreps_in, irreps_out_list, instructions)
        
        self.irreps_in = irreps_in
        self.irreps_out = sorted_irreps.irreps
    
    def forward(self, x):
        extracted = self.extr(x)
        return torch.cat(extracted, dim=-1)


class Rotate:
    def __init__(self, default_dtype_torch, device_torch='cpu'):
        sqrt_2 = 1.4142135623730951
        # openmx的实球谐函数基组变复球谐函数
        self.Us_openmx = {
            0: torch.tensor([1], dtype=torch.cfloat, device=device_torch),
            1: torch.tensor([[-1 / sqrt_2, 1j / sqrt_2, 0], [0, 0, 1], [1 / sqrt_2, 1j / sqrt_2, 0]], dtype=torch.cfloat, device=device_torch),
            2: torch.tensor([[0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0]], dtype=torch.cfloat, device=device_torch),
            3: torch.tensor([[0, 0, 0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [0, 0, 0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, -1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2, 0, 0],
                             [0, 0, 0, 0, 0, 1 / sqrt_2, 1j / sqrt_2]], dtype=torch.cfloat, device=device_torch),
        }
        # openmx的实球谐函数基组变wiki的实球谐函数 https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        self.Us_openmx2wiki = {
            0: torch.eye(1, dtype=default_dtype_torch).to(device=device_torch),
            1: torch.eye(3, dtype=default_dtype_torch)[[1, 2, 0]].to(device=device_torch),
            2: torch.eye(5, dtype=default_dtype_torch)[[2, 4, 0, 3, 1]].to(device=device_torch),
            3: torch.eye(7, dtype=default_dtype_torch)[[6, 4, 2, 0, 1, 3, 5]].to(device=device_torch)
        }
        self.Us_wiki2openmx = {k: v.T for k, v in self.Us_openmx2wiki.items()}

    def rotate_e3nn_v(self, v, R, l, order_xyz=True):
        if order_xyz:
            # R是(x, y, z)顺序
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn是(y, z, x)顺序
        else:
            # R是(y, z, x)顺序
            R_e3nn = R
        return v @ Irrep(l, 1).D_from_matrix(R_e3nn)

    def rotate_openmx_H(self, H, R, l_left, l_right, order_xyz=True):
        if order_xyz:
            # R是(x, y, z)顺序
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn是(y, z, x)顺序
        else:
            # R是(y, z, x)顺序
            R_e3nn = R
        return self.Us_openmx2wiki[l_left].T @ Irrep(l_left, 1).D_from_matrix(R_e3nn).transpose(-1, -2) @ self.Us_openmx2wiki[l_left] @ H \
               @ self.Us_openmx2wiki[l_right].T @ Irrep(l_right, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_right]

    def wiki2openmx_H(self, H, l_left, l_right):
        return self.Us_openmx2wiki[l_left].T @ H @ self.Us_openmx2wiki[l_right]

    def openmx2wiki_H(self, H, l_left, l_right):
        return self.Us_openmx2wiki[l_left] @ H @ self.Us_openmx2wiki[l_right].T

    def rotate_matrix_convert(self, R):
        # (x, y, z)顺序排列的旋转矩阵转换为(y, z, x)顺序(see e3nn.o3.spherical_harmonics() and https://docs.e3nn.org/en/stable/guide/change_of_basis.html)
        return torch.eye(3)[[1, 2, 0]] @ R @ torch.eye(3)[[1, 2, 0]].T # todo: cuda

class construct_H:
    def __init__(self, net_irreps_out, l1, l2):
        net_irreps_out = Irreps(net_irreps_out)
        
        self.l1, self.l2 = l1, l2
        
        # = check angular momentum quantum number =
        if l1 == l2:
            assert len(net_irreps_out) == 1, 'It is recommended to combine the irreps together if the two output angular momentum quantum number are the same'
            assert net_irreps_out[0].mul % 2 == 0
            self.mul = net_irreps_out[0].mul // 2
        elif l1 * l2 == 0:
            assert len(net_irreps_out) == 1, 'Only need one irrep if one angular momentum is 0'
            assert net_irreps_out[0].mul == 1, 'Only need multiplicity one if one angular momentum is 0'
            self.mul = 1
            if l1 == 0:
                assert net_irreps_out[0].ir.l == l2
            elif l2 == 0:
                assert net_irreps_out[0].ir.l == l1
        else:
            assert net_irreps_out[0].ir.l == l1
            assert net_irreps_out[1].ir.l == l2
            assert net_irreps_out[0].mul == net_irreps_out[1].mul
            self.mul = net_irreps_out[0].mul
            # assert self.mul != 1, 'Too few multiplicities'
               
        # = check parity
        for mul_ir in net_irreps_out:
            assert mul_ir.ir.p == (- 1) ** mul_ir.ir.l
            
        self.rotate_kernel = Rotate(torch.get_default_dtype())

    def get_H(self, net_out):
        r''' get openmx type H from net output '''
        if self.l1 == 0:
            H_pred = net_out.unsqueeze(-2)
        elif self.l2 == 0:
            H_pred = net_out.unsqueeze(-1)
        else:
            vec1 = net_out[:, :(self.mul * (2 * self.l1 + 1))].reshape(-1, self.mul, 2 * self.l1 + 1)
            vec2 = net_out[:, (self.mul * (2 * self.l1 + 1)):].reshape(-1, self.mul, 2 * self.l2 + 1)
            H_pred = torch.sum(vec1[:, :, :, None] * vec2[:, :, None, :], dim=-3)

        H_pred = self.rotate_kernel.wiki2openmx_H(H_pred, self.l1, self.l2)
        
        return H_pred

class e3TensorDecomp:
    def __init__(self, net_irreps_out, H_l1, H_l2, default_dtype_torch, device_torch='cpu'):
        
        self.H_l1, self.H_l2 = H_l1, H_l2
        p = (- 1) ** (H_l1 + H_l2) # required parity
        required_ls = range(abs(H_l1 - H_l2), H_l1 + H_l2 + 1)
        
        # = check net irreps out =
        net_irreps_out = Irreps(net_irreps_out)
        mul = net_irreps_out[0].mul
        required_irreps_out = Irreps([(mul, (l, p)) for l in required_ls])
        assert net_irreps_out == required_irreps_out, f'requires {required_irreps_out} but got {net_irreps_out}'
        
        # = get CG coefficients multiplier to act on net_out =
        wigner_multiplier = []
        for l in required_ls:
            for i in range(mul):
                wigner_multiplier.append(wigner_3j(H_l1, H_l2, l, dtype=default_dtype_torch, device=device_torch))
        self.wigner_multiplier = torch.cat(wigner_multiplier, dim=-1)
        
        # = register rotate kernel =
        self.rotate_kernel = Rotate(default_dtype_torch, device_torch)
    
    def get_H(self, net_out):
        r''' get openmx type H from net output '''
        H = torch.einsum('ijk,uk->uij', self.wigner_multiplier, net_out)
        H = self.rotate_kernel.wiki2openmx_H(H, self.H_l1, self.H_l2)
        return H