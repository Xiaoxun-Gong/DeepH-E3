from collections import defaultdict
from math import sqrt
import os
import shutil
import sys
from configparser import ConfigParser
import warnings

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from e3nn.o3 import Irreps


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


class RevertDecayLR:
    def __init__(self, model, optimizer, save_model_dir, decay_epoch=[], decay_gamma=[], torch_scheduler=None):
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
        
        self.scheduler = None
        if torch_scheduler:
            if torch_scheduler == 'ReduceLROnPlateau':
                self.scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=100, verbose=True)
            else:
                raise NotImplementedError(f'scheduler {torch_scheduler} is not supported yet')
        
    def load_state_dict(self, state_dict):
        self.best_epoch = state_dict['best_epoch']
        self.best_loss = state_dict['best_loss']
        self.decay_epoch = state_dict['decay_epoch']
        self.decay_gamma = state_dict['decay_gamma']
        assert len(self.decay_epoch) == len(self.decay_gamma)
        self.decay_step = state_dict['decay_step']
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['torch_scheduler_state_dict'])
    
    def state_dict(self):
        state_dict = {
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
            'decay_epoch': self.decay_epoch,
            'decay_gamma': self.decay_gamma,
            'decay_step': self.decay_step
        }
        if self.scheduler is not None:
            state_dict.update({'torch_scheduler_state_dict': self.scheduler.state_dict()})
        return state_dict
    
    def step(self, epoch, val_loss):
        # = revert to best model =
        # if val_loss > 50 * self.best_loss:
        #     print(f'Validation Loss {val_loss} at epoch {epoch} is larger than 50 times of best validation loss {self.best_loss} at epoch {self.best_epoch}. Reverting to the state there.')
        #     self.revert()
        #     self.decay()
        
        # # = decay learning rate when epoch reaches decay_epoch =
        # if self.decay_step < len(self.decay_epoch):
        #     if epoch >= self.decay_epoch[self.decay_step]:
        #         self.revert()
        #         self.decay()
        
        # = step torch scheduler =
        if self.scheduler is not None:
            self.scheduler.step(val_loss)
        
        # = check is best =
        is_best = val_loss < self.best_loss
        if is_best:
            self.best_loss = val_loss
            self.best_epoch = epoch

        # = save model =
        self.save_model(epoch, val_loss, is_best=is_best)
    
    def revert(self):
        best_checkpoint = torch.load(os.path.join(self.save_model_dir, 'best_model.pkl'))
        print(f'Reverting to epoch {best_checkpoint["epoch"]}')
        self.model.load_state_dict(best_checkpoint['state_dict'])
        self.optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
        self.load_state_dict(best_checkpoint['scheduler_state_dict'])
    
    def decay(self):
        if self.decay_step > len(self.decay_gamma) - 1:
            print(f'Learning rate decay step reaches {self.decay_step} which exceeds maximum {len(self.decay_gamma) - 1}. Learning rate is decayed 0.7 times by default.')
            gamma = 0.8
        else:
            gamma = self.decay_gamma[self.decay_step]
        num_lr = 0
        for param_group in self.optimizer.param_groups:
            last_lr = param_group['lr']
            param_group['lr'] *= gamma
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
            

def process_targets(orbital_types, index_to_Z, targets): # TODO use index_to_Z    
    Z_to_index = torch.full((100,), -1, dtype=torch.int64)
    Z_to_index[index_to_Z] = torch.arange(len(index_to_Z))
    
    orbital_types = list(map(lambda x: np.array(x, dtype=np.int32), orbital_types))
    orbital_types_cumsum = list(map(lambda x: np.concatenate([np.zeros(1, dtype=np.int32), 
                                                                np.cumsum(2 * x + 1)]), orbital_types))

    # = process the orbital indices into block slices =
    equivariant_blocks, out_js_list = [], []
    out_slices = [0]
    for target in targets:
        out_js = None
        equivariant_block = dict()
        for N_M_str, block_indices in target.items():
            i, j = map(lambda x: Z_to_index[int(x)], N_M_str.split())
            block_slice = [
                orbital_types_cumsum[i][block_indices[0]],
                orbital_types_cumsum[i][block_indices[0] + 1],
                orbital_types_cumsum[j][block_indices[1]],
                orbital_types_cumsum[j][block_indices[1] + 1]
            ]
            equivariant_block.update({N_M_str: block_slice})
            if out_js is None:
                out_js = (orbital_types[i][block_indices[0]], orbital_types[j][block_indices[1]])
            else:
                assert out_js == (orbital_types[i][block_indices[0]], orbital_types[j][block_indices[1]])
        equivariant_blocks.append(equivariant_block)
        out_js_list.append(tuple(map(int, out_js)))
        out_slices.append(out_slices[-1] + (2 * out_js[0] + 1) * (2 * out_js[1] + 1))
    
    return equivariant_blocks, out_js_list, out_slices


class assemble_H:
    def __init__(self, orbital_types, index_to_Z, targets):
        self.equivariant_blocks, out_js_list, self.out_slices = process_targets(orbital_types, index_to_Z, targets)
        
        self.index_to_Z = index_to_Z
                
        self.atom_num_orbital = [sum(map(lambda x: 2 * x + 1, atom_orbital_types)) for atom_orbital_types in orbital_types]
        
    def get_H(self, x, edge_idx, edge_fea):
        Hij_list = []
        for index_edge in range(edge_idx.shape[1]):
            Hij_edge = torch.zeros(self.atom_num_orbital[x[edge_idx[0, index_edge]]],
                                self.atom_num_orbital[x[edge_idx[1, index_edge]]],
                                dtype=torch.get_default_dtype(), device='cpu')
            N_M_str_edge = str(self.index_to_Z[x[edge_idx[0, index_edge]]].item()) + ' ' + str(self.index_to_Z[x[edge_idx[1, index_edge]]].item())
            for index_target, target in enumerate(self.equivariant_blocks):
                for N_M_str, block_slice in target.items():
                    if N_M_str == N_M_str_edge:
                        slice_row = slice(block_slice[0], block_slice[1])
                        slice_col = slice(block_slice[2], block_slice[3])
                        len_row = block_slice[1] - block_slice[0]
                        len_col = block_slice[3] - block_slice[2]
                        slice_out = slice(self.out_slices[index_target], self.out_slices[index_target + 1])
                        Hij_edge[slice_row, slice_col] = edge_fea[index_edge, slice_out].reshape(len_row, len_col)
            Hij_list.append(Hij_edge)
            
        return Hij_list


def orbital_analysis(atom_orbitals, required_block_type, targets=None, verbose=''):
    # example of atom_orbitals: {'42': [0, 0, 0, 1, 1, 2, 2], '16': [0, 0, 1, 1, 2]}
    # required_block_type should be 'a' or 'o' or 'd' which means all, off-diagonal, diagonal, respectively.
    assert required_block_type in ['s', 'a', 'o', 'd']
    if required_block_type == 's':
        net_out_irreps = Irreps(None)
        for target in targets:
            l1, l2 = None, None
            for N_M_str, block_indices in target.items():
                atom1, atom2 = N_M_str.split()
                if l1 is None and l2 is None:
                    l1 = atom_orbitals[atom1][block_indices[0]]
                    l2 = atom_orbitals[atom2][block_indices[1]]
                    required_ls = range(abs(l1 - l2), l1 + l2 + 1)
                    required_p = (-1) ** (l1 + l2)
                    net_out_irreps = net_out_irreps + Irreps([(1, (l, required_p)) for l in required_ls])
                else:
                    assert l1 == atom_orbitals[atom1][block_indices[0]] and l2 == atom_orbitals[atom2][block_indices[1]], f'Hamiltonian block angular quantum numbers not all the same in target {target}'
        return targets, net_out_irreps, net_out_irreps.sort()[0].simplify()
    else:
        hoppings_list = [] # [{'42 16': [4, 3]}, ...]
        for atom1, orbitals1 in atom_orbitals.items():
            for atom2, orbitals2 in atom_orbitals.items():
                hopping_key = atom1 + ' ' + atom2
                for orbital1 in range(len(orbitals1)):
                    for orbital2 in range(len(orbitals2)):
                        hopping_orbital = [orbital1, orbital2]
                        hoppings_list.append({hopping_key: hopping_orbital})
                        

        il_list = [] # [[1, 1, 2, 0], ...] this means the hopping is from 1st l=1 orbital to 0th l=2 orbital.
        for hopping in hoppings_list:
            for N_M_str, block in hopping.items():
                atom1, atom2 = N_M_str.split()
                l1 = atom_orbitals[atom1][block[0]]
                l2 = atom_orbitals[atom2][block[1]]
                il1 = block[0] - atom_orbitals[atom1].index(l1)
                il2 = block[1] - atom_orbitals[atom2].index(l2)
            il_list.append([l1, il1, l2, il2])
                
        # print(il_list)

        hoppings_list_mask = [False for _ in range(len(hoppings_list))] # if that hopping is already included, then it is True
        targets = []
        net_out_irreps_list = []
        net_out_irreps = Irreps(None)
        for hopping1_index in range(len(hoppings_list)):
            target = {}
            if not hoppings_list_mask[hopping1_index]:
                is_diagonal = il_list[hopping1_index][0: 2] == il_list[hopping1_index][2: 4]
                hoppings_list_mask[hopping1_index] = True
                if is_diagonal and required_block_type == 'o':
                    continue
                if not is_diagonal and required_block_type == 'd':
                    continue
                target.update(hoppings_list[hopping1_index])
                for hopping2_index in range(len(hoppings_list)):
                    if not hoppings_list_mask[hopping2_index]:
                        if il_list[hopping1_index] == il_list[hopping2_index]: # il1 = il2 means the two hoppings are similar
                            target.update(hoppings_list[hopping2_index])
                            hoppings_list_mask[hopping2_index] = True
                targets.append(target)
                
                l1, l2 = il_list[hopping1_index][0], il_list[hopping1_index][2]
                required_ls = range(abs(l1 - l2), l1 + l2 + 1)
                required_p = (-1) ** (l1 + l2)
                net_out_irreps_list.append(Irreps([(1, (l, required_p)) for l in required_ls]))
                net_out_irreps = net_out_irreps + Irreps([(1, (l, required_p)) for l in required_ls])
        
        for i in hoppings_list_mask:
            assert i
        
        if verbose:
            with open(verbose, 'w') as v:
                print('------- All hoppings -------', file=v)
                for index, (hopping, il) in enumerate(zip(hoppings_list, il_list)):
                    print(index, hopping, il, file=v)
                print('\n------- All targets -------', file=v)
                for index, (target, irreps) in enumerate(zip(targets, net_out_irreps_list)):
                    print(index, target, irreps, file=v)
                print('\n------- Target for net ------', file=v)
                print(targets, file=v)
                print('\n------- Required net out irreps -------', file=v)
                print(net_out_irreps, file=v)
                print('\n------- Simplified net out irreps -------', file=v)
                print(net_out_irreps.sort()[0].simplify(), file=v)
        
        return targets, net_out_irreps, net_out_irreps.sort()[0].simplify()


class TrainConfig:
    def __init__(self, config_file, inference=False):
        assert os.path.isfile(config_file), 'cannot find config file'
        self.config_file = config_file
        self.inference = inference
        config = self.get_config(config_file)
        
        # basic
        self.device = torch.device(config.get('basic', 'torch_device'))
        self.torch_dtype = config.get('basic', 'torch_dtype')
        if self.torch_dtype == 'float':
            self.torch_dtype = torch.float32
        elif self.torch_dtype == 'double':
            self.torch_dtype == torch.float64
        self.seed = config.getint('basic', 'seed')
        self.save_dir = config.get('basic', 'save_dir')
        self.additional_folder_name = config.get('basic', 'additional_folder_name')
        self.checkpoint_dir = config.get('basic', 'checkpoint_dir')
        self.simp_out = config.getboolean('basic', 'simplified_output')
        
        # train
        self.batch_size = config.getint('train', 'batch_size')
        self.num_epoch = config.getint('train', 'num_epoch')
        self.revert_decay_epoch = eval(config.get('train', 'revert_decay_epoch'))
        self.revert_decay_gamma = eval(config.get('train', 'revert_decay_gamma'))
        self.torch_scheduler = config.get('train', 'torch_scheduler')
        
        # hyperparameters
        self.lr = config.getfloat('hyperparameters', 'learning_rate')
        self.train_ratio = config.getfloat('hyperparameters', 'train_ratio')
        self.val_ratio = config.getfloat('hyperparameters', 'val_ratio')
        self.test_ratio = config.getfloat('hyperparameters', 'test_ratio')
        
        if self.checkpoint_dir:
            config1_path = os.path.join(os.path.dirname(self.checkpoint_dir), 'src/train.ini')
            assert os.path.isfile(config1_path), 'cannot find train.ini under checkpoint dir'
            print('Overwriting config using train.ini from checkpoint')
            verbose = True
        else:
            config1_path = config_file
            verbose = False
        self.overwrite(config1_path, verbose=verbose)
        
    def get_config(self, *args, verbose=True):
        config = ConfigParser()
        config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_configs/train_default.ini'))
        for config_file in args:
            if verbose:
                print('Loading train config from: ', config_file)
            config.read(config_file)
        assert config['basic']['torch_dtype'] in ['float', 'double'], f"{config['basic']['torch_dtype']}"
        assert config['target']['target'] in ['hamiltonian']
        assert config['target']['target_blocks_type'][0] in ['s', 'a', 'o', 'd']

        return config
    
    def set_target(self, orbital_types, index_to_Z, output_file):
        atom_orbitals = {}
        for Z, orbital_type in zip(index_to_Z, orbital_types):
            atom_orbitals[str(Z.item())] = orbital_type
            
        target_blocks, net_out_irreps, irreps_post_edge = orbital_analysis(atom_orbitals, self.tbt0, targets=self.target_blocks, verbose=output_file)
        
        if not self.target_blocks:
            self.target_blocks = target_blocks
        if not self.net_out_irreps:
            self.net_out_irreps = net_out_irreps
        if not self.irreps_post_edge:
            self.irreps_post_edge = irreps_post_edge
    
    def overwrite(self, config_file, verbose=True):
        config = self.get_config(config_file, verbose=verbose)
        
        # data
        self.processed_data_dir = config.get('data', 'processed_data_dir')
        if self.processed_data_dir:
            self.graph_dir = config.get('data', 'save_graph_dir')
            self.dataset_name = config.get('data', 'dataset_name')
            self.cutoff_radius = config.getfloat('data', 'cutoff_radius')
            self.only_ij = config.getboolean('data', 'use_undirected_graph')
            self.target = None
        else:
            graph = config.get('data', 'graph_dir')
            if not self.inference:
                assert os.path.isfile(graph), 'Required graph does not exist'
            self.graph_dir = os.path.dirname(graph)
            graph = os.path.basename(graph)
            assert graph[-4:] == '.pkl', 'graph filename extension should be .pkl'
            options = graph.rstrip('.pkl').split('-')
            if options[0][0] == 'H':
                self.target = 'hamiltonian'
            elif options[0][0:2] == 'DM':
                self.target = 'density_matrix'
            else:
                raise ValueError(f'Cannot identify graph file {graph}')
            self.dataset_name = options[1]
            self.cutoff_radius = float(options[2].split('r')[0])
            self.only_ij = options[-1] == 'undrct'
            # todo: max_num_nbr, edge_Aij
        
        # target
        target1 = config.get('target', 'target')
        if self.target is not None:
            assert self.target == target1, 'target and graph data does not match'
        else:
            self.target = target1
        self.tbt0 = config.get('target', 'target_blocks_type')[0]
        self.target_blocks = None
        if self.tbt0 == 's':
            self.target_blocks = eval(config.get('target', 'target_blocks'))
        
        # network
        sh_lmax = config.getint('network', 'spherical_harmonics_lmax')
        self.irreps_sh = Irreps([(1, (i, (-1) ** i)) for i in range(sh_lmax + 1)])
        irreps_mid = config.get('network', 'irreps_mid')
        if irreps_mid:
            self.irreps_mid_node = irreps_mid
            self.irreps_post_node = irreps_mid
            self.irreps_mid_edge = irreps_mid
        irreps_embed = config.get('network', 'irreps_embed')
        if irreps_embed:
            self.irreps_embed_node = irreps_embed
            self.irreps_edge_init = irreps_embed
        if self.target == 'hamiltonian':
            self.irreps_out_node = '1x0e'
        self.num_blocks = config.getint('network', 'num_blocks')
        # ! post edge          
        ien = config.get('network', 'irreps_embed_node')
        if ien:
            self.irreps_embed_node = ien
        iei = config.get('network', 'irreps_edge_init')
        if iei:
            self.irreps_edge_init = iei
        imn = config.get('network', 'irreps_mid_node')
        if imn:
            self.irreps_mid_node = imn
        ipn = config.get('network', 'irreps_post_node')
        if ipn:
            self.irreps_post_node = ipn
        ion = config.get('network', 'irreps_out_node')
        if ion:
            self.irreps_out_node = ion
        ime = config.get('network', 'irreps_mid_edge')
        if ime:
            self.irreps_mid_edge = ime
        
        self.net_out_irreps = config.get('network', 'out_irreps')
        self.irreps_post_edge = config.get('network', 'irreps_post_edge')
        
        # network irreps are not fully set yet. 
        # remember to call set_target after graph data is available.
            
class EvalConfig:
    def __init__(self, config_file):
        self.config_file = config_file
        config = self.get_config(config_file)
        
        # basic
        self.model_dir = config.get('basic', 'trained_model_dir')
        self.device = torch.device(config.get('basic', 'device'))
        self.dtype = config.get('basic', 'dtype')
        if self.dtype == 'float':
            self.torch_dtype = torch.float32
            self.np_dtype = np.float32
        elif self.dtype == 'double':
            self.torch_dtype == torch.float64
            self.np_dtype = np.float64
        self.out_dir = config.get('basic', 'output_dir')
        os.makedirs(self.out_dir, exist_ok=True)
        
        # data (copied from TrainConfig)
        self.processed_data_dir = config.get('data', 'processed_data_dir')
        if self.processed_data_dir:
            self.graph_dir = config.get('data', 'save_graph_dir')
            self.dataset_name = config.get('data', 'dataset_name')
            self.cutoff_radius = config.getfloat('data', 'cutoff_radius')
            self.only_ij = config.getboolean('data', 'use_undirected_graph')
            self.target = None
        else:
            graph = config.get('data', 'graph_dir')
            assert os.path.isfile(graph), 'Required graph does not exist'
            self.graph_dir = os.path.dirname(graph)
            graph = os.path.basename(graph)
            assert graph[-4:] == '.pkl', 'graph filename extension should be .pkl'
            options = graph.rstrip('.pkl').split('-')
            if options[0][0] == 'H':
                self.target = 'hamiltonian'
            elif options[0][0:2] == 'DM':
                self.target = 'density_matrix'
            else:
                raise ValueError(f'Cannot identify graph file {graph}')
            self.dataset_name = options[1]
            self.cutoff_radius = float(options[2].split('r')[0])
            self.only_ij = options[-1] == 'undrct'
            # todo: max_num_nbr, edge_Aij
            
        target1 = config.get('basic', 'target')
        if self.target is not None:
            assert self.target == target1, 'target and graph data does not match'
        else:
            self.target = target1

    def get_config(self, *args):
        config = ConfigParser()
        config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_configs/eval_default.ini'))
        for config_file in args:
            print('Loading eval config from: ', config_file)
            config.read(config_file)
        assert config['basic']['dtype'] in ['float', 'double'], f"{config['basic']['dtype']}"
        assert config['basic']['target'] in ['hamiltonian']
        
        return config