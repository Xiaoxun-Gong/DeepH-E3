import os
import shutil
import sys

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
        mse = torch.pow(torch.abs(input - target), 2)
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

def flt2cplx(flt_dtype):
    if flt_dtype == torch.float32:
        cplx_dtype = torch.complex64
    elif flt_dtype == torch.float64:
        cplx_dtype = torch.complex128
    elif flt_dtype == np.float32:
        cplx_dtype = np.complex64
    elif flt_dtype == np.float64:
        cplx_dtype = np.complex128
    else:
        raise NotImplementedError(f'Unsupported float dtype: {flt_dtype}')
    return cplx_dtype


class RevertDecayLR:
    def __init__(self, model, optimizer, save_model_dir, decay_patience=20, decay_rate=0.8, torch_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        
        os.makedirs(save_model_dir, exist_ok=True)
        self.save_model_dir = save_model_dir
        
        self.next_epoch = 0
        self.best_epoch = 0
        self.best_loss = 1e10
        
        # assert len(decay_epoch) == len(decay_gamma)
        # self.decay_epoch = decay_epoch
        # self.decay_gamma = decay_gamma
        # self.decay_step = 0
        self.decay_patience = decay_patience
        self.decay_rate = decay_rate
        
        self.bad_epochs = 0
        
        self.scheduler = None
        if torch_scheduler:
            self.scheduler = ReduceLROnPlateau(optimizer=optimizer, **torch_scheduler)
        
    def load_state_dict(self, state_dict):
        tssd = state_dict.pop('torch_scheduler_state_dict', None)
        if tssd is not None:
            self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer)
            self.scheduler.load_state_dict(tssd)
        
        self.__dict__.update(state_dict)
    
    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() 
                      if key not in ['model', 'optimizer', 'save_model_dir', 'scheduler']}
        if self.scheduler is not None:
            state_dict.update({'torch_scheduler_state_dict': self.scheduler.state_dict()})
        return state_dict
    
    def step(self, val_loss):
        epoch = self.next_epoch
        self.next_epoch += 1
        # = revert to best model =
        # if val_loss > 50 * self.best_loss:
        #     print(f'Validation Loss {val_loss} at epoch {epoch} is larger than 50 times of best validation loss {self.best_loss} at epoch {self.best_epoch}. Reverting to the state there.')
        #     self.revert()
        #     self.decay()
        if val_loss > 2 * self.best_loss:
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0
        if self.bad_epochs >= self.decay_patience:
            print(f'Validation loss has been more than 2 times of best loss for more than {self.decay_patience} epochs.')
            self.revert()
            self.decay()
        
        # = decay learning rate when epoch reaches decay_epoch =
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
        print(f'Reverting to epoch {best_checkpoint["epoch"]} with loss {best_checkpoint["val_loss"]}')
        self.model.load_state_dict(best_checkpoint['state_dict'])
        self.optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
        self.load_state_dict(best_checkpoint['scheduler_state_dict'])
    
    def decay(self):
        # if self.decay_step > len(self.decay_gamma) - 1:
        #     print(f'Learning rate decay step reaches {self.decay_step} which exceeds maximum {len(self.decay_gamma) - 1}. Learning rate is decayed 0.8 times by default.')
        #     gamma = 0.8
        # else:
        #     gamma = self.decay_gamma[self.decay_step]
        num_lr = 0
        for param_group in self.optimizer.param_groups:
            last_lr = param_group['lr']
            param_group['lr'] *= self.decay_rate
            new_lr = param_group['lr'] 
            print(f'Learning rate {num_lr} is decayed from {last_lr} to {new_lr}.')
        # self.decay_step += 1
        
        if self.scheduler is not None:
            self.scheduler.cooldown_counter = self.scheduler.cooldown # start torch_scheduler cooldown
    
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
            

def process_targets(orbital_types, index_to_Z, targets): 
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

def irreps_from_l1l2(l1, l2, mul, spinful):
    r'''
    non-spinful example: l1=1, l2=2 (1x2) ->
    required_irreps_full=1+2+3, required_irreps=1+2+3, required_irreps_x1=None
    
    spinful example: l1=1, l2=2 (1x0.5)x(2x0.5) ->
    required_irreps_full = 1+2+3 + 0+1+2 + 1+2+3 + 2+3+4
    required_irreps = (1+2+3)x0 = 1+2+3
    required_irreps_x1 = (1+2+3)x1 = [0+1+2, 1+2+3, 2+3+4]
    
    notice that required_irreps_x1 is a list of Irreps
    '''
    
    p = (-1) ** (l1 + l2)
    required_ls = range(abs(l1 - l2), l1 + l2 + 1)
    required_irreps = Irreps([(mul, (l, p)) for l in required_ls])
    required_irreps_full = required_irreps
    required_irreps_x1 = None
    if spinful:
        required_irreps_x1 = []
        for _, ir in required_irreps:
            required_ls_irx1 = range(abs(ir.l - 1), ir.l + 1 + 1)
            irx1 = Irreps([(mul, (l, p)) for l in required_ls_irx1])
            required_irreps_x1.append(irx1)
            required_irreps_full += irx1
    return required_irreps_full, required_irreps, required_irreps_x1
    

def orbital_analysis(atom_orbitals, required_block_type, spinful, targets=None, element_pairs=None, verbose=''):
    r'''
    example of atom_orbitals: {'42': [0, 0, 0, 1, 1, 2, 2], '16': [0, 0, 1, 1, 2]}
    
    required_block_type: 's' - specify; 'a' - all; 'o' - off-diagonal; 'd' - diagonal; 
    '''
    
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
                    net_out_irreps += irreps_from_l1l2(l1, l2, 1, spinful)[0]
                else:
                    assert l1 == atom_orbitals[atom1][block_indices[0]] and l2 == atom_orbitals[atom2][block_indices[1]], f'Hamiltonian block angular quantum numbers not all the same in target {target}'
                    
    else:
        hoppings_list = [] # [{'42 16': [4, 3]}, ...]
        for atom1, orbitals1 in atom_orbitals.items():
            for atom2, orbitals2 in atom_orbitals.items():
                hopping_key = atom1 + ' ' + atom2
                if element_pairs:
                    if hopping_key not in element_pairs:
                        continue
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
                irreps_new = irreps_from_l1l2(l1, l2, 1, spinful)[0]
                net_out_irreps_list.append(irreps_new)
                net_out_irreps = net_out_irreps + irreps_new 
        
        for i in hoppings_list_mask:
            assert i
        
    if spinful:
        net_out_irreps = net_out_irreps + net_out_irreps
        
    if verbose and required_block_type != 's':
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
