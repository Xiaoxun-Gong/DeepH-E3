import os
import shutil
import sys
import random

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from e3nn.o3 import Irreps, Irrep

from .from_nequip.tp_utils import tp_path_exists


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


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


class SlipSlopLR:
    def __init__(self, optimizer, start=1400, interval=200, decay_rate=0.5) -> None:
        self.optimizer = optimizer
        
        self.start = start
        self.interval = interval
        self.decay_rate = decay_rate
        
        self.next_epoch = 0
        self.last_decayed = None
        
    def step(self, val_loss=None):
        epoch = self.next_epoch
        self.next_epoch += 1
        
        next_decay = -1
        if self.last_decayed is not None:
            next_decay = self.last_decayed + self.interval
        
        if epoch == self.start or (epoch == next_decay):
            self.decay()
            self.last_decayed = epoch
        
    def decay(self):
        num_lr = 0
        for param_group in self.optimizer.param_groups:
            last_lr = param_group['lr']
            param_group['lr'] *= self.decay_rate
            new_lr = param_group['lr'] 
            print(f'Learning rate {num_lr} is decayed from {last_lr} to {new_lr}.')
    
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class RevertDecayLR:
    def __init__(self, model, optimizer, save_model_dir, decay_patience=20, decay_rate=0.8, scheduler_type=0, scheduler_params=None):
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
        self.alpha = None
        self.loss_smoothed = None
        
        self.scheduler_type = scheduler_type
        if scheduler_type == 1:
            self.alpha = scheduler_params.pop('alpha', 0.1)
            self.scheduler = ReduceLROnPlateau(optimizer=optimizer, **scheduler_params)
        elif scheduler_type == 2:
            self.scheduler = SlipSlopLR(optimizer=optimizer, **scheduler_params)
        elif scheduler_type == 0:
            pass
        else:
            raise ValueError(f'Unknown scheduler type: {scheduler_type}')
        
    def load_state_dict(self, state_dict):
        tssd = state_dict.pop('torch_scheduler_state_dict', None)
        if tssd is not None:
            self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer)
            self.scheduler.load_state_dict(tssd)
        sssd = state_dict.pop('slipslop_state_dict', None)
        if sssd is not None:
            self.scheduler = SlipSlopLR(optimizer=self.optimizer)
            self.scheduler.load_state_dict(sssd)
        
        self.__dict__.update(state_dict)
    
    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() 
                      if key not in ['model', 'optimizer', 'save_model_dir', 'scheduler']}
        if self.scheduler_type == 1:
            state_dict.update({'torch_scheduler_state_dict': self.scheduler.state_dict()})
        elif self.scheduler_type == 2:
            state_dict.update({'slipslop_state_dict': self.scheduler.state_dict()})
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
        if self.scheduler_type == 1:
            # exponential smoothing of val_loss
            # loss_smoothed(t) = alpha*loss(t)+(1-alpha)*loss_smoothed(t-1)
            # alpha=0.1 by default
            if self.loss_smoothed is None:
                self.loss_smoothed = val_loss
            else:
                self.loss_smoothed = self.alpha * val_loss + (1.0 - self.alpha) * self.loss_smoothed
            self.scheduler.step(self.loss_smoothed)
        elif self.scheduler_type == 2:
            self.scheduler.step()
        
        # = check is best =
        is_best = val_loss < self.best_loss
        if is_best:
            self.best_loss = val_loss
            self.best_epoch = epoch

        # = save model =
        save_complete = False
        while not save_complete:
            try:
                self.save_model(epoch, val_loss, is_best=is_best)
                save_complete = True
            except KeyboardInterrupt:
                print('Interrupting while saving model might cause the saved model to be deprecated')

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
        
        if self.scheduler_type == 1:
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


def irreps_from_l1l2(l1, l2, mul, spinful, no_parity=False):
    r'''
    non-spinful example: l1=1, l2=2 (1x2) ->
    required_irreps_full=1+2+3, required_irreps=1+2+3, required_irreps_x1=None
    
    spinful example: l1=1, l2=2 (1x0.5)x(2x0.5) ->
    required_irreps_full = 1+2+3 + 0+1+2 + 1+2+3 + 2+3+4
    required_irreps = (1+2+3)x0 = 1+2+3
    required_irreps_x1 = (1+2+3)x1 = [0+1+2, 1+2+3, 2+3+4]
    
    notice that required_irreps_x1 is a list of Irreps
    '''
    p = 1
    if not no_parity:
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
    

def orbital_analysis(atom_orbitals, required_block_type, spinful, targets=None, element_pairs=None, no_parity=False, verbose=''):
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
                    net_out_irreps += irreps_from_l1l2(l1, l2, 1, spinful, no_parity=no_parity)[0]
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
                irreps_new = irreps_from_l1l2(l1, l2, 1, spinful, no_parity=no_parity)[0]
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
        print(f'\nAutomatically generated target and net_out_irreps. \nDetails saved to: {verbose}')
    
    return targets, net_out_irreps, net_out_irreps.sort()[0].simplify()

def find_required_irs(irreps_in1, irreps_in2, irreps_out, if_verbose=False):
    # find the irs needed in irreps_in1 in order for in1 x in2 -> irreps_out
    # Returns a list of list. At least one irrep in each sublist is needed. 
    # All the sublists returned are sorted in reversed order.
    
    irreps_in1 = Irreps(irreps_in1)
    irreps_in2 = Irreps(irreps_in2).sort().irreps.simplify()
    irreps_out = Irreps(irreps_out).sort().irreps.simplify()
    
    required_irs = []
    missing_ir_out = []
    for _, ir_out in irreps_out:
        if not tp_path_exists(irreps_in1, irreps_in2, ir_out):
            missing_ir_out.append(ir_out)
            req_irs_path = [] # having only one of the irrep in req_irs_path is ok
            for _, ir_in2 in irreps_in2:
                p = ir_out.p * ir_in2.p
                for l in range(abs(ir_out.l - ir_in2.l), ir_out.l + ir_in2.l + 1):
                    req_ir_path = Irrep(l, p)
                    if req_ir_path not in req_irs_path:
                        req_irs_path.append(req_ir_path)
            assert req_irs_path, 'all possible irrep eliminated'
            req_irs_path = sorted(req_irs_path, reverse=True)
            required_irs.append(req_irs_path)
            
    if if_verbose and len(missing_ir_out) > 0:
        print(f'Required ir {missing_ir_out} in irreps_out cannot be generated by the last edge update.')
    
    return required_irs

def refine_post_node(irreps_post_node, irreps_mid_node, irreps_mid_edge, irreps_sh, irreps_post_edge, if_verbose=False):
    '''In spinful case, parity of irreps in irreps_post_edge will no longer be exactly (-1)^l. However, we usually prefer to set the parity of irreps in irreps_mid_node and irreps_mid_edge to (-1)^l, so the output irreps might not always be able to be fully produced by the convolution in the last edge update. To tackle this problem, the most efficient method is to add some irreps of different parity to irreps_post_node.
    
    This function generates the minimal possible irreps_post_node according to other irreps used in the net.'''

    # (irreps_mid_node + irreps_mid_edge) x irreps_sh -> irreps_post_node
    # (irreps_post_node + irreps_mid_edge) x irreps_sh -> irreps_post_edge
    # irreps_post_node is irreps_mid_node by default
    
    irreps_post_node = Irreps(irreps_post_node)
    irreps_mid_node = Irreps(irreps_mid_node)
    irreps_mid_edge = Irreps(irreps_mid_edge)
    irreps_sh = Irreps(irreps_sh)
    irreps_post_edge = Irreps(irreps_post_edge)
    
    required_irs = find_required_irs(irreps_post_node + irreps_mid_edge, irreps_sh, irreps_post_edge, if_verbose=if_verbose)
    
    irs_add = []
    irs_excluded = []
    
    complete = False
    while not complete:
        ix = len(irs_add) - 1
        while ix >= 0:
            if irs_add[ix] in irs_excluded:
                irs_add.pop()
            ix -= 1
        for req_irs_path in required_irs:
            ir = req_irs_path[-1]
            if ir in irs_excluded:
                req_irs_path.pop() # popping the last element is efficient, but maybe this does not make much difference...
                ir = req_irs_path[-1]
            if ir not in irs_add:
                irs_add.append(ir)
        complete = True
        for ir_add in irs_add:
            if not tp_path_exists(irreps_mid_node + irreps_mid_edge, irreps_sh, ir_add):
                irs_excluded.append(ir_add)
                complete = False
                
    irreps_add = Irreps(None)
    for ir_add in irs_add:
        found = False
        for ix in range(len(irreps_post_node)):
            if ir_add.l == irreps_post_node[ix].ir.l:
                # example: need 4o, has 4x4e, then add 4x4o
                found = True
                mul = irreps_post_node[ix].mul # max(irreps_post_node[ix].mul // 2, 1) # prevent it to be 0
                irreps_add += Irreps([(mul, ir_add)])
        if not found:
            # take multiplicity to be half that of the mul with highest ir.l
            mul = irreps_post_node[-1].mul
            irreps_add += Irreps([(mul, ir_add)])
    
    if if_verbose and len(irreps_add) > 0:
        print(f'Automatically adding irreps {irreps_add} to irreps_post_node to avoid this problem.')
        
    return irreps_post_node + irreps_add