import time
import os
from os.path import dirname, abspath
import sys
import shutil
import numpy as np
from collections import namedtuple
import json
import h5py

import torch
from torch import optim
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import AijData
from .graph import Collater, convert_ijji
from .model import Net
from .parse_configs import BaseConfig, TrainConfig, EvalConfig
from .utils import Logger, RevertDecayLR, MaskMSELoss, LossRecord, process_targets, set_random_seed, flt2cplx
from .e3modules import e3TensorDecomp

class e3AijKernel:
    
    def __init__(self):
        
        self.train_config = None
        self.eval_config = None
        
        self.dataset = None
        self.net = None
        self.net_out_info = None
        self.construct_kernel = None
        
        self.spinful = None
        
        self.train_utils = None
        self.train_info = None
        
    def load_config(self, train_config_path=None, eval_config_path=None):
        if train_config_path is not None:
            self.train_config = TrainConfig(train_config_path)
        if eval_config_path is not None:
            self.eval_config = EvalConfig(eval_config_path)
            
        if self.train_config is not None and self.eval_config is not None:
            assert self.train_config.torch_dtype == self.eval_config.torch_dtype, f'model uses dtype {self.train_config.torch_dtype} but evaluation requires dtype {self.eval_config.torch_dtype}'
            assert self.train_config.target == self.eval_config.target, f'model predicts {self.train_config.target} but evaluation requires prediction of {self.eval_config.target}'
            # if train_config.cutoff_radius != eval_config.cutoff_radius:
            #     warnings.warn(f'Model has cutoff radius r={train_config.cutoff_radius} but evaluation requires r={eval_config.cutoff_radius}')
            # assert self.train_config.only_ij == self.eval_config.only_ij, f'evaluation uses {"un" if eval_config.only_ij else ""}directed graph but model does not'
    
    def preprocess(self, preprocess_config):
        print('\n------- Preprocessing data -------')
        config = BaseConfig(preprocess_config)
        self.get_graph(config)
    
    def train(self, train_config):
        self.load_config(train_config_path=train_config)
        
        config = self.train_config

        # = record output =
        os.makedirs(config.save_dir)
        sys.stdout = Logger(os.path.join(config.save_dir, "result.txt"))
        sys.stderr = Logger(os.path.join(config.save_dir, "stderr.txt"))

        print('\n------- e3Aij model training begins -------')
        print(f'Output will be stored under: {config.save_dir}')

        # = random seed =
        print(f'Using random seed: {config.seed}')
        set_random_seed(config.seed)

        # = default dtype =
        print(f'Data type during training: {config.torch_dtype}')
        torch.set_default_dtype(config.torch_dtype)

        # = save e3Aij script =
        self.save_script()
        print('Saved e3Aij source code to output dir')
        
        # = prepare dataset =
        print('\n------- Preparation of training data -------')
        dataset = self.get_graph(config)
        
        # = register constructer (e3TensorDecomp) =
        self.register_constructor()
        
        # set dataset mask
        dataset.set_mask(config.target_blocks, convert_to_net=config.convert_net_out)
        
        # = data loader =
        print('\n------- Data loader for training -------')
        train_loader, val_loader, extra_val_loader, test_loader = self.get_loader()
        
        # = Build net =
        print('\n------- Build model -------')
        if config.checkpoint_dir:
            print('Building model from checkpoint')
            net : Net = self.load_model(os.path.join(dirname(config.checkpoint_dir), 'src'), device=config.device)
        else:
            self.save_model(os.path.join(config.save_dir, 'src'))
            net : Net = self.load_model(os.path.join(config.save_dir, 'src'), device=config.device)
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("The model you built has %d parameters." % params)
        print(net)
        
        print('\n------- Preparation for training -------')
        # = select optimizer =
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = optim.Adam(model_parameters, lr=config.lr, betas=config.adam_betas)
        # model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        # optimizer_sgd = optim.SGD(model_parameters, lr=config.lr)
        print(f'Using optimizer Adam with initial lr={config.lr}, betas={config.adam_betas}')
        criterion = MaskMSELoss()
        print('Loss type: MSE over all matrix elements')
        # = tensorboard =
        tb_writer = SummaryWriter(os.path.join(config.save_dir, 'tensorboard'))
        print('Tensorboard recorder initialized')
        # = LR scheduler =
        scheduler = RevertDecayLR(net, optimizer, config.save_dir, config.revert_decay_patience, config.revert_decay_rate, config.torch_scheduler)
        if config.torch_scheduler:
            print('Using pytorch scheduler ReduceLROnPlateau')
        
        # load from checkpoint
        if config.checkpoint_dir:
            print(f'Loading from checkpoint at {config.checkpoint_dir}')
            checkpoint = torch.load(config.checkpoint_dir)
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            with open(os.path.join(config.save_dir, 'tensorboard/info.json'), 'r') as f:
                global_step = json.load(f)['global_step'] + 1
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = config.lr
            # scheduler.decay_epoch = config.revert_decay_epoch
            # scheduler.decay_gamma = config.revert_decay_gamma
            print(f'Starting from epoch {checkpoint["epoch"]} with validation loss {checkpoint["val_loss"]}')
        else:
            global_step = 0
            print('Starting new training process')
            
        train_utils_recorder = namedtuple('train_utils', ['optimizer', 'tb_writer', 'loss_criterion'])
        self.train_utils = train_utils_recorder(optimizer, tb_writer, criterion)
        train_info_recorder = namedtuple('train_info', ['epoch', 'global_step', 'lr', 'total_time', 'epoch_time', 'best_loss', 'train_losses', 'val_losses', 'val_loss_list', 'extra_val', 'extra_val_list', ])
            
        print('\n------- Begin training -------')
        # = train and validation =
        begin_time = time.time()
        epoch_begin_time = time.time()
        best_loss = 1e10
        
        epoch = scheduler.next_epoch
        learning_rate = optimizer.param_groups[0]['lr']
        try:
            while epoch < config.num_epoch and learning_rate > config.min_lr:
                
                # = train =
                train_losses, _ = self.get_loss(train_loader, is_train=True)
                
                with torch.no_grad():
                    # = val =
                    val_losses, val_loss_list = self.get_loss(val_loader)
                    
                    # = extra_val =
                    extra_val_losses, extra_loss_record_list = None, None
                    if config.extra_val:
                        extra_val_losses, extra_loss_record_list = self.get_loss(extra_val_loader)
                
                # = RECORD LOSS =
                learning_rate = optimizer.param_groups[0]['lr']
                self.train_info = train_info_recorder(epoch=epoch, 
                                                      global_step=global_step,
                                                      lr=learning_rate,
                                                      total_time=time.time()-begin_time,
                                                      epoch_time=time.time()-epoch_begin_time,
                                                      best_loss=best_loss,
                                                      train_losses=train_losses,
                                                      val_losses=val_losses,
                                                      val_loss_list=val_loss_list,
                                                      extra_val=extra_val_losses,
                                                      extra_val_list=extra_loss_record_list)
                self.record_train()
                
                if val_losses.avg < best_loss:
                    best_loss = val_losses.avg
                
                # = save model, revert, etc. =
                scheduler.step(val_losses.avg)
                
                epoch_begin_time = time.time()
                epoch = scheduler.next_epoch
                global_step += 1
                
        except KeyboardInterrupt:
            print('\nTraining stopped due to KeyboardInterrupt')
            
        # test_losses, test_loss_list = self.get_loss(test_loader)
        
    def eval(self, config, debug=False):
        self.load_config(eval_config_path=config)
        eval_config = self.eval_config
        
        torch.set_default_dtype(eval_config.torch_dtype)
        
        print('\n------- Preparation of graph data for evaluation -------')
        dataset = self.get_graph(eval_config, inference=True)
        collate = Collater() # todo: multiple data
        
        H_pred_list = []

        print('\n------- Finding trained models -------')
        # get models
        model_path_list = self.find_model(eval_config.model_dir)
        
        print('\n------- Evaluating model -------')
        for index_model, model_path in enumerate(model_path_list):
            print(f'\nLoading model {index_model}:')
            self.load_config(train_config_path=os.path.join(model_path, 'src/train.ini'))
            construct_kernel = self.register_constructor()
            net: Net = self.load_model(os.path.join(model_path, 'src'), device=eval_config.device)
            checkpoint = torch.load(os.path.join(model_path, 'best_model.pkl'))
            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
            
            with torch.no_grad():
                for index_stru, data in enumerate(dataset):
                    if len(H_pred_list) <= index_stru:
                        H_pred_list.append({})
                    print(f'Getting model {index_model} output on structure "{data.stru_id}"...')
                    batch = collate([data])
                    output, output_edge = net(batch.to(device=eval_config.device))
                    H_pred = construct_kernel.get_H(output_edge).detach().cpu().numpy()
                    self.update_hopping(H_pred_list[index_stru], batch, H_pred, debug=debug)
            
            print(f'Finished evaluating model {index_model} on all structures')
        
        print('\nFinished evaluating all models')
        
        if not debug:
            for hamiltonians in H_pred_list:
                for key_term, hopping in hamiltonians.items():
                    assert np.all(np.isnan(hopping)==False), f'Some orbitals are not predicted. You can include option --debug to fill unpredicted matrix elements with 0.'
        
        # convert ijji
        if eval_config.only_ij:
            self.convert_ijji_hamiltonians(H_pred_list)
            
        print('\n------- Output -------')
        for H_dict, data in zip(H_pred_list, dataset):
            os.makedirs(os.path.join(eval_config.out_dir, data.stru_id), exist_ok=True)
            print(f'Writing output to "{data.stru_id}/hamiltonians_pred.h5"')
            with h5py.File(os.path.join(eval_config.out_dir, f'{data.stru_id}/hamiltonians_pred.h5'), 'w') as f:
                for k, v in H_dict.items():
                    f[k] = v
        
    def get_graph(self, config: BaseConfig, inference=False):
        # prepare graph data
        if config.graph_dir:
            dataset = AijData.from_existing_graph(config.graph_dir, torch.get_default_dtype())
        else:
            if config.dft_data_dir:
                print('\nPreprocessing data from DFT calculated result...')
                process_data_py = os.path.join(dirname(dirname(abspath(__file__))), 'process_data_tools/process_data.py')
                cmd = f'python {process_data_py} --input_dir {config.dft_data_dir} --output_dir {config.processed_data_dir} --simpout' + (' --olp' if inference else '')
                return_code = os.system(cmd)
                assert return_code == 0, f'Error occured in executing command: "{cmd}"'
            print('\nProcessing graph data...')
            dataset = AijData(raw_data_dir=config.processed_data_dir,
                              graph_dir=config.save_graph_dir,
                              target=config.target_data,
                              dataset_name=config.dataset_name,
                              multiprocessing=False,
                              radius=-1,
                              max_num_nbr=0,
                              edge_Aij=True,
                              inference=inference,
                              only_ij=False,
                              default_dtype_torch=torch.get_default_dtype(),
                              load_graph=config.__class__!=BaseConfig)
            
        if self.train_config is not None:
            assert self.train_config.target == dataset.target, 'Train target and dataset target does not match'
        if self.eval_config is not None:
            assert self.eval_config.target == dataset.target, 'Eval target and dataset target does not match'
        
        self.dataset = dataset
        if config.__class__ != BaseConfig:
            self.spinful = dataset.info['spinful']
            
        return dataset
    
    def build_model(self):
        assert self.train_config is not None
        assert self.dataset is not None
        config = self.train_config
        
        num_species = len(self.dataset.info["index_to_Z"])
        print('Building model...')
        begin = time.time()
        net = Net(
                  num_species=num_species,
                  irreps_embed_node=config.irreps_embed_node,
                  irreps_edge_init=config.irreps_edge_init,
                  irreps_sh=config.irreps_sh,
                  irreps_mid_node=config.irreps_mid_node,
                  irreps_post_node=config.irreps_post_node,
                  irreps_out_node=config.irreps_out_node,
                  irreps_mid_edge=config.irreps_mid_edge,
                  irreps_post_edge=config.irreps_post_edge,
                  irreps_out_edge=config.net_out_irreps,
                  num_block=config.num_blocks,
                  r_max=config.cutoff_radius,
                  use_sc=True,
                  only_ij=config.only_ij,
                  spinful=False,
                  if_sort_irreps=False
              )
        print(f'Finished building model, cost {time.time() - begin:.2f} seconds.')
        self.net = net
        
        return net
    
    def save_model(self, src_path):
        assert self.train_config is not None
        assert self.dataset is not None
        assert os.path.isdir(src_path)
        config = self.train_config
        
        num_species = len(self.dataset.info["index_to_Z"])
        with open(os.path.join(src_path, 'build_model.py'), 'w') as f:
            pf = lambda x: print(x, file=f)
            pf(f"from e3Aij_1 import Net")
            pf(f"net = Net(")
            pf(f"    num_species={num_species},")
            pf(f"    irreps_embed_node='{config.irreps_embed_node}',")
            pf(f"    irreps_edge_init='{config.irreps_edge_init}',")
            pf(f"    irreps_sh='{config.irreps_sh}',")
            pf(f"    irreps_mid_node='{config.irreps_mid_node}',")
            pf(f"    irreps_post_node='{config.irreps_post_node}',")
            pf(f"    irreps_out_node='{config.irreps_out_node}',")
            pf(f"    irreps_mid_edge='{config.irreps_mid_edge}',")
            pf(f"    irreps_post_edge='{config.irreps_post_edge}',")
            pf(f"    irreps_out_edge='{config.net_out_irreps}',")
            pf(f"    num_block={config.num_blocks},")
            pf(f"    r_max={config.cutoff_radius},")
            pf(f"    use_sc={True},")
            pf(f"    only_ij={config.only_ij},")
            pf(f"    spinful={False},")
            pf(f"    if_sort_irreps={False}")
            pf(f")")
    
    def load_model(self, src_path, device='cpu'):
        assert os.path.isdir(src_path)
        sys.path.append(src_path)
        
        print('Building model...')
        begin = time.time()
        from build_model import net
        print(f'Finished building model, cost {time.time() - begin:.2f} seconds.')
        net.to(device)
        
        self.net = net
        
        return net
        
    def register_constructor(self):
        assert self.train_config is not None
        assert self.dataset is not None
        
        dataset = self.dataset
        config = self.train_config
        
        o, i, s = dataset.info['orbital_types'], dataset.info['index_to_Z'], dataset.info['spinful']
        verbose = ''
        if self.eval_config is None:
            verbose = os.path.join(config.save_dir, 'targets.txt')
        self.train_config.set_target(o, i, s, verbose)
        Ret = namedtuple("net_out_info", ['blocks', 'js', 'slices'])
        self.net_out_info = Ret(*process_targets(o, i, self.train_config.target_blocks))
        print('\nTargets in train config successfully processed.')
        
        construct_kernel = e3TensorDecomp(config.net_out_irreps, 
                                          self.net_out_info.js, 
                                          default_dtype_torch=torch.get_default_dtype(), 
                                          spinful=dataset.info['spinful'], 
                                          if_sort=config.convert_net_out, 
                                          device_torch=config.device)
        print('Output constructer associated to net.')
        
        self.construct_kernel = construct_kernel
        
        return construct_kernel
    
    @staticmethod
    def find_model(dir):
        model_path_list = []
        for root, dirs, files in os.walk(dir):
            if 'best_model.pkl' in files and 'src' in dirs:
                model_path_list.append(os.path.abspath(root))
        assert len(model_path_list) > 0, 'Cannot find any model'
        print(f'Successfully found {len(model_path_list)} model(s):')
        for index_model, model_path in enumerate(model_path_list):
            print(f'model {index_model}:', model_path)
        return model_path_list
    
    def get_loss(self, loader, is_train=False):
        # with torch.no_grad():
        #     net.eval()
        assert self.net is not None
        net = self.net
        assert self.train_config is not None
        config = self.train_config
        assert self.net_out_info is not None
        assert self.construct_kernel is not None
        construct_kernel = self.construct_kernel
        criterion = self.train_utils.loss_criterion
        
        if is_train:
            net.train()
        else:
            net.eval()
        
        losses = LossRecord()
        loss_list = None
        if len(self.net_out_info.js) > 1 and not is_train:
            loss_list = [LossRecord() for _ in range(len(self.net_out_info.js))]

        for batch in loader:
            # get predicted H
            output, output_edge = net(batch.to(device=config.device))
            if config.convert_net_out:
                H_pred = output_edge
            else:
                H_pred = construct_kernel.get_H(output_edge)
            # get loss
            loss = criterion(H_pred, batch.label.to(device=config.device), batch.mask)
            
            if is_train:
                self.train_utils.optimizer.zero_grad()
                loss.backward()
                # TODO clip_grad_norm_(net.parameters(), 4.2, error_if_nonfinite=True)
                self.train_utils.optimizer.step()
                
            with torch.no_grad():
                losses.update(loss.item(), batch.num_edges)
                
                if len(self.net_out_info.js) > 1 and not is_train:
                    for i in range(len(self.net_out_info.js)):
                        target_loss = criterion(H_pred[..., slice(self.net_out_info.slices[i], self.net_out_info.slices[i + 1])], 
                                                batch.label[..., slice(self.net_out_info.slices[i], self.net_out_info.slices[i + 1])].to(device=config.device),
                                                batch.mask[..., slice(self.net_out_info.slices[i], self.net_out_info.slices[i + 1])])
                        if self.spinful:
                            if config.convert_net_out:
                                num_hoppings = batch.mask[:, self.net_out_info.slices[i] * 4].sum() # ! this is not correct
                            else:
                                num_hoppings = batch.mask[:, 0, self.net_out_info.slices[i]].sum()
                        else:
                            num_hoppings = batch.mask[:, self.net_out_info.slices[i]].sum()
                        loss_list[i].update(target_loss.item(), num_hoppings)
                    
        return losses, loss_list
    
    def update_hopping(self, H_prev, batch, H_pred, debug=False):
        np_dtype = self.train_config.np_dtype
        atom_num_orbitals = [sum(map(lambda x: 2 * x + 1, atom_orbital_types)) for atom_orbital_types in self.dataset.info['orbital_types']]
        hamiltonians = {}
        for index_edge in range(batch.edge_index.shape[1]):
            key_term = str(batch.edge_key[index_edge].tolist())
            i, j = batch.x[batch.edge_index[:, index_edge]]
            if key_term not in H_prev.keys():
                fill = 0 if debug else np.nan
                fill_sp = 0 + 0j if debug else np.nan + np.nan * 1j
                if self.spinful:
                    init = np.full((atom_num_orbitals[i] * 2, atom_num_orbitals[j] * 2), fill_sp, dtype=flt2cplx(np_dtype))
                else:
                    init = np.full((atom_num_orbitals[i], atom_num_orbitals[j]), fill, dtype=np_dtype)
                H_prev[key_term] = init
            N_M_str_edge = f'{self.dataset.info["index_to_Z"][i].item()} {self.dataset.info["index_to_Z"][j].item()}'
            for index_target, equivariant_block in enumerate(self.net_out_info.blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    if N_M_str == N_M_str_edge:
                        slice_row = slice(block_slice[0], block_slice[1])
                        slice_col = slice(block_slice[2], block_slice[3])
                        len_row = block_slice[1] - block_slice[0]
                        len_col = block_slice[3] - block_slice[2]
                        slice_out = slice(self.net_out_info.slices[index_target], self.net_out_info.slices[index_target + 1])
                        if self.spinful:
                            slice_row_ds = slice(atom_num_orbitals[i] + block_slice[0], atom_num_orbitals[i] + block_slice[1])
                            slice_col_ds = slice(atom_num_orbitals[j] + block_slice[2], atom_num_orbitals[j] + block_slice[3])
                            H_prev[key_term][slice_row, slice_col] = H_pred[index_edge, 0, slice_out].reshape(len_row, len_col)
                            H_prev[key_term][slice_row, slice_col_ds] = H_pred[index_edge, 1, slice_out].reshape(len_row, len_col)
                            H_prev[key_term][slice_row_ds, slice_col] = H_pred[index_edge, 2, slice_out].reshape(len_row, len_col)
                            H_prev[key_term][slice_row_ds, slice_col_ds] = H_pred[index_edge, 3, slice_out].reshape(len_row, len_col)
                        else:
                            H_prev[key_term][slice_row, slice_col] = H_pred[index_edge, slice_out].reshape(len_row, len_col)
        return H_prev
                            
    def save_script(self):
        # current source code will be stored under src
        # if checkpoint is specified, then current source code will be stored under src_restart, 
        # and source code in checkpoint will be copied to src
        assert self.train_config is not None
        config = self.train_config
        src = dirname(dirname(abspath(__file__)))
        dst = os.path.join(config.save_dir, 'src')
        if config.checkpoint_dir:
            old_dir = os.path.dirname(config.checkpoint_dir)
            shutil.copytree(os.path.join(old_dir, 'src'), os.path.join(config.save_dir, 'src'))
            shutil.copytree(os.path.join(old_dir, 'tensorboard'), os.path.join(config.save_dir, 'tensorboard'))
            shutil.copyfile(os.path.join(old_dir, 'best_model.pkl'), os.path.join(config.save_dir, 'best_model.pkl'))
            dst = os.path.join(config.save_dir, 'src_restart')
        os.makedirs(dst)
        shutil.copyfile(os.path.join(src, 'train.py'), os.path.join(dst, 'train.py'))
        shutil.copyfile(config.config_file, os.path.join(dst, 'train.ini'))
        shutil.copytree(os.path.join(src, 'e3Aij'), os.path.join(dst, 'e3Aij_1'))
    
    def get_loader(self):
        assert self.train_config is not None
        config = self.train_config
        dataset = self.dataset
        
        indices = list(range(len(dataset)))

        if config.extra_val:
            extra_val_indices = []
            for extra_val_id in config.extra_val:
                ind = dataset.data.stru_id.index(extra_val_id)
                extra_val_indices.append(ind)
                indices.remove(ind)
            
        dataset_size = len(indices)
        train_size = int(config.train_ratio * dataset_size)
        val_size = int(config.val_ratio * dataset_size)
        test_size = int(config.test_ratio * dataset_size)
        assert train_size + val_size + test_size <= dataset_size
        
        dataset_size = len(indices)
        train_size = int(config.train_ratio * dataset_size)
        val_size = int(config.val_ratio * dataset_size)
        test_size = int(config.test_ratio * dataset_size)
        assert train_size + val_size + test_size <= dataset_size

        np.random.shuffle(indices)
        print(f'size of train set: {len(indices[:train_size])}')
        print(f'size of val set: {len(indices[train_size:train_size + val_size])}')
        print(f'size of test set: {len(indices[train_size + val_size:train_size + val_size + test_size])}')
        print(f'Batch size: {config.batch_size}')
        if config.extra_val:
            print(f'Additionally validating on {len(extra_val_indices)} structure(s)')

        train_loader = DataLoader(dataset, 
                                 batch_size=config.batch_size,
                                 shuffle=False, 
                                 sampler=SubsetRandomSampler(indices[:train_size]),
                                 collate_fn=Collater())
        val_loader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                sampler=SubsetRandomSampler(indices[train_size:train_size + val_size]),
                                collate_fn=Collater())
        extra_val_loader = None
        if config.extra_val:
            extra_val_loader = DataLoader(dataset,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          sampler=SubsetRandomSampler(extra_val_indices),
                                          collate_fn=Collater())
        test_loader = DataLoader(dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 sampler=SubsetRandomSampler(indices[train_size + val_size:train_size + val_size + test_size]),
                                 collate_fn=Collater())
        
        return train_loader, val_loader, extra_val_loader, test_loader
    
    def record_train(self):
        info = self.train_info
        config = self.train_config
        
        # = PRINT LOSS =
        time_r = round(info.total_time)
        d, h, m, s = time_r//86400, time_r%86400//3600, time_r%3600//60, time_r%60
        out_info = (f'Epoch #{info.epoch:<5d}  | '
                    f'Time: {d:02d}d {h:02d}h {m:02d}m  | '
                    f'LR: {info.lr:.2e}  | '
                    f'Batch time: {info.epoch_time:6.2f}  | '
                    f'Train loss: {info.train_losses.avg:.2e}  | ' # :11.8f
                    f'Val loss: {info.val_losses.avg:.2e}'
                    )
        if config.extra_val:
            out_info += f'  | Extra val: {info.extra_val.avg:0.2e}'
        if len(self.net_out_info.js) > 1:
            out_info = '====================\n' + out_info + '\n'
            loss_list = [info.val_loss_list[i].avg for i in range(len(self.net_out_info.js))]
            max_loss = max(loss_list)
            min_loss = min(loss_list)
            out_info += f'Target {loss_list.index(max_loss):03} has maximum loss {max_loss:.2e}; '
            out_info += f'Target {loss_list.index(min_loss):03} has minimum loss {min_loss:.2e}'
            if not config.simp_out:
                out_info += '\n'
                i = 0
                while i < len(self.net_out_info.js):
                    out_info += f'Target {i:03}: {info.val_loss_list[i].avg:.2e}'
                    if i % 5 == 4:
                        out_info += '\n'
                    else:
                        out_info += ' \t|'
                    i += 1
        print(out_info)
        
        # = TENSORBOARD =
        tb_writer = self.train_utils.tb_writer
        tb_writer.add_scalar('Learning rate', info.lr, global_step=info.global_step)
        tb_writer.add_scalars('Loss', {'Train loss': info.train_losses.avg}, global_step=info.global_step)
        tb_writer.add_scalars('Loss', {'Validation loss': info.val_losses.avg}, global_step=info.global_step)
        if config.extra_val:
            tb_writer.add_scalars('Loss', {'Extra Validation': info.extra_val.avg}, global_step=info.global_step)
        if len(self.net_out_info.js) > 1:
            tb_writer.add_scalars('Loss', {'Max loss': max_loss}, global_step=info.global_step)
            tb_writer.add_scalars('Loss', {'Min loss': min_loss}, global_step=info.global_step)
            tb_writer.add_scalars('Target losses', {'Validation loss': info.val_losses.avg}, global_step=info.global_step)
            for i in range(len(self.net_out_info.js)):
                tb_writer.add_scalars('Target losses', {f'Target {i} loss': info.val_loss_list[i].avg}, global_step=info.global_step)
        with open(os.path.join(config.save_dir, 'tensorboard/info.json'), 'w') as f:
            json.dump({'global_step': info.global_step}, f)
            
        # = write report =
        if info.val_losses.avg < info.best_loss:
            if len(self.net_out_info.js) > 1:
                target_loss_list = [(info.val_loss_list[i].avg, i) for i in range(len(info.val_loss_list))]
                target_loss_list.sort(key=lambda x: x[0], reverse=True)
                report = open(os.path.join(config.save_dir, 'report.txt'), 'w')
                print(f'Best model:', file=report)
                print(out_info, file=report)
                print('\n------- Detailed losses of each target -------', file=report)
                print('Losses are sorted in descending order', file=report)
                for i in range(len(self.net_out_info.js)):
                    index_target = target_loss_list[i][1]
                    print(f'\n======= No.{i:03}: Target {index_target:03} =======', file=report)
                    print('Validation loss:           ', target_loss_list[i][0], file=report)
                    print('Angular quantum numbers:   ', self.net_out_info.js[index_target], file=report)
                    print('Target blocks:             ', config.target_blocks[index_target], file=report)
                    print('Position in H matrix:      ', self.net_out_info.blocks[index_target], file=report)
                report.close()
    
    @staticmethod
    def convert_ijji_hamiltonians(H_pred_list):
        for index_stru in range(len(H_pred_list)):
            H_dict_inv = {}
            for key_term, hopping in H_pred_list[index_stru].items():
                # if not args.debug:
                #     assert np.all(np.isnan(hopping)==False), f'Some orbitals are not predicted'
                key_inv = str(convert_ijji(key_term))
                if key_inv == key_term:
                    H_pred_list[index_stru][key_term] = (hopping + hopping.T) / 2.0
                else:
                    assert key_inv not in H_pred_list[index_stru].keys()
                    H_dict_inv[key_inv] = hopping.T
            H_pred_list[index_stru].update(H_dict_inv)