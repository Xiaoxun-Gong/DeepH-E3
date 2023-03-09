import time
import os
from os.path import dirname, abspath
import sys
import shutil
import warnings
import numpy as np
from collections import namedtuple
import json
import h5py
from tqdm import tqdm

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

# dataset_info_recorder = namedtuple('dataset_info', ['spinful', 'index_to_Z', 'orbital_types'])
# net_out_info_recorder = namedtuple("net_out_info", ['blocks', 'js', 'slices'])
train_info_recorder = namedtuple('train_info', ['epoch', 'global_step', 'lr', 'total_time', 'epoch_time', 'best_loss', 'train_losses', 'val_losses', 'val_loss_list', 'extra_val', 'extra_val_list', ])
train_utils_recorder = namedtuple('train_utils', ['optimizer', 'tb_writer', 'loss_criterion'])


class DatasetInfo:
    
    def __init__(self, spinful, index_to_Z, orbital_types):
        
        self.spinful = spinful
        if isinstance(index_to_Z, list):
            self.index_to_Z = torch.tensor(index_to_Z)
        elif isinstance(index_to_Z, torch.Tensor):
            self.index_to_Z = index_to_Z
        else:
            raise NotImplementedError
        self.orbital_types = orbital_types
        
        self.Z_to_index = torch.full((100,), -1, dtype=torch.int64)
        self.Z_to_index[self.index_to_Z] = torch.arange(len(index_to_Z))

    @classmethod
    def from_dataset(cls, dataset: AijData):
        return cls(dataset.info['spinful'], dataset.info['index_to_Z'], dataset.info['orbital_types'])
    
    @classmethod
    def from_json(cls, src_dir):
        with open(os.path.join(src_dir, 'dataset_info.json'), 'r') as f:
            info = json.load(f)
        return cls(info['spinful'], info['index_to_Z'], info['orbital_types'])
    
    def save_json(self, src_dir):
        with open(os.path.join(src_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dict(spinful=self.spinful, index_to_Z=self.index_to_Z.tolist(), orbital_types=self.orbital_types), f)
    
    def __eq__(self, __o) -> bool:
        if __o.__class__ != __class__:
            raise ValueError
        a = __o.spinful == self.spinful
        b = torch.all(__o.index_to_Z == self.index_to_Z)
        c = __o.orbital_types == self.orbital_types
        return a * b * c


class NetOutInfo:
    
    def __init__(self, target_blocks, dataset_info: DatasetInfo):
        self.target_blocks = target_blocks
        self.dataset_info = dataset_info
        self.blocks, self.js, self.slices = process_targets(dataset_info.orbital_types, 
                                                            dataset_info.index_to_Z, 
                                                            target_blocks)

    def save_json(self, src_dir):
        if os.path.isfile(os.path.join(src_dir, 'dataset_info.json')):
            dataset_info_o = DatasetInfo.from_json(src_dir)
            assert self.dataset_info == dataset_info_o
        else:
            self.dataset_info.save_json(src_dir)
        with open(os.path.join(src_dir, 'target_blocks.json'), 'w') as f:
            json.dump(self.target_blocks, f)
    
    @classmethod
    def from_json(cls, src_dir):
        dataset_info = DatasetInfo.from_json(src_dir)
        with open(os.path.join(src_dir, 'target_blocks.json'), 'r') as f:
            target_blocks = json.load(f)
        return cls(target_blocks, dataset_info)
    
    def merge(self, other):
        assert other.__class__ == __class__
        assert self.dataset_info == other.dataset_info
        self.target_blocks.extend(other.target_blocks)
        self.blocks.extend(other.blocks)
        self.js.extend(other.js)
        length = self.slices.pop()
        for i in other.slices:
            self.slices.append(i + length)

    def __eq__(self, __o) -> bool:
        if __o.__class__ != __class__:
            raise ValueError
        flag = True
        for k in self.__dict__.keys():
            flag *= getattr(self, k) == getattr(__o, k)
        return flag
    
    
class DeepHE3Kernel:
    
    def __init__(self):
        
        # how to determine kernel mode:
        # train mode: self.train_config is not None and self.eval_config is None
        # eval mode: self.eval_config is not None
        self.train_config = None
        self.eval_config = None
        
        self.dataset = None
        self.dataset_info = None
        self.net = None
        self.net_out_info = None
        self.construct_kernel = None
        
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

        print('\n------- DeepH-E3 model training begins -------')
        print(f'Output will be stored under: {config.save_dir}')

        # = random seed =
        print(f'Using random seed: {config.seed}')
        set_random_seed(config.seed)

        # = default dtype =
        print(f'Data type during training: {config.torch_dtype}')
        torch.set_default_dtype(config.torch_dtype)

        # = save DeepH-E3 script =
        self.save_script()
        print('Saved DeepH-E3 source code to output dir')
        
        # = prepare dataset =
        print('\n------- Preparation of training data -------')
        dataset = self.get_graph(config)

        self.config_set_target(verbose=os.path.join(config.save_dir, 'targets.txt'))
        
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

        # = register constructer (e3TensorDecomp) =
        self.register_constructor(device=config.device)
        print('Output constructer associated to net.')

        print()
        print(net)
        # net.analyze_tp(os.path.join(config.save_dir, 'analyze_tp'))
        
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
        scheduler = RevertDecayLR(net, optimizer, config.save_dir, config.revert_decay_patience, config.revert_decay_rate, config.scheduler_type, config.scheduler_params)
        if config.scheduler_type == 1:
            print('Using pytorch scheduler ReduceLROnPlateau')
        elif config.scheduler_type == 2:
            print('Using "slippery slope" scheduler')
        
        # load from checkpoint
        if config.checkpoint_dir:
            print(f'Loading from checkpoint at {config.checkpoint_dir}')
            checkpoint = torch.load(config.checkpoint_dir, map_location='cpu')
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            with open(os.path.join(config.save_dir, 'tensorboard/info.json'), 'r') as f:
                global_step = json.load(f)['global_step'] + 1
            best_loss = checkpoint['val_loss']
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = config.lr
            # scheduler.decay_epoch = config.revert_decay_epoch
            # scheduler.decay_gamma = config.revert_decay_gamma
            print(f'Starting from epoch {checkpoint["epoch"]} with validation loss {checkpoint["val_loss"]}')
        else:
            global_step = 0
            print('Starting new training process')
            best_loss = 1e10
            
        self.train_utils = train_utils_recorder(optimizer, tb_writer, criterion)
            
        print('\n------- Begin training -------')
        # = train and validation =
        begin_time = time.time()
        epoch_begin_time = time.time()
        
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
            print('\nKeyboardInterrupt')

        print('\nTraining finished.')

        print('\n------- Testing network on test set -------')
        checkpoint = torch.load(os.path.join(config.save_dir, 'best_model.pkl'), map_location=config.device)
        self.net.load_state_dict(checkpoint['state_dict'])
        print(f'Using best model at epoch {checkpoint["epoch"]} with val_loss {checkpoint["val_loss"]}')
        print(f'Testing...')
        test_begin = time.time()
        test_h5 = os.path.join(self.train_config.save_dir, 'test_result.h5')
        if os.path.isfile(test_h5):
            print(f'Warning: file already exists and will be replaced: {test_h5}')
            os.remove(test_h5)
        with torch.no_grad():
            test_losses, test_loss_list = self.get_loss(test_loader, save_h5=True)
        print(f'Test finished, cost {time.time() - test_begin:.2f}s.')
        print(f'Test loss: {test_losses.avg:.4e}')
        print('Test results saved to "test_result.h5". It can be analyzed using "deephe3-analyze.py".')
        with open(os.path.join(config.save_dir, 'test_report.txt'), 'w') as f:
            print(f'Test loss: {test_losses.avg:.4e}\n', file=f)
            if len(self.net_out_info.js) > 1:
                self.write_report(test_loss_list, file=f)
        print('Test report written to: "test_report.txt".')
        
    def eval(self, config, debug=False):
        self.load_config(eval_config_path=config)
        eval_config = self.eval_config
        
        torch.set_default_dtype(eval_config.torch_dtype)
        
        print('\n------- Preparation of graph data for evaluation -------')
        dataset = self.get_graph(eval_config, inference=eval_config.inference)
        collate = Collater() # todo: multiple data
        
        H_pred_list = []
        if not eval_config.inference:
            net_out_combined = None

        print('\n------- Finding trained models -------')
        # get models
        model_path_list = self.find_model(eval_config.model_dir)
        
        if not eval_config.inference:
            h5file = os.path.join(eval_config.out_dir, 'test_result.h5')
            if os.path.isfile(h5file):
                print(f'Warning: file already exists and will be replaced: {h5file}')
                os.remove(h5file)
            h5_fp = h5py.File(h5file, 'w')
        
        print('\n------- Evaluating model -------')
        for index_model, model_path in enumerate(model_path_list):
            print(f'\nLoading model {index_model}:')
            self.load_config(train_config_path=os.path.join(model_path, 'src/train.ini'))
            self.config_set_target()
            if not eval_config.inference:
                dataset.set_mask(self.train_config.target_blocks, del_Aij=False, convert_to_net=self.train_config.convert_net_out)
            construct_kernel = self.register_constructor(device=eval_config.device)
            net: Net = self.load_model(os.path.join(model_path, 'src'), device=eval_config.device)
            checkpoint = torch.load(os.path.join(model_path, 'best_model.pkl'), map_location='cpu')
            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
            
            if not eval_config.inference:
                net_out_info = NetOutInfo.from_json(os.path.join(model_path, 'src'))
                if net_out_combined is None:
                    net_out_combined = net_out_info
                else:
                    net_out_combined.merge(net_out_info)
            
            with torch.no_grad():
                iterable = tqdm(enumerate(dataset)) if eval_config.test_only else enumerate(dataset)
                for index_stru, data in iterable:
                    if len(H_pred_list) <= index_stru:
                        H_pred_list.append({})
                    if not eval_config.test_only: print(f'Getting model {index_model} output on structure "{data.stru_id}"...')
                    batch = collate([data])
                    start = time.time()
                    output, output_edge = net(batch.to(device=eval_config.device))
                    H_pred = construct_kernel.get_H(output_edge).cpu()
                    self.update_hopping(H_pred_list[index_stru], H_pred, batch.x, batch.edge_index, batch.edge_key, debug=debug)
                    if not eval_config.test_only: print(f'Finished, cost {time.time() - start:.2f} seconds.')
                    
                    if not eval_config.inference:
                        if not eval_config.test_only: print(f'Saving model {index_model} output on structure "{data.stru_id}" to test_result.h5')
                        self.save_test_result(batch, H_pred, h5_fp)
            
            print(f'Finished evaluating model {index_model} on all structures')
        
        print('\nFinished evaluating all models')
        
        if not debug:
            for hamiltonians in H_pred_list:
                for key_term, hopping in hamiltonians.items():
                    msg = f'Some orbitals are not predicted. You can include option --debug to fill unpredicted matrix elements with 0.'
                    assert torch.all(torch.isnan(hopping)==False), msg
        
        # convert ijji
        if eval_config.only_ij:
            self.convert_ijji_hamiltonians(H_pred_list)
            
        if not eval_config.inference:
            h5_fp.close()
            src = os.path.join(eval_config.out_dir, 'src')
            os.makedirs(src, exist_ok=True)
            net_out_combined.save_json(os.path.join(eval_config.out_dir, 'src'))
            shutil.copyfile(self.train_config.config_file, os.path.join(src, 'train.ini'))
            
        if not eval_config.test_only:
            print('\n------- Output -------')
            for H_dict, data in zip(H_pred_list, dataset):
                os.makedirs(os.path.join(eval_config.out_dir, data.stru_id), exist_ok=True)
                print(f'Writing output to "{data.stru_id}/hamiltonians_pred.h5"')
                with h5py.File(os.path.join(eval_config.out_dir, f'{data.stru_id}/hamiltonians_pred.h5'), 'w') as f:
                    for k, v in H_dict.items():
                        f[k] = v
        
    def get_graph(self, config: BaseConfig, inference=False): # todo: dataset info stored separately
        process_only = config.__class__ == BaseConfig # isinstance(config, BaseConfig)
        # prepare graph data
        if config.graph_dir:
            dataset = AijData.from_existing_graph(config.graph_dir, torch.get_default_dtype())
        else:
            if config.dft_data_dir:
                print('\nPreprocessing data from DFT calculated result...')
                process_data_py = os.path.join(dirname(abspath(__file__)), 'process_data_tools/process_data.py')
                cmd = f'python {process_data_py} --input_dir {config.dft_data_dir} --output_dir {config.processed_data_dir} --simpout' + (' --olp' if config.get_olp else '')
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
                              load_graph=not process_only) 
        
        self.dataset = dataset
        if not process_only:
            # self.process_dataset_info()
            # check target
            if self.train_config is not None:
                assert self.train_config.target == self.dataset.target, 'Train target and dataset target does not match'
            if self.eval_config is not None:
                assert self.eval_config.target == self.dataset.target, 'Eval target and dataset target does not match'
            self.dataset_info = DatasetInfo.from_dataset(dataset)
            
        return dataset
    
    def build_model(self):
        # it is recommended to use save_model first and load model from there, instead of using build_model
        assert self.train_config is not None
        assert self.dataset_info is not None
        config = self.train_config
        
        num_species = len(self.dataset_info.index_to_Z)
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
                  no_parity=config.no_parity,
                  use_sbf=config.use_sbf,
                  only_ij=config.only_ij,
                  spinful=False,
                  if_sort_irreps=False
              )
        print(f'Finished building model, cost {time.time() - begin:.2f} seconds.')
        self.net = net
        
        return net
    
    def save_model(self, src_path):
        assert self.train_config is not None
        assert self.dataset_info is not None
        assert self.net_out_info is not None
        assert os.path.isdir(src_path)
        config = self.train_config
        
        num_species = len(self.dataset_info.index_to_Z)
        with open(os.path.join(src_path, 'build_model.py'), 'w') as f:
            pf = lambda x: print(x, file=f)
            pf(f"from deephe3_1 import Net")
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
            pf(f"    no_parity={config.no_parity},")
            pf(f"    use_sbf={config.use_sbf},")
            pf(f"    only_ij={config.only_ij},")
            pf(f"    if_sort_irreps={False}")
            pf(f")")
        
        self.net_out_info.save_json(src_path)
    
    def load_model(self, src_path, device='cpu'):
        assert os.path.isdir(src_path)
        sys.path.append(src_path)
        
        print('Building model...')
        begin = time.time()
        from build_model import net
        print(f'Finished building model, cost {time.time() - begin:.2f} seconds.')
        
        sys.path.pop()
        sys.modules.pop('build_model')
        
        net.to(device)
        
        self.net = net
        
        # load dataset_info and net_out_info
        net_out_info_o = NetOutInfo.from_json(src_path)
        if self.dataset_info is None:
            self.dataset_info = net_out_info_o.dataset_info
        else:
            assert self.dataset_info == net_out_info_o.dataset_info
        if self.net_out_info is None:
            self.net_out_info = net_out_info_o
        else:
            assert self.net_out_info == net_out_info_o
            
        return net
        
    def register_constructor(self, device='cpu'):
        # requires: dataset_info, train_config which target is already set
        # returns: construct_kernel, net_out_info
        assert self.train_config is not None
        assert self.dataset_info is not None
        
        config = self.train_config
        
        construct_kernel = e3TensorDecomp(config.net_out_irreps, 
                                          self.net_out_info.js, 
                                          default_dtype_torch=torch.get_default_dtype(), 
                                          spinful=self.dataset_info.spinful,
                                          no_parity=config.no_parity, 
                                          if_sort=config.convert_net_out, 
                                          device_torch=device)
        
        self.construct_kernel = construct_kernel
        
        return construct_kernel
    
    def config_set_target(self, verbose=None):
        o, i, s = self.dataset_info.orbital_types, self.dataset_info.index_to_Z, self.dataset_info.spinful# dataset.info['orbital_types'], dataset.info['index_to_Z'], dataset.info['spinful']
        self.train_config.set_target(o, i, s, verbose)
        self.net_out_info = NetOutInfo(self.train_config.target_blocks, self.dataset_info)

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
    
    def get_loss(self, loader, is_train=False, save_h5=False):
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
            
        if save_h5:
            h5_fp = h5py.File(os.path.join(self.train_config.save_dir, 'test_result.h5'), 'w')

        for batch in loader:
            # get predicted H
            output, output_edge = net(batch.to(device=config.device))
            if config.convert_net_out:
                H_pred = output_edge
            else:
                H_pred = construct_kernel.get_H(output_edge)
            # get loss
            loss = criterion(H_pred, batch.label.to(device=config.device), batch.mask)
            
            # backward propagation
            if is_train:
                self.train_utils.optimizer.zero_grad()
                loss.backward()
                # TODO clip_grad_norm_(net.parameters(), 4.2, error_if_nonfinite=True)
                self.train_utils.optimizer.step()
                
            # detailed loss on each target
            with torch.no_grad():
                losses.update(loss.item(), batch.num_edges)
                
                if len(self.net_out_info.js) > 1 and not is_train:
                    for i in range(len(self.net_out_info.js)):
                        target_loss = criterion(H_pred[..., slice(self.net_out_info.slices[i], self.net_out_info.slices[i + 1])], 
                                                batch.label[..., slice(self.net_out_info.slices[i], self.net_out_info.slices[i + 1])].to(device=config.device),
                                                batch.mask[..., slice(self.net_out_info.slices[i], self.net_out_info.slices[i + 1])])
                        if self.dataset_info.spinful:
                            if config.convert_net_out:
                                num_hoppings = batch.mask[:, self.net_out_info.slices[i] * 4].sum().item() # ! this is not correct
                            else:
                                num_hoppings = batch.mask[:, 0, self.net_out_info.slices[i]].sum().item()
                        else:
                            num_hoppings = batch.mask[:, self.net_out_info.slices[i]].sum().item()
                        loss_list[i].update(target_loss.item(), num_hoppings)
                    
            # record output in h5 (for testing)
            if save_h5:
                self.save_test_result(batch, H_pred, h5_fp) # 
                
        if save_h5:
            h5_fp.close()
        
        return losses, loss_list
    
    def save_test_result(self, batch, H_pred, h5_fp):
        assert batch.num_graphs == 1
        stru_id = batch.stru_id[0]
        
        batch = batch.to('cpu')
        if isinstance(H_pred, torch.Tensor):
            H_pred = H_pred.cpu().numpy()
   
        if stru_id in h5_fp:
            g = h5_fp[stru_id]
            for name in ['H_pred', 'label', 'mask']:
                prev = np.array(g[name])
                del g[name]
                if name == 'H_pred':
                    g[name] = np.concatenate((prev, H_pred), axis=-1)
                else:
                    g[name] = np.concatenate((prev, getattr(batch, name)), axis=-1)
        else:
            g = h5_fp.create_group(stru_id)
            g['node_attr'] = batch.x
            g['edge_index'] = batch.edge_index
            g['edge_key'] = batch.edge_key
            g['edge_attr'] = batch.edge_attr
            g['label'] = batch.label
            g['mask'] = batch.mask
            g['H_pred'] = H_pred
            
            stru = g.create_group('structure')
            stru['element'] = self.dataset_info.index_to_Z[batch.x]
            stru['lat'] = batch.lattice[0]
            stru['sites'] = batch.pos
        
        '''test_result.h5 file structure
        +--"/"
        |   +-- group "stru1_id"
        |   |   +-- dataset node_attr
        |   |   +-- dataset edge_index
        |   |   +-- dataset edge_key
        |   |   +-- dataset edge_attr
        |   |   +-- dataset label
        |   |   +-- dataset mask
        |   |   +-- dataset H_pred
        |   |   +-- group "structure"
        |   |   |   +-- dataset element
        |   |   |   +-- dataset lat
        |   |   |   +-- dataset sites
        |   |   |   |
        |   +-- group "stru2_id"
        |   |   +-- ... (similar with above)
        |   |   |
        |   +-- ...
        '''
    
    def update_hopping(self, H_prev, H_pred, node_attr, edge_index, edge_key, debug=False):
        # requires dataset_info, train_config, net_out_info
        # node_attr (element type -- batch.x), edge_index, edge_key come from batch
                
        if isinstance(H_pred, torch.Tensor):
            module = torch
            dtype = self.train_config.torch_dtype
        elif isinstance(H_pred, np.ndarray):
            module = np
            dtype = self.train_config.np_dtype
        else:
            raise ValueError
        
        atom_num_orbitals = [sum(map(lambda x: 2 * x + 1, atom_orbital_types)) for atom_orbital_types in self.dataset_info.orbital_types]
        for index_edge in range(edge_index.shape[1]):
            key_term = str(edge_key[index_edge].tolist())
            i, j = node_attr[edge_index[:, index_edge]]
            if key_term not in H_prev.keys():
                fill = 0 if debug else np.nan
                fill_sp = 0 + 0j if debug else np.nan + np.nan * 1j
                if self.dataset_info.spinful:
                    init = module.full((atom_num_orbitals[i] * 2, atom_num_orbitals[j] * 2), fill_sp, dtype=flt2cplx(dtype))
                else:
                    init = module.full((atom_num_orbitals[i], atom_num_orbitals[j]), fill, dtype=dtype)
                H_prev[key_term] = init
            N_M_str_edge = f'{self.dataset_info.index_to_Z[i].item()} {self.dataset_info.index_to_Z[j].item()}'
            for index_target, equivariant_block in enumerate(self.net_out_info.blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    if N_M_str == N_M_str_edge:
                        slice_row = slice(block_slice[0], block_slice[1])
                        slice_col = slice(block_slice[2], block_slice[3])
                        len_row = block_slice[1] - block_slice[0]
                        len_col = block_slice[3] - block_slice[2]
                        slice_out = slice(self.net_out_info.slices[index_target], self.net_out_info.slices[index_target + 1])
                        if self.dataset_info.spinful:
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
            shutil.copyfile(os.path.join(old_dir, 'model.pkl'), os.path.join(config.save_dir, 'model.pkl'))
            dst = os.path.join(config.save_dir, 'src_restart')
        os.makedirs(dst)
        shutil.copyfile(os.path.join(src, 'deephe3-train.py'), os.path.join(dst, 'train.py'))
        shutil.copyfile(config.config_file, os.path.join(dst, 'train.ini'))
        shutil.copytree(os.path.join(src, 'deephe3'), os.path.join(dst, 'deephe3_1'))
    
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
        if config.train_size > 0:
            train_size = config.train_size
        if config.val_size > 0:
            val_size = config.val_size
        if config.test_size >= 0:
            test_size = config.test_size
        assert train_size + val_size + test_size <= dataset_size

        np.random.shuffle(indices)
        

        print(f'size of train set: {len(indices[:train_size])}')
        train_loader = DataLoader(dataset, 
                                  batch_size=config.batch_size,
                                  shuffle=False, 
                                  sampler=SubsetRandomSampler(indices[:train_size]),
                                  collate_fn=Collater())
        
        val_indices = indices[train_size:train_size + val_size]
        print(f'size of val set: {len(val_indices)}')
        if config.extra_val and not config.extra_val_test_only:
            val_indices.extend(extra_val_indices)
        val_loader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                sampler=SubsetRandomSampler(val_indices),
                                collate_fn=Collater())
        
        extra_val_loader = None
        if config.extra_val:
            print(f'Additionally validating on {len(extra_val_indices)} structure(s)')
            extra_val_loader = DataLoader(dataset,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          sampler=SubsetRandomSampler(extra_val_indices),
                                          collate_fn=Collater())
        
        test_indices = indices[train_size + val_size:train_size + val_size + test_size]
        if config.extra_val:
            test_indices.extend(extra_val_indices)
        print(f'size of test set: {len(test_indices)}')
        test_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 sampler=SubsetRandomSampler(test_indices),
                                 collate_fn=Collater())
        
        print(f'Batch size: {config.batch_size}')
        
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
                    f'Epoch time: {info.epoch_time:6.2f}  | '
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
                file = open(os.path.join(config.save_dir, 'train_report.txt'), 'w')
                print(f'Best model:', file=file)
                print(out_info, end='\n\n', file=file)
                self.write_report(info.val_loss_list, file=file)
                file.close()
            
    def write_report(self, loss_list, file=sys.stdout):
        target_loss_list = [(loss_list[i].avg, i) for i in range(len(loss_list))]
        target_loss_list.sort(key=lambda x: x[0], reverse=True)
        
        print('------- Detailed losses of each target -------', file=file)
        print('Losses are sorted in descending order', file=file)
        for i in range(len(self.net_out_info.js)):
            index_target = target_loss_list[i][1]
            print(f'\n======= No.{i:03}: Target {index_target:03} =======', file=file)
            print('Validation loss:           ', target_loss_list[i][0], file=file)
            print('Angular quantum numbers:   ', self.net_out_info.js[index_target], file=file)
            print('Target blocks:             ', self.train_config.target_blocks[index_target], file=file)
            print('Position in H matrix:      ', self.net_out_info.blocks[index_target], file=file)
    
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