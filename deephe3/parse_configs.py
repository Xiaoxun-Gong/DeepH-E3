import os
import time
from configparser import ConfigParser
import numpy as np

import torch

from e3nn.o3 import Irreps

from .utils import orbital_analysis, refine_post_node

class BaseConfig:
    def __init__(self, config_file=None):
        
        self._config = ConfigParser(inline_comment_prefixes=(';',))
        
        if self.__class__ is __class__:
            base_default = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_configs/base_default.ini')
            print(f'Getting config file from: {config_file}')
            self.get_config(config_file, config_file_default=base_default)
            
            self.get_basic_section()
            self.get_data_section()
    
    def get_config(self, config_file, config_file_default=''):
        if config_file_default:
            assert os.path.isfile(config_file_default)
            self._config.read(config_file_default)
        assert os.path.isfile(config_file)
        self.config_file = config_file
        self._config.read(config_file)
    
    def get_data_section(self):
        self.graph_dir = self._config.get('data', 'graph_dir')

        self.dft_data_dir = self._config.get('data', 'DFT_data_dir')
        self.processed_data_dir = self._config.get('data', 'processed_data_dir')
        
        self.save_graph_dir = self._config.get('data', 'save_graph_dir')
        self.target_data = self._config.get('data', 'target_data')
        self.dataset_name = self._config.get('data', 'dataset_name')
        
        self.get_olp = self._config.getboolean('data', 'get_overlap')
        
        self.only_ij = False # todo
    
    def set_dtype(self, dtype):
        if dtype == 'float':
            self.torch_dtype = torch.float32
            self.np_dtype = np.float32
        elif dtype == 'double':
            self.torch_dtype = torch.float64
            self.np_dtype = np.float64
        else:
            raise NotImplementedError
    
    def get_basic_section(self):
        self.device = torch.device(self._config.get('basic', 'device'))
        dtype = self._config.get('basic', 'dtype')
        self.set_dtype(dtype)

class TrainConfig(BaseConfig):
    def __init__(self, config_file):
        super().__init__()
        train_default = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_configs/train_default.ini')

        print(f'Loading train config from: {config_file}')
        self.get_config(config_file, config_file_default=train_default)
        
        self.get_basic_section()
        self.get_data_section()
        self.get_train_section()
        
        if self.use_new_hypp:
            self.get_hypp_section()
        
        if self.checkpoint_dir:
            assert self.checkpoint_dir.endswith('.pkl'), 'checkpoint should point to model.pkl or best_model.pkl'
            config1_path = os.path.join(os.path.dirname(self.checkpoint_dir), 'src/train.ini')
            assert os.path.isfile(config1_path), 'cannot find train.ini under checkpoint dir'
            print(f'Overwriting sections [hyperparameters], [target] and [network] using train config in checkpoint: {config1_path}')
            self.get_config(config1_path, config_file_default=train_default)
        
        # overwrite settings using those in config
        if not self.use_new_hypp:
            self.get_hypp_section()
        
        self.get_target_section()
        self.get_network_section()
        
        self._target_set_flag = False
        # set_target should be called once dataset has been prepared
        
    def get_basic_section(self):
        super().get_basic_section()
        self.seed = self._config.getint('basic', 'seed')
        self.checkpoint_dir = self._config.get('basic', 'checkpoint_dir')
        self.simp_out = self._config.getboolean('basic', 'simplified_output')
        self.use_new_hypp = self._config.getboolean('basic', 'use_new_hypp')

        # = save to time folder =
        save_dir = self._config.get('basic', 'save_dir')
        additional_folder_name = self._config.get('basic', 'additional_folder_name')
        self.save_dir = os.path.join(save_dir, str(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
        if additional_folder_name:
            self.save_dir = self.save_dir + '_' + additional_folder_name
        assert not os.path.exists(self.save_dir)
        
    def get_train_section(self):
        self.batch_size = self._config.getint('train', 'batch_size')
        self.num_epoch = self._config.getint('train', 'num_epoch')
        
        self.min_lr = self._config.getfloat('train', 'min_lr')
            
        ev = self._config.get('train', 'extra_validation')
        self.extra_val = eval(ev) if ev else []
        self.extra_val_test_only = self._config.getboolean('train', 'extra_val_test_only')
        
        self.train_ratio = self._config.getfloat('train', 'train_ratio')
        self.val_ratio = self._config.getfloat('train', 'val_ratio')
        self.test_ratio = self._config.getfloat('train', 'test_ratio')
        
        self.train_size = self._config.getint('train', 'train_size')
        self.val_size = self._config.getint('train', 'val_size')
        self.test_size = self._config.getint('train', 'test_size')
    
    def get_hypp_section(self):
        self.lr = self._config.getfloat('hyperparameters', 'learning_rate')
        self.adam_betas = eval(self._config.get('hyperparameters', 'Adam_betas'))

        self.scheduler_type = self._config.getint('hyperparameters', 'scheduler_type')
        ts = self._config.get('hyperparameters', 'scheduler_params')
        if ts:
            ts = 'dict' + ts
            self.scheduler_params = eval(ts)
        else:
            self.scheduler_params = dict()
        
        self.revert_decay_patience = self._config.getint('hyperparameters', 'revert_decay_patience')
        self.revert_decay_rate = self._config.getfloat('hyperparameters', 'revert_decay_rate')
        
    def get_target_section(self):
        # target
        self.target = self._config.get('target', 'target')
        tbt = self._config.get('target', 'target_blocks_type')
        assert len(tbt) > 0, 'Invalid target_blocks_type'
        self.tbt0 = tbt[0].lower()
        self._target_blocks = None
        if self.tbt0 == 's':
            self._target_blocks = eval(self._config.get('target', 'target_blocks'))
        sep = self._config.get('target', 'selected_element_pairs')
        self.element_pairs = eval(sep) if sep else None
        self.convert_net_out = self._config.getboolean('target', 'convert_net_out')
        assert not self.convert_net_out
    
    def get_network_section(self):
        # network
        self.cutoff_radius = self._config.getfloat('network', 'cutoff_radius')
        self.only_ij = self._config.getboolean('network', 'only_ij')
        assert not self.only_ij
        self.no_parity = self._config.getboolean('network', 'ignore_parity')
        
        sh_lmax = self._config.get('network', 'spherical_harmonics_lmax')
        sbf_irreps = self._config.get('network', 'spherical_basis_irreps')
        if sh_lmax:
            assert not sbf_irreps, 'spherical_harmonics_lmax and spherical_basis_irreps cannot be provided simultaneously'
            sh_lmax = int(sh_lmax)
            self.irreps_sh = Irreps([(1, (i, 1 if self.no_parity else (-1) ** i)) for i in range(sh_lmax + 1)])
            self.use_sbf = False
        else:
            assert sbf_irreps, 'at least one of spherical_harmonics_lmax and spherical_basis_irreps should be provided'
            self.irreps_sh = Irreps(sbf_irreps)
            self.use_sbf = True
        
        irreps_mid = self._config.get('network', 'irreps_mid')
        if irreps_mid:
            self.irreps_mid_node = irreps_mid
            self._irreps_post_node = irreps_mid
            self.irreps_mid_edge = irreps_mid
        irreps_embed = self._config.get('network', 'irreps_embed')
        if irreps_embed:
            self.irreps_embed_node = irreps_embed
            self.irreps_edge_init = irreps_embed
        if self.target in ['hamiltonian']:
            self.irreps_out_node = '1x0e'
        self.num_blocks = self._config.getint('network', 'num_blocks')
        
        # ! post edge
        for name in ['irreps_embed_node', 'irreps_edge_init', 'irreps_mid_node', 'irreps_out_node', 'irreps_mid_edge']:
            irreps = self._config.get('network', name)
            if irreps:
                delattr(self, name)
                setattr(self, name, irreps)
        
        self._net_out_irreps = self._config.get('network', 'out_irreps')
        self._irreps_post_edge = self._config.get('network', 'irreps_post_edge')
    
    def set_target(self, orbital_types, index_to_Z, spinful, output_file):
        atom_orbitals = {}
        for Z, orbital_type in zip(index_to_Z, orbital_types):
            atom_orbitals[str(Z.item())] = orbital_type
            
        target_blocks, net_out_irreps, irreps_post_edge = orbital_analysis(atom_orbitals, self.tbt0, spinful, targets=self._target_blocks, 
                                                                           element_pairs=self.element_pairs, no_parity=self.no_parity, verbose=output_file)
        
        self._target_blocks = target_blocks
        if not self._net_out_irreps:
            self._net_out_irreps = net_out_irreps
        if not self._irreps_post_edge:
            self._irreps_post_edge = irreps_post_edge
            
        if self.convert_net_out:
            self._net_out_irreps = Irreps(self._net_out_irreps).sort().irreps.simplify()
            
        if spinful:
            if_verbose = False
            if output_file:
                if_verbose = True
            self._irreps_post_node = refine_post_node(irreps_post_node=self._irreps_post_node,
                                                      irreps_mid_node=self.irreps_mid_node,
                                                      irreps_mid_edge=self.irreps_mid_edge,
                                                      irreps_sh=self.irreps_sh,
                                                      irreps_post_edge=self._irreps_post_edge,
                                                      if_verbose=if_verbose)
            
        self._target_set_flag = True
    
    @property
    def target_blocks(self):
        assert self._target_set_flag
        return self._target_blocks
    
    @property
    def net_out_irreps(self):
        assert self._target_set_flag
        return self._net_out_irreps
    
    @property
    def irreps_post_edge(self):
        assert self._target_set_flag
        return self._irreps_post_edge
    
    @property
    def irreps_post_node(self):
        assert self._target_set_flag
        ipn = self._config.get('network', 'irreps_post_node')
        if ipn:
            return ipn
        else:
            return self._irreps_post_node
    
            
class EvalConfig(BaseConfig):
    def __init__(self, config_file):
        super().__init__()
        eval_default = (os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_configs/eval_default.ini'))
        
        print(f'Loading evaluation config from: {config_file}')
        self.get_config(config_file, config_file_default=eval_default)
        
        self.get_basic_section()
        self.get_data_section()
        
    def get_basic_section(self):
        # basic
        self.model_dir = self._config.get('basic', 'trained_model_dir')
        super().get_basic_section()
        self.out_dir = self._config.get('basic', 'output_dir')
        os.makedirs(self.out_dir, exist_ok=True)
        self.target = self._config.get('basic', 'target')
        self.inference = self._config.getboolean('basic', 'inference')
        self.test_only = self._config.getboolean('basic', 'test_only')
        if self.test_only: assert not self.inference
        