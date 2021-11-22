import os
import time
from configparser import ConfigParser
import numpy as np

import torch

from e3nn.o3 import Irreps

from .data import AijData
from .utils import orbital_analysis

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

        # = save to time folder =
        self.save_dir = os.path.join(self.save_dir, str(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
        self.save_dir = self.save_dir + '_' + self.additional_folder_name
        assert not os.path.exists(self.save_dir)
        
        # train
        self.graph_dir = config.get('train', 'graph_dir')
        self.batch_size = config.getint('train', 'batch_size')
        self.num_epoch = config.getint('train', 'num_epoch')
        # rde = config.get('train', 'revert_decay_epoch')
        # self.revert_decay_epoch = eval(rde) if rde else []
        # rdg = config.get('train', 'revert_decay_gamma')
        # self.revert_decay_gamma = eval(rdg) if rdg else []
        self.revert_decay_patience = config.getint('train', 'revert_decay_patience')
        self.revert_decay_rate = config.getfloat('train', 'revert_decay_rate')
        ts = config.get('train', 'ReduceLROnPlateau')
        self.torch_scheduler = None
        if ts:
            self.torch_scheduler = eval(ts)
        ev = config.get('train', 'extra_validation')
        self.extra_val = eval(ev) if ev else []
        self.min_lr = config.getfloat('train', 'min_lr')
        
        # hyperparameters
        self.lr = config.getfloat('train', 'learning_rate')
        self.train_ratio = config.getfloat('train', 'train_ratio')
        self.val_ratio = config.getfloat('train', 'val_ratio')
        self.test_ratio = config.getfloat('train', 'test_ratio')
        
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
        config = ConfigParser(inline_comment_prefixes=(';',))
        config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_configs/train_default.ini'))
        for config_file in args:
            if verbose:
                print('Loading train config from: ', config_file)
            config.read(config_file)
        assert config['basic']['torch_dtype'] in ['float', 'double'], f"{config['basic']['torch_dtype']}"
        assert config['target']['target'] in ['hamiltonian']
        assert config['target']['target_blocks_type'][0] in ['s', 'a', 'o', 'd']

        return config
    
    def set_target(self, orbital_types, index_to_Z, spinful, output_file):
        atom_orbitals = {}
        for Z, orbital_type in zip(index_to_Z, orbital_types):
            atom_orbitals[str(Z.item())] = orbital_type
            
        target_blocks, net_out_irreps, irreps_post_edge = orbital_analysis(atom_orbitals, self.tbt0, spinful, targets=self.target_blocks, element_pairs=self.element_pairs, verbose=output_file)
        
        if not self.target_blocks:
            self.target_blocks = target_blocks
        if not self.net_out_irreps:
            self.net_out_irreps = net_out_irreps
        if not self.irreps_post_edge:
            self.irreps_post_edge = irreps_post_edge
            
        if self.convert_net_out:
            self.net_out_irreps = Irreps(self.net_out_irreps).sort().irreps.simplify()
    
    def overwrite(self, config_file, verbose=True):
        config = self.get_config(config_file, verbose=verbose)
        
        # target
        self.target = config.get('target', 'target') # ! 'target and graph data does not match'
        self.tbt0 = config.get('target', 'target_blocks_type')[0].lower()
        self.target_blocks = None
        if self.tbt0 == 's':
            self.target_blocks = eval(config.get('target', 'target_blocks'))
        sep = config.get('target', 'selected_element_pairs')
        self.element_pairs = eval(sep) if sep else None
        self.convert_net_out = config.getboolean('target', 'convert_net_out')
        
        # network
        self.cutoff_radius = config.getfloat('network', 'cutoff_radius')
        self.only_ij = config.getboolean('network', 'only_ij')
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
        config = ConfigParser(inline_comment_prefixes=(';',))
        config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_configs/eval_default.ini'))
        for config_file in args:
            print('Loading eval config from: ', config_file)
            config.read(config_file)
        assert config['basic']['dtype'] in ['float', 'double'], f"{config['basic']['dtype']}"
        assert config['basic']['target'] in ['hamiltonian']
        
        return config