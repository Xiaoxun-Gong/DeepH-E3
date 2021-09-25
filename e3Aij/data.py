from typing import Union, Dict, Tuple, List
import os
import time
import tqdm

from pymatgen.core.structure import Structure
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from pathos.multiprocessing import ProcessingPool as Pool

from .graph import get_graph, load_orbital_types

class AijData(InMemoryDataset):
    def __init__(self, raw_data_dir: str, graph_dir: str, target: str,
                 dataset_name : str, multiprocessing: bool, radius: float, max_num_nbr: int, edge_Aij: bool,
                 default_dtype_torch, nums: int = None, inference:bool = False):
        """
        :param raw_data_dir: 原始数据目录, 允许存在嵌套
when interface == 'h5',
raw_data_dir
├── 00
│     ├──<target>s.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── 01
│     ├──<target>s.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── 02
│     ├──<target>s.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── ...
        :param graph_dir: 存储图的目录
        :param multiprocessing: 多进程生成图
        :param radius: 生成图的截止半径
        :param max_num_nbr: 生成图限制最大近邻数, 为 0 时不限制
        :param edge_Aij: 图的边是否一一对应 Aij, 如果是为 False 则存在两套图的连接, 一套用于节点更新, 一套用于记录 Aij
        :param default_dtype_torch: 浮点数数据类型
        """
        self.raw_data_dir = raw_data_dir
        assert dataset_name.find('-') == -1, '"-" can not be included in the dataset name'
        if target == 'hamiltonian':
            graph_file_name = f'HGraph-{dataset_name}-{radius}r{max_num_nbr}mn-edge{"" if edge_Aij else "!"}=Aij.pkl'
        elif target == 'density_matrix':
            graph_file_name = f'DMGraph-{dataset_name}-{radius}r{max_num_nbr}mn-{edge_Aij}edge.pkl'
        else:
            raise ValueError('Unknown prediction target: {}'.format(target))
        self.data_file = os.path.join(graph_dir, graph_file_name)
        os.makedirs(graph_dir, exist_ok=True)
        self.data, self.slices = None, None
        self.target = target
        self.target_file_name = f'{self.target}s.h5'
        self.dataset_name = dataset_name
        self.multiprocessing = multiprocessing
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.edge_Aij = edge_Aij
        self.default_dtype_torch = default_dtype_torch

        self.nums = nums
        self.inference = inference
        self.transform = None
        self.__indices__ = None
        self.__data_list__ = None
        self._indices = None
        self._data_list = None

        print(f'Graph data file: {graph_file_name}')
        if os.path.exists(self.data_file):
            print('Use existing graph data file')
        else:
            print('Process new data file......')
            self.process()
        begin = time.time()
        loaded_data = torch.load(self.data_file)
        self.data, self.slices, self.info = loaded_data
        print(f'Finish loading the processed {len(self)} structures (spinful: {self.info["spinful"]}, '
              f'the number of atomic types: {len(self.info["index_to_Z"])}), cost {time.time() - begin:.2f} seconds')

    def process_worker(self, folder, **kwargs):
        stru_id = os.path.split(folder)[-1]

        structure = Structure(np.loadtxt(os.path.join(folder, 'lat.dat')).T,
                              np.loadtxt(os.path.join(folder, 'element.dat')),
                              np.loadtxt(os.path.join(folder, 'site_positions.dat')).T,
                              coords_are_cartesian=True,
                              to_unit_cell=False)

        cart_coords = torch.tensor(structure.cart_coords, dtype=self.default_dtype_torch)
        frac_coords = torch.tensor(structure.frac_coords, dtype=self.default_dtype_torch)
        numbers = torch.tensor(structure.atomic_numbers)
        structure.lattice.matrix.setflags(write=True)
        lattice = torch.tensor(structure.lattice.matrix, dtype=self.default_dtype_torch)
        return get_graph(cart_coords, frac_coords, numbers, stru_id, r=self.radius, max_num_nbr=self.max_num_nbr,
                         edge_Aij=self.edge_Aij, lattice=lattice, default_dtype_torch=self.default_dtype_torch,
                         data_folder=folder, target_file_name=self.target_file_name, inference=self.inference, **kwargs)

    def process(self):
        begin = time.time()
        folder_list = []
        for root, dirs, files in os.walk(self.raw_data_dir):
            if {'element.dat', 'orbital_types.dat', 'info.json', 'lat.dat', 'site_positions.dat'}.issubset(files):
                if self.inference == True or self.target_file_name in files:
                    folder_list.append(root)
        folder_list = folder_list[: self.nums]
        assert len(folder_list) != 0, "Can not find any structure"
        print('Found %d structures, have cost %d seconds' % (len(folder_list), time.time() - begin))

        if self.multiprocessing:
            print('Use multiprocessing')
            with Pool() as pool:
                data_list = list(tqdm.tqdm(pool.imap(self.process_worker, folder_list), total=len(folder_list)))
        else:
            data_list = [self.process_worker(folder) for folder in tqdm.tqdm(folder_list)]
        print('Finish processing %d structures, have cost %d seconds' % (len(data_list), time.time() - begin))
        index_to_Z, Z_to_index = self.element_statistics(data_list)

        spinful = data_list[0].spinful
        for d in data_list:
            assert spinful == d.spinful
            
        _, orbital_types = load_orbital_types(path=os.path.join(folder_list[0], 'orbital_types.dat'),
                                           return_orbital_types=True) 
        elements = np.loadtxt(os.path.join(folder_list[0], 'element.dat'))
        orbital_types_new = []
        for i in range(len(index_to_Z)):
            orbital_types_new.append(orbital_types[np.where(elements == index_to_Z[i].numpy())[0][0]])
        #TODO 数据集包含不同元素

        data, slices = self.collate(data_list)
        torch.save((data, slices, dict(spinful=spinful, index_to_Z=index_to_Z, Z_to_index=Z_to_index, orbital_types=orbital_types_new)), self.data_file)
        print('Finish saving %d structures to raw_data_file, have cost %d seconds' % (len(data_list), time.time() - begin))

    def element_statistics(self, data_list):
        # TODO 没有处理数据集包括不同元素组成的情况
        index_to_Z, inverse_indices = torch.unique(data_list[0].x, sorted=True, return_inverse=True)
        Z_to_index = torch.full((100,), -1, dtype=torch.int64)
        Z_to_index[index_to_Z] = torch.arange(len(index_to_Z))

        for data in data_list:
            data.x = Z_to_index[data.x]

        return index_to_Z, Z_to_index

    def set_mask(self, targets):
        # = process the orbital indices into block slices =
        orbital_types = self.info['orbital_types']
        orbital_types = list(map(lambda x: np.array(x, dtype=np.int32), orbital_types))
        orbital_types_cumsum = list(map(lambda x: np.concatenate([np.zeros(1, dtype=np.int32), 
                                                                  np.cumsum(2 * x + 1)]), orbital_types))
        #! equivariant_blocks, out_js_list, label init
        equivariant_blocks, out_js_list = [], []
        equivariant_block = dict()
        for target in targets:
            out_js = None
            for N_M_str, block_indices in target.items():
                i, j = map(lambda x: self.info["Z_to_index"][int(x)], N_M_str.split())
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
        self.equivariant_blocks = equivariant_blocks # [{"42 42": [3, 6, 9, 14], "42 16": [3, 6, 8, 13], "16 42": [2, 5, 9, 14], "16 16": [2, 5, 8, 13]}]
        # self.equivariant_blocks = [{"42 42": [3, 6, 3, 6]}]
        # self.equivariant_blocks = [{"42 42": [0, 1, 0, 1], "42 16": [0, 1, 0, 1], "16 42": [0, 1, 0, 1], "16 16": [0, 1, 0, 1]}]
        assert len(self.equivariant_blocks) == 1

        begin = time.time()
        print("Set mask for dataset")
        out_fea_len = 0
        out_indices_list = [0]
        for index_out, equivariant_block in enumerate(self.equivariant_blocks):
            block_len = None
            for N_M_str, block_slice in equivariant_block.items():
                if block_len is None:
                    block_len = (block_slice[1] - block_slice[0]) * (block_slice[3] - block_slice[2])
                else:
                    assert block_len == (block_slice[1] - block_slice[0]) * (block_slice[3] - block_slice[2])
            out_fea_len += block_len
            out_indices_list.append(out_indices_list[-1] + block_len)

        data_list_mask = []
        for data in self:
            assert data.spinful == False
            if data.Aij is not None:
                if not torch.all(data.Aij_mask):
                    raise NotImplementedError("Not yet have support for graph radius including Aij without calculation")

            # mask = torch.zeros(data.num_edges, out_fea_len, dtype=torch.int8)
            # label = torch.zeros(data.num_edges, out_fea_len, dtype=torch.get_default_dtype())
            block_size = map(lambda x: 2 * x + 1, out_js_list[0])
            label = torch.zeros(data.num_edges, *block_size, dtype=torch.get_default_dtype()) # TODO have only considered one target here

            atomic_number_edge_i = data.x[data.edge_index[0]]
            atomic_number_edge_j = data.x[data.edge_index[1]]

            for index_out, equivariant_block in enumerate(self.equivariant_blocks):
                assert index_out == 0
                for N_M_str, block_slice in equivariant_block.items():
                    condition_atomic_number_i, condition_atomic_number_j = map(lambda x: self.info["Z_to_index"][int(x)], N_M_str.split())
                    condition_slice_i = slice(block_slice[0], block_slice[1])
                    condition_slice_j = slice(block_slice[2], block_slice[3])
                    if data.Aij is not None:
                        condition_index = torch.where(
                            (atomic_number_edge_i == condition_atomic_number_i)
                            & (atomic_number_edge_j == condition_atomic_number_j)
                        )
                        label[condition_index] += data.Aij[:, condition_slice_i, condition_slice_j][condition_index]
            del data.Aij_mask
            if data.Aij is not None:
                data.label = label
                del data.Aij
            data_list_mask.append(data)

        self.__indices__ = None
        self.__data_list__ = None
        self._indices = None
        self._data_list = None
        data, slices = self.collate(data_list_mask)
        self.data, self.slices = data, slices
        print(f"Finished setting mask for dataset, cost {time.time() - begin:.2f} seconds")

        return out_js_list