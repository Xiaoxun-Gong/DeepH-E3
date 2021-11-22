import sys
sys.path.append('/home/gongxx/projects/DeepH/e3nn_DeepH/e3Aij')
from e3Aij import AijData
import torch


edge_Aij = True
dataset = AijData(
    raw_data_dir='/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0923/processed',
    graph_dir='/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0923/test_fwad',
    target='hamiltonian',
    dataset_name='fwad',
    multiprocessing=False,
    radius=7.2,
    max_num_nbr=0,
    edge_Aij=edge_Aij,
    only_ij=False,
    default_dtype_torch=torch.float32,
    load_graph=False
)