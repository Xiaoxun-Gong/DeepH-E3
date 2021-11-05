import sys
sys.path.append('/home/gongxx/projects/DeepH/e3nn_DeepH/e3Aij')
from e3Aij import AijData
import torch



edge_Aij = True
dataset = AijData(
    raw_data_dir='/home/lihe/hdd/materials_data/BG_300_ij/processed',
    graph_dir='/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/1105_BG300_ij/graph',
    target='hamiltonian',
    dataset_name='BG300ij',
    multiprocessing=False,
    radius=7.2,
    max_num_nbr=0,
    edge_Aij=edge_Aij,
    only_ij=True,
    default_dtype_torch=torch.float32
)