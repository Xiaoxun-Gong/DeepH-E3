import time
import sys

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader

sys.path.append('/home/gongxx/projects/DeepH/e3nn_DeepH/e3Aij/') # !
from e3Aij import Net, AijData, Collater, Rotate
from e3Aij.utils import construct_H, e3TensorDecomp

# ! below might need change in different runs: 
# ! state_dict_dir net net_out_irreps set_mask
state_dict_dir = '/home/gongxx/projects/DeepH/e3nn_DeepH/test_runs/1006_first_largeData/2021-10-06_20-10-07_1x1/best_model.pkl'
torch.set_default_dtype(torch.float32)
net_out_irreps = '4x0e+4x1e+4x2e'

edge_Aij = True
dataset = AijData(
    raw_data_dir="/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0923/processed",
    graph_dir="/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0923/graph_data",
    target="hamiltonian",
    dataset_name="test_MoS2_new1",
    multiprocessing=False,
    # radius=5.0,
    radius=7.4,
    max_num_nbr=0,
    edge_Aij=edge_Aij,
    default_dtype_torch=torch.get_default_dtype()
)
out_js_list = dataset.set_mask([{"42 42": [3, 3], "42 16": [3, 2], "16 42": [2, 3], "16 16": [2, 2]}])

num_species = len(dataset.info["index_to_Z"])
begin = time.time()
print('Building model...')
net = Net(
    num_species=num_species,
    irreps_embed_node="32x0e",
    irreps_edge_init="64x0e",
    irreps_sh='1x0e + 1x1o + 1x2e + 1x3o + 1x4e + 1x5o + 1x6e',
    irreps_mid_node='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o',#'16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+4x4e+4x4o+4x5e+4x5o+4x6e+4x6o',
    irreps_post_node='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o',
    irreps_out_node="1x0e",
    irreps_mid_edge='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o',
    irreps_post_edge='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o',
    irreps_out_edge=net_out_irreps,
    num_block=3,
    use_sc=False,
    r_max = 7.4,
    if_sort_irreps=False
)
print(f'Finished building model, cost {time.time() - begin:.2f} seconds.')

net.load_state_dict(torch.load(state_dict_dir)['state_dict'])
net.eval()

test_loader = DataLoader(dataset, batch_size=1,
                            shuffle=False, sampler=SubsetRandomSampler([0]),
                            collate_fn=Collater(edge_Aij))

rotate_kernel = Rotate(torch.get_default_dtype())
construct_kernel = e3TensorDecomp(net_out_irreps, *out_js_list[0], default_dtype_torch=torch.get_default_dtype())

for batch in test_loader:
    output, output_edge = net(batch)

    H_pred = construct_kernel.get_H(output_edge)
    
    H_pred = rotate_kernel.openmx2wiki_H(H_pred, *out_js_list[0])
    
    batch.label = rotate_kernel.openmx2wiki_H(batch.label, *out_js_list[0])
    
    print(torch.sum(torch.abs(batch.label - H_pred), dim=0) / H_pred.shape[0])
    print('==================')
    for index in range(batch.edge_attr.shape[0]):
        print('edge number:', index)
        print(batch.edge_key[index])
        print('dataset:')
        print(batch.label[index])
        print('predicted:')
        print(H_pred[index])
        print('--------------------------')
        