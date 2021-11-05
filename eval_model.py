import time
import sys
import os
import argparse
import warnings
import numpy as np
import h5py

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from e3Aij import AijData, Collater
from e3Aij.graph import convert_ijji
from e3Aij.utils import TrainConfig, EvalConfig, process_targets
from e3Aij.e3modules import e3TensorDecomp

parser = argparse.ArgumentParser(description='Evaluate trained e3Aij model')
parser.add_argument('--config', type=str, help='Config file for evaluation')
args = parser.parse_args()

eval_config = EvalConfig(args.config)

# = default dtype =
torch.set_default_dtype(eval_config.torch_dtype)

print('\n------- Preparation of graph data for evaluation -------')
# get graph data
edge_Aij = True
dataset = AijData(
    raw_data_dir=eval_config.processed_data_dir,
    graph_dir=eval_config.graph_dir,
    target=eval_config.target,
    dataset_name=eval_config.dataset_name,
    multiprocessing=False,
    radius=eval_config.cutoff_radius,
    max_num_nbr=0,
    edge_Aij=edge_Aij,
    inference=False, # todo
    only_ij=eval_config.only_ij,
    default_dtype_torch=torch.get_default_dtype()
)
collate = Collater(edge_Aij) # todo: multiple data
atom_num_orbitals = [sum(map(lambda x: 2 * x + 1, atom_orbital_types)) for atom_orbital_types in dataset.info['orbital_types']]

num_structure = len(dataset.slices['x']) - 1
H_pred_list = [{} for _ in range(num_structure)]

print('\n------- Finding trained models -------')
# get models
model_path_list = []
for root, dirs, files in os.walk(eval_config.model_dir):
    if 'best_model.pkl' in files and 'src' in dirs:
        model_path_list.append(os.path.abspath(root))
assert len(model_path_list) > 0, 'Cannot find any model'
print(f'Successfully found {len(model_path_list)} model(s):')
for index_model, model_path in enumerate(model_path_list):
    print(f'model {index_model}:', model_path)


print('\n------- Evaluating model -------')
for index_model, model_path in enumerate(model_path_list):
    print(f'\nLoading model {index_model}:')
    train_config = TrainConfig(os.path.join(model_path, 'src/train.ini'), inference=True)
    assert train_config.torch_dtype == eval_config.torch_dtype, f'model uses dtype {train_config.torch_dtype} but evaluation requires dtype {eval_config.torch_dtype}'
    assert train_config.target == eval_config.target, f'model predicts {train_config.target} but evaluation requires prediction of {eval_config.target}'
    if train_config.cutoff_radius != eval_config.cutoff_radius:
        warnings.warn(f'Model has cutoff radius r={train_config.cutoff_radius} but evaluation requires r={eval_config.cutoff_radius}')
    assert train_config.only_ij == eval_config.only_ij, f'evaluation uses {"un" if eval_config.only_ij else ""}directed graph but model does not'
    
    train_config.set_target(dataset.info['orbital_types'], dataset.info['index_to_Z'], None)
    equivariant_blocks, out_js_list, out_slices = process_targets(dataset.info['orbital_types'], dataset.info['index_to_Z'], train_config.target_blocks)
    construct_kernel = e3TensorDecomp(train_config.net_out_irreps, out_js_list, default_dtype_torch=torch.get_default_dtype(), device_torch=eval_config.device)
    
    sys.path.append(os.path.join(model_path, 'src'))
    checkpoint = torch.load(os.path.join(model_path, 'best_model.pkl'))
    begin = time.time()
    print('Building model...')
    from build_model import net
    print(f'Finished building model, cost {time.time() - begin:.2f} seconds.')
    net.load_state_dict(checkpoint['state_dict'])
    net.to(eval_config.device)
    net.eval()
    
    for index_stru, data in enumerate(dataset):
        print(f'Getting model {index_model} output on structure "{data.stru_id}"...')
        batch = collate([data])
        with torch.no_grad():
            output, output_edge = net(batch.to(device=eval_config.device))
            H_pred = construct_kernel.get_H(output_edge).detach().numpy() # .cpu()
        for index_edge in range(batch.edge_index.shape[1]):
            key_term = str(batch.edge_key[index_edge].tolist())
            i, j = batch.x[batch.edge_index[:, index_edge]]
            if key_term not in H_pred_list[index_stru].keys():
                H_pred_list[index_stru][key_term] = np.full((atom_num_orbitals[i], atom_num_orbitals[j]), np.nan, dtype=eval_config.np_dtype)
            N_M_str_edge = f'{dataset.info["index_to_Z"][i].item()} {dataset.info["index_to_Z"][j].item()}'
            for index_target, equivariant_block in enumerate(equivariant_blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    if N_M_str == N_M_str_edge:
                        slice_row = slice(block_slice[0], block_slice[1])
                        slice_col = slice(block_slice[2], block_slice[3])
                        len_row = block_slice[1] - block_slice[0]
                        len_col = block_slice[3] - block_slice[2]
                        slice_out = slice(out_slices[index_target], out_slices[index_target + 1])
                        H_pred_list[index_stru][key_term][slice_row, slice_col] = H_pred[index_edge, slice_out].reshape(len_row, len_col)
                    
    print(f'Finished evaluating model {index_model} on all structures')
    
print('\nFinished evaluating all models')
# convert ijji
if eval_config.only_ij:
    for index_stru in range(num_structure):
        H_dict_inv = {}
        for key_term, hopping in H_pred_list[index_stru].items():
            assert np.all(np.isnan(hopping)==False), f'Some orbitals are not predicted'
            key_inv = str(convert_ijji(key_term))
            if key_inv == key_term:
                H_pred_list[index_stru][key_term] = (hopping + hopping.T) / 2.0
            else:
                assert key_inv not in H_pred_list[index_stru].keys()
                H_dict_inv[key_inv] = hopping.T
        H_pred_list[index_stru].update(H_dict_inv)

print('\n------- Output -------')
for H_dict, data in zip(H_pred_list, dataset):
    print(f'Writing output to "{data.stru_id}_Hpred.h5"')
    with h5py.File(os.path.join(eval_config.out_dir, f'{data.stru_id}_Hpred.h5'), 'w') as f:
        for k, v in H_dict.items():
            f[k] = v


# ! below might need change in different runs: 
# ! state_dict_dir net net_out_irreps set_mask
# state_dict_dir = '/home/gongxx/projects/DeepH/e3nn_DeepH/test_runs/1006_first_largeData/2021-10-06_20-10-07_1x1/best_model.pkl'
# torch.set_default_dtype(torch.float32)
# net_out_irreps = '4x0e+4x1e+4x2e'

# edge_Aij = True
# data = AijData(
#     raw_data_dir="/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0923/processed",
#     graph_dir="/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0923/graph_data",
#     target="hamiltonian",
#     dataset_name="test_MoS2_new1",
#     multiprocessing=False,
#     # radius=5.0,
#     radius=7.4,
#     max_num_nbr=0,
#     edge_Aij=edge_Aij,
#     default_dtype_torch=torch.get_default_dtype()
# )
# out_js_list = data.set_mask([{"42 42": [3, 3], "42 16": [3, 2], "16 42": [2, 3], "16 16": [2, 2]}])

# num_species = len(data.info["index_to_Z"])
# begin = time.time()
# print('Building model...')
# net = Net(
#     num_species=num_species,
#     irreps_embed_node="32x0e",
#     irreps_edge_init="64x0e",
#     irreps_sh='1x0e + 1x1o + 1x2e + 1x3o + 1x4e + 1x5o + 1x6e',
#     irreps_mid_node='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o',#'16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+4x4e+4x4o+4x5e+4x5o+4x6e+4x6o',
#     irreps_post_node='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o',
#     irreps_out_node="1x0e",
#     irreps_mid_edge='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o',
#     irreps_post_edge='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o',
#     irreps_out_edge=net_out_irreps,
#     num_block=3,
#     r_max=7.4,
#     use_sc=False,
#     if_sort_irreps=False
# )
# print(f'Finished building model, cost {time.time() - begin:.2f} seconds.')

# net.load_state_dict(torch.load(state_dict_dir)['state_dict'])
# net.eval()

# test_loader = DataLoader(data, batch_size=1,
#                             shuffle=False, sampler=SubsetRandomSampler([0]),
#                             collate_fn=Collater(edge_Aij))

# rotate_kernel = Rotate(torch.get_default_dtype())
# construct_kernel = e3TensorDecomp(net_out_irreps, *out_js_list[0], default_dtype_torch=torch.get_default_dtype())

# for batch in test_loader:
#     output, output_edge = net(batch)

#     H_pred = construct_kernel.get_H(output_edge)
    
#     H_pred = rotate_kernel.openmx2wiki_H(H_pred, *out_js_list[0])
    
#     batch.label = rotate_kernel.openmx2wiki_H(batch.label, *out_js_list[0])
    
#     print(torch.sum(torch.abs(batch.label - H_pred), dim=0) / H_pred.shape[0])
#     print('==================')
#     for index in range(batch.edge_attr.shape[0]):
#         print('edge number:', index)
#         print(batch.edge_key[index])
#         print('dataset:')
#         print(batch.label[index])
#         print('predicted:')
#         print(H_pred[index])
#         print('--------------------------')
        