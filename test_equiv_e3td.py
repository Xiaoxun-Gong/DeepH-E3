from torch import nn, optim
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from e3Aij import AijData, Net, Collater, Rotate
from e3Aij.utils import assemble_H
from e3Aij.e3modules import construct_H, e3TensorDecomp

torch_dtype = torch.float32
torch.set_default_dtype(torch_dtype)
test_net = True
print_H_block = True
state_dict_dir = '/home/gongxx/projects/DeepH/e3nn_DeepH/test_runs/1011_0tMoS2/2021-10-24_14-27-35_allblocks_minimum/best_model.pkl'

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
    default_dtype_torch=torch.get_default_dtype(),
    only_ij=True
)
# targets = [{"42 42": [3, 5], "42 16": [3, 4], "16 42": [2, 5], "16 16": [2, 4]}, {"42 42": [3, 4], "42 16": [3, 3], "16 42": [2, 4], "16 16": [2, 3]}]
targets = [{'42 42': [0, 0], '42 16': [0, 0], '16 42': [0, 0], '16 16': [0, 0]}, {'42 42': [0, 1], '42 16': [0, 1], '16 42': [0, 1], '16 16': [0, 1]}, {'42 42': [0, 2], '16 42': [0, 2]}, {'42 42': [0, 3], '42 16': [0, 2], '16 42': [0, 3], '16 16': [0, 2]}, {'42 42': [0, 4], '42 16': [0, 3], '16 42': [0, 4], '16 16': [0, 3]}, {'42 42': [0, 5], '42 16': [0, 4], '16 42': [0, 5], '16 16': [0, 4]}, {'42 42': [0, 6], '16 42': [0, 6]}, {'42 42': [1, 0], '42 16': [1, 0], '16 42': [1, 0], '16 16': [1, 0]}, {'42 42': [1, 1], '42 16': [1, 1], '16 42': [1, 1], '16 16': [1, 1]}, {'42 42': [1, 2], '16 42': [1, 2]}, {'42 42': [1, 3], '42 16': [1, 2], '16 42': [1, 3], '16 16': [1, 2]}, {'42 42': [1, 4], '42 16': [1, 3], '16 42': [1, 4], '16 16': [1, 3]}, {'42 42': [1, 5], '42 16': [1, 4], '16 42': [1, 5], '16 16': [1, 4]}, {'42 42': [1, 6], '16 42': [1, 6]}, {'42 42': [2, 0], '42 16': [2, 0]}, {'42 42': [2, 1], '42 16': [2, 1]}, {'42 42': [2, 2]}, {'42 42': [2, 3], '42 16': [2, 2]}, {'42 42': [2, 4], '42 16': [2, 3]}, {'42 42': [2, 5], '42 16': [2, 4]}, {'42 42': [2, 6]}, {'42 42': [3, 0], '42 16': [3, 0], '16 42': [2, 0], '16 16': [2, 0]}, {'42 42': [3, 1], '42 16': [3, 1], '16 42': [2, 1], '16 16': [2, 1]}, {'42 42': [3, 2], '16 42': [2, 2]}, {'42 42': [3, 3], '42 16': [3, 2], '16 42': [2, 3], '16 16': [2, 2]}, {'42 42': [3, 4], '42 16': [3, 3], '16 42': [2, 4], '16 16': [2, 3]}, {'42 42': [3, 5], '42 16': [3, 4], '16 42': [2, 5], '16 16': [2, 4]}, {'42 42': [3, 6], '16 42': [2, 6]}, {'42 42': [4, 0], '42 16': [4, 0], '16 42': [3, 0], '16 16': [3, 0]}, {'42 42': [4, 1], '42 16': [4, 1], '16 42': [3, 1], '16 16': [3, 1]}, {'42 42': [4, 2], '16 42': [3, 2]}, {'42 42': [4, 3], '42 16': [4, 2], '16 42': [3, 3], '16 16': [3, 2]}, {'42 42': [4, 4], '42 16': [4, 3], '16 42': [3, 4], '16 16': [3, 3]}, {'42 42': [4, 5], '42 16': [4, 4], '16 42': [3, 5], '16 16': [3, 4]}, {'42 42': [4, 6], '16 42': [3, 6]}, {'42 42': [5, 0], '42 16': [5, 0], '16 42': [4, 0], '16 16': [4, 0]}, {'42 42': [5, 1], '42 16': [5, 1], '16 42': [4, 1], '16 16': [4, 1]}, {'42 42': [5, 2], '16 42': [4, 2]}, {'42 42': [5, 3], '42 16': [5, 2], '16 42': [4, 3], '16 16': [4, 2]}, {'42 42': [5, 4], '42 16': [5, 3], '16 42': [4, 4], '16 16': [4, 3]}, {'42 42': [5, 5], '42 16': [5, 4], '16 42': [4, 5], '16 16': [4, 4]}, {'42 42': [5, 6], '16 42': [4, 6]}, {'42 42': [6, 0], '42 16': [6, 0]}, {'42 42': [6, 1], '42 16': [6, 1]}, {'42 42': [6, 2]}, {'42 42': [6, 3], '42 16': [6, 2]}, {'42 42': [6, 4], '42 16': [6, 3]}, {'42 42': [6, 5], '42 16': [6, 4]}, {'42 42': [6, 6]}]
out_js_list, out_slices = dataset.set_mask(targets)
# net_out = '4x1o+4x2o+4x3o + 4x0e+4x1e+4x2e'
net_out = '1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e+1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e+1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e+1x1o+1x1o+1x1o+1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x1o+1x2o+1x3o+1x1o+1x2o+1x3o+1x1o+1x1o+1x1o+1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x1o+1x2o+1x3o+1x1o+1x2o+1x3o+1x2e+1x2e+1x2e+1x1o+1x2o+1x3o+1x1o+1x2o+1x3o+1x0e+1x1e+1x2e+1x3e+1x4e+1x0e+1x1e+1x2e+1x3e+1x4e+1x2e+1x2e+1x2e+1x1o+1x2o+1x3o+1x1o+1x2o+1x3o+1x0e+1x1e+1x2e+1x3e+1x4e+1x0e+1x1e+1x2e+1x3e+1x4e'

if test_net:
    num_species = len(dataset.info["index_to_Z"])
    net = Net(
        num_species=num_species,
        irreps_embed_node="32x0e",
        irreps_edge_init="64x0e",
        irreps_sh='1x0e + 1x1o + 1x2e + 1x3o + 1x4e + 1x5o + 1x6e',
        irreps_mid_node='17x0e+20x1o+8x1e+8x2o+20x2e+8x3o+4x3e+4x4e',#'16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+4x4e+4x4o+4x5e+4x5o+4x6e+4x6o',
        irreps_post_node='17x0e+20x1o+8x1e+8x2o+20x2e+8x3o+4x3e+4x4e',
        irreps_out_node="1x0e",
        irreps_mid_edge='17x0e+20x1o+8x1e+8x2o+20x2e+8x3o+4x3e+4x4e',
        irreps_post_edge='17x0e+20x1o+8x1e+8x2o+20x2e+8x3o+4x3e+4x4e',
        irreps_out_edge=net_out,
        num_block=3,
        r_max=7.2,
        use_sc=True,
        if_sort_irreps=False,
        only_ij=True
    )
    # net = Net(
    #     num_species=num_species,
    #     irreps_embed_node="64x0e",
    #     irreps_edge_init="64x0e",
    #     irreps_sh='1x0e + 1x1o + 1x2e + 1x3o + 1x4e',
    #     irreps_mid_node='64x0e + 64x0o + 32x1e + 32x1o + 16x2e + 16x2o + 8x3e + 8x3o + 4x4e + 4x4o',
    #     irreps_post_node="64x0e + 64x0o + 32x1e + 32x1o + 16x2e + 16x2o + 8x3e + 8x3o + 4x4e + 4x4o",
    #     irreps_out_node="1x0e",
    #     irreps_mid_edge="64x0e + 64x0o + 32x1e + 32x1o + 16x2e + 16x2o + 8x3e + 8x3o + 4x4e + 4x4o",
    #     irreps_post_edge="64x0e + 64x0o + 32x1e + 32x1o + 16x2e + 16x2o + 8x3e + 8x3o + 4x4e + 4x4o",
    #     irreps_out_edge="4x1o+4x2e",
    #     num_block=5,
    #     use_sc=False,
    #     r_max=7.4,
    # )
    if state_dict_dir:
        state_dict = torch.load(state_dict_dir)['state_dict']
        net.load_state_dict(state_dict)
    net.to('cpu')
    net.eval()

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("The model you built has: %d parameters" % params)
    # print(net)

# model_parameters = filter(lambda p: p.requires_grad, net.parameters())
# optimizer = optim.Adam(model_parameters, lr=0.005, betas=(0.9, 0.999))
# criterion = nn.MSELoss()

# rotation matrix, applied on the right of the rotated vector
alpha, beta = 0.1, -0.4
c, s = np.cos(alpha), np.sin(alpha)
rotation_matrix = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
c, s = np.cos(beta), np.sin(beta)
rotation_matrix = torch.tensor(rotation_matrix @ np.array(((c, 0, -s), (0, 1, 0), (s, 0, c))),
                                dtype=torch.get_default_dtype())
rotation_matrix_inv = torch.inverse(rotation_matrix)

rotate_kernel = Rotate(torch.get_default_dtype())
assemble_kernel = assemble_H(dataset.info['orbital_types'], dataset.info['index_to_Z'], targets)

# data loader
loader = DataLoader(dataset=dataset, 
                    batch_size=3, 
                    sampler=SubsetRandomSampler(list(range(3))),
                    collate_fn=Collater(edge_Aij),
                    )

    
for batch in loader:
    
    original = batch.stru_id.index('openmx_test')
    rotate = batch.stru_id.index('openmx_test_rotate')
    inversion = batch.stru_id.index('openmx_test_inversion')
    
    # print(batch.stru_id)
    # print(batch.__slices__)
    # print(batch.x)
    # print(batch.label.shape)
    
    label = assemble_kernel.get_H(batch.x, batch.edge_index, batch.label)
    
    if test_net:
        # get network output
        _, output_edge = net(batch)
        construct_kernel = e3TensorDecomp(net_out, out_js_list, default_dtype_torch=torch.get_default_dtype())
        H_pred = construct_kernel.get_H(output_edge)
        H_pred = assemble_kernel.get_H(batch.x, batch.edge_index, H_pred)
    
    # select an edge in original that is not too long so that the H matrix will not be too small
    index_in_original = batch.__slices__['edge_attr'][original] + 4
    # lengths = batch.edge_attr[batch.__slices__['edge_attr'][original] : batch.__slices__['edge_attr'][original + 1]][:,0]
    edge_coord = batch.edge_attr[index_in_original].type(torch.get_default_dtype())
    
    
    print('\n====compare inversion====')
    # find the edge in inversion that corresponds to index_in_original
    edge_coord_inverse = torch.cat([edge_coord[0].unsqueeze(0), -edge_coord[1:4]])
    found_indices = torch.where(torch.all(torch.abs(batch.edge_attr - edge_coord_inverse) < 1e-7, dim=-1))
    assert len(found_indices) == 1
    found_indices = found_indices[0]
    index_in_inversion = found_indices[torch.where(
        (batch.__slices__['edge_attr'][inversion] <= found_indices)
        & (found_indices <= batch.__slices__['edge_attr'][inversion + 1])
    )].item()
    print(index_in_inversion)
    # index_ij_ji = found_indices[torch.where(
    #     (batch.__slices__['edge_attr'][original] <= found_indices)
    #     & (found_indices <= batch.__slices__['edge_attr'][original + 1])
    # )].item()
    # print(index_ij_ji)
    
    print('\n---test dataset inversion---')
    # print(label[index_in_original])
    # print(label[index_in_inversion])
    H_original = label[index_in_original]
    H_inv_back = rotate_kernel.rotate_openmx_H_full(label[index_in_inversion],
                                                    - torch.eye(3, dtype=torch.get_default_dtype()),
                                                    dataset.info['orbital_types'][batch.x[batch.edge_index[0, index_in_inversion]]],
                                                    dataset.info['orbital_types'][batch.x[batch.edge_index[1, index_in_inversion]]])
    if print_H_block:
        # print(H_original)
        # print(H_inv_back)
        print(H_original - H_inv_back)
    assert torch.allclose(H_original, H_inv_back)
    print('ok')
    # print(batch.label[index_ij_ji])
    
    if test_net:
        print('\n---test net inversion---')
        H_original = H_pred[index_in_original]
        H_inv_back = rotate_kernel.rotate_openmx_H_full(H_pred[index_in_inversion],
                                                        - torch.eye(3, dtype=torch.get_default_dtype()),
                                                        dataset.info['orbital_types'][batch.x[batch.edge_index[0, index_in_inversion]]],
                                                        dataset.info['orbital_types'][batch.x[batch.edge_index[1, index_in_inversion]]])
        if print_H_block:
            # print(H_original)
            # print(H_inv_back)
            print(H_original - H_inv_back)
        assert torch.allclose(H_original, H_inv_back)
        print('ok')
        # print(H_pred[index_ij_ji])
    
    
    print('\n====compare rotation====')
    # find the edge in rotate that corresponds to index_in_original
    edge_coord_rotate = torch.cat([edge_coord[0].unsqueeze(0), 
                                   edge_coord[1:4] @ rotation_matrix])
    found_indices = torch.where(torch.all(torch.abs(batch.edge_attr - edge_coord_rotate[None, :]) < 1e-6, 
                                          dim=-1))
    assert len(found_indices) == 1
    found_indices = found_indices[0]
    index_in_rotate = found_indices[torch.where(
        (batch.__slices__['edge_attr'][rotate] <= found_indices)
        & (found_indices <= batch.__slices__['edge_attr'][rotate + 1])
    )].item()
    print(index_in_rotate)
    
    print('\n---test dataset rotation---')
    H_original = label[index_in_original]
    H_rotate_back = rotate_kernel.rotate_openmx_H_full(label[index_in_rotate], 
                                                       rotation_matrix_inv,
                                                       dataset.info['orbital_types'][batch.x[batch.edge_index[0, index_in_rotate]]],
                                                       dataset.info['orbital_types'][batch.x[batch.edge_index[1, index_in_rotate]]])
    if print_H_block:
        # print(H_original)
        # print(H_rotate_back)
        print(H_original - H_rotate_back)
    assert torch.allclose(H_rotate_back, H_original)
    print('ok')
    
    if test_net:
        print('\n---test net rotation---')
        H_original = H_pred[index_in_original]
        H_rotate_back = rotate_kernel.rotate_openmx_H_full(H_pred[index_in_rotate], 
                                                            rotation_matrix_inv,
                                                            dataset.info['orbital_types'][batch.x[batch.edge_index[0, index_in_rotate]]],
                                                            dataset.info['orbital_types'][batch.x[batch.edge_index[1, index_in_rotate]]])
        if print_H_block:
            # print(H_original)
            # print(H_rotate_back)
            print(H_original - H_rotate_back)
        assert torch.allclose(H_rotate_back, H_original)
        print('ok')