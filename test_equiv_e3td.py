from torch import nn, optim
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from e3Aij import AijData, Net, Collater, Rotate
from e3Aij.utils import construct_H, e3TensorDecomp

torch_dtype = torch.float32
torch.set_default_dtype(torch_dtype)
test_net = True

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
l1, l2 = out_js_list[0]
net_out = '4x0e+4x1e+4x2e'

if test_net:
    num_species = len(dataset.info["index_to_Z"])
    net = Net(
        num_species=num_species,
        irreps_embed_node="32x0e",
        irreps_edge_init="64x0e",
        irreps_sh='1x0e + 1x1o + 1x2e + 1x3o + 1x4e + 1x5o + 1x6e',
        irreps_mid_node='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o',#'16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+4x4e+4x4o+4x5e+4x5o+4x6e+4x6o',
        irreps_post_node='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o',
        irreps_out_node="1x0e",
        irreps_mid_edge='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o',
        irreps_post_edge='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o',
        irreps_out_edge=net_out,
        num_block=3,
        use_sc=False,
        r_max = 7.4,
        if_sort_irreps=False
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
    net.to('cpu')

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
    
    print(batch.stru_id)
    print(batch.__slices__)
    print(batch.x)
    print(batch.label.shape)
    
    if test_net:
        # get network output
        _, output_edge = net(batch)
        # construct_kernel = construct_H('8x1o+8x1o', *out_js_list[0])
        # H_pred = construct_kernel.get_H(output_edge)
        construct_kernel = e3TensorDecomp(net_out, *out_js_list[0], default_dtype_torch=torch.get_default_dtype())
        H_pred = construct_kernel.get_H(output_edge)
        # H_pred = rotate_kernel.wiki2openmx_H(H_pred, *out_js_list[0])
    
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
    print(batch.label[index_in_original])
    print(batch.label[index_in_inversion])
    assert torch.allclose(batch.label[index_in_original], 
                          (-1) ** l1 * (-1) ** l2 * batch.label[index_in_inversion])
    print('ok')
    # print(batch.label[index_ij_ji])
    
    if test_net:
        print('\n---test net inversion---')
        print(H_pred[index_in_original])
        print(H_pred[index_in_inversion])
        assert torch.allclose(H_pred[index_in_original],
                              (-1) ** l1 * (-1) ** l2 * H_pred[index_in_inversion])
        print('ok')
        # print(H_pred[index_ij_ji])
    
    
    print('\n====compare rotation====')
    # find the edge in rotate that corresponds to index_in_original
    edge_coord_rotate = torch.cat([edge_coord[0].unsqueeze(0), 
                                   edge_coord[1:4] @ rotation_matrix])
    found_indices = torch.where(torch.all(torch.abs(batch.edge_attr - edge_coord_rotate[None, :]) < 1e-7, 
                                          dim=-1))
    assert len(found_indices) == 1
    found_indices = found_indices[0]
    index_in_rotate = found_indices[torch.where(
        (batch.__slices__['edge_attr'][rotate] <= found_indices)
        & (found_indices <= batch.__slices__['edge_attr'][rotate + 1])
    )].item()
    print(index_in_rotate)
    
    print('\n---test dataset rotation---')
    H_original = batch.label[index_in_original]
    H_rotate_back = rotate_kernel.rotate_openmx_H(batch.label[index_in_rotate], rotation_matrix_inv, l1, l2)
    print(H_original)
    print(H_rotate_back)
    assert torch.allclose(H_rotate_back, H_original)
    print('ok')
    
    if test_net:
        print('\n---test net rotation---')
        H_original = H_pred[index_in_original]
        H_rotate_back = rotate_kernel.rotate_openmx_H(H_pred[index_in_rotate], rotation_matrix_inv, l1, l2)
        print(H_original)
        print(H_rotate_back)
        assert torch.allclose(H_rotate_back, H_original)
        print('ok')