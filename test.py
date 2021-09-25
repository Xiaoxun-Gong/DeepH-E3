import os
import random
import time

import torch
from e3nn.o3 import wigner_D, matrix_x, inverse_angles, Irrep, Irreps
import numpy as np
import argparse
import h5py
from torch import optim, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import SubsetRandomSampler, DataLoader

from e3Aij import AijData, Collater, Net, LossRecord


parser = argparse.ArgumentParser(description='Predict Hamiltonian')
parser.add_argument('--input_dir', type=str, default='processed/openmx_test', help='')
parser.add_argument('--input_rotate_dir', type=str, default='processed/openmx_test_rotate', help='')
parser.add_argument('--input_inversion_dir', type=str, default='processed/openmx_test_inversion', help='')
args = parser.parse_args()


class Rotate:
    def __init__(self, default_dtype_torch):
        sqrt_2 = 1.4142135623730951
        # openmx的实球谐函数基组变复球谐函数
        self.Us_openmx = {
            0: torch.tensor([1], dtype=torch.cfloat),
            1: torch.tensor([[-1 / sqrt_2, 1j / sqrt_2, 0], [0, 0, 1], [1 / sqrt_2, 1j / sqrt_2, 0]], dtype=torch.cfloat),
            2: torch.tensor([[0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0]], dtype=torch.cfloat),
            3: torch.tensor([[0, 0, 0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [0, 0, 0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, -1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2, 0, 0],
                             [0, 0, 0, 0, 0, 1 / sqrt_2, 1j / sqrt_2]], dtype=torch.cfloat),
        }
        # openmx的实球谐函数基组变wiki的实球谐函数 https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        self.Us_openmx2wiki = {
            0: torch.eye(1, dtype=default_dtype_torch),
            1: torch.eye(3, dtype=default_dtype_torch)[[1, 2, 0]],
            2: torch.eye(5, dtype=default_dtype_torch)[[2, 4, 0, 3, 1]],
            3: torch.eye(7, dtype=default_dtype_torch)[[6, 4, 2, 0, 1, 3, 5]]
        }
        self.Us_wiki2openmx = {k: v.T for k, v in self.Us_openmx2wiki.items()}

    def rotate_e3nn_v(self, v, R, l, order_xyz=True):
        if order_xyz:
            # R是(x, y, z)顺序
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn是(y, z, x)顺序
        else:
            # R是(y, z, x)顺序
            R_e3nn = R
        return v @ Irrep(l, 1).D_from_matrix(R_e3nn)

    def rotate_openmx_H(self, H, R, l_left, l_right, order_xyz=True):
        if order_xyz:
            # R是(x, y, z)顺序
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn是(y, z, x)顺序
        else:
            # R是(y, z, x)顺序
            R_e3nn = R
        return self.Us_openmx2wiki[l_left].T @ Irrep(l_left, 1).D_from_matrix(R_e3nn).transpose(-1, -2) @ self.Us_openmx2wiki[l_left] @ H \
               @ self.Us_openmx2wiki[l_right].T @ Irrep(l_right, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_right]

    def wiki2openmx_H(self, H, l_left, l_right):
        return self.Us_openmx2wiki[l_left].T @ H @ self.Us_openmx2wiki[l_right]

    def rotate_matrix_convert(self, R):
        # (x, y, z)顺序排列的旋转矩阵转换为(y, z, x)顺序(see e3nn.o3.spherical_harmonics() and https://docs.e3nn.org/en/stable/guide/change_of_basis.html)
        return torch.eye(3)[[1, 2, 0]] @ R @ torch.eye(3)[[1, 2, 0]].T


def test_rotate_Hamiltonian():
    # e3nn 中喂给 D_from_matrix 的旋转矩阵 R 作用 3 维空间矢量 r 的方式是 r @ R, 其中 r 和 R 按照 (y, z, x) 而不是 (x, y, z)顺序排列
    alpha, beta = 0.1, -0.4
    c, s = np.cos(alpha), np.sin(alpha)
    rotation_matrix = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    c, s = np.cos(beta), np.sin(beta)
    rotation_matrix = rotation_matrix @ np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)))
    rotation_matrix_inv = np.linalg.inv(rotation_matrix)


    print('[compare] lattice')
    lattice = np.loadtxt(os.path.join(args.input_dir, 'lat.dat')).T
    lattice_rotate = np.loadtxt(os.path.join(args.input_rotate_dir, 'lat.dat')).T
    assert np.allclose(lattice, lattice_rotate @ rotation_matrix_inv)
    assert np.allclose(lattice_rotate, lattice @ rotation_matrix)
    print('ok')

    print('\n\n[compare] cart_coords')
    cart_coords = np.loadtxt(os.path.join(args.input_dir, 'site_positions.dat')).T
    cart_coords_rotate = np.loadtxt(os.path.join(args.input_rotate_dir, 'site_positions.dat')).T
    assert np.allclose(cart_coords, cart_coords_rotate @ rotation_matrix_inv)
    assert np.allclose(cart_coords_rotate, cart_coords @ rotation_matrix)
    print('ok')

    fid = h5py.File(os.path.join(args.input_dir, 'hamiltonians.h5'), 'r')
    fid_rotate = h5py.File(os.path.join(args.input_rotate_dir, 'hamiltonians.h5'), 'r')
    fid_inversion = h5py.File(os.path.join(args.input_inversion_dir, 'hamiltonians.h5'), 'r')

    # torch_dtype = torch.cfloat
    torch_dtype = torch.float32
    mat = torch.tensor(fid['[1, 0, 0, 1, 3]'], dtype=torch_dtype)
    mat_rotate = torch.tensor(fid_rotate['[1, 0, 0, 1, 3]'], dtype=torch_dtype)
    mat_inversion = torch.tensor(fid_inversion['[1, 0, 0, 1, 3]'], dtype=torch_dtype)
    rotation_matrix = torch.tensor(rotation_matrix, dtype=torch_dtype)

    rotate_kernel = Rotate(torch_dtype)

    print('\n\n[compare] s-s orbital Hamiltonian')
    l_left, l_right = 0, 0
    mat_block = mat[0:1, 1:2]
    mat_block_rotate = mat_rotate[0:1, 1:2]
    mat_block_inversion = mat_inversion[0:1, 1:2]
    assert np.allclose(mat_block_rotate, rotate_kernel.rotate_openmx_H(mat_block, rotation_matrix, l_left, l_right))
    assert np.allclose(mat_block, mat_block_inversion * ((-1) ** (l_left + l_right)))
    print('ok')

    print('\n\n[compare] p-p orbital Hamiltonian')
    l_left, l_right = 1, 1
    mat_block = mat[3:6, 2:5]
    mat_block_rotate = mat_rotate[3:6, 2:5]
    mat_block_inversion = mat_inversion[3:6, 2:5]
    assert np.allclose(mat_block_rotate, rotate_kernel.rotate_openmx_H(mat_block, rotation_matrix, l_left, l_right))
    assert np.allclose(mat_block, mat_block_inversion * ((-1) ** (l_left + l_right)))
    print('ok')

    print('\n\n[compare] p-p orbital Hamiltonian')
    l_left, l_right = 1, 1
    mat_block = mat[6:9, 2:5]
    mat_block_rotate = mat_rotate[6:9, 2:5]
    mat_block_inversion = mat_inversion[6:9, 2:5]
    assert np.allclose(mat_block_rotate, rotate_kernel.rotate_openmx_H(mat_block, rotation_matrix, l_left, l_right))
    assert np.allclose(mat_block, mat_block_inversion * ((-1) ** (l_left + l_right)))
    print('ok')

    print('\n\n[compare] p-d orbital Hamiltonian')
    l_left, l_right = 1, 2
    mat_block = mat[3:6, 8:13]
    mat_block_rotate = mat_rotate[3:6, 8:13]
    mat_block_inversion = mat_inversion[3:6, 8:13]
    assert np.allclose(mat_block_rotate, rotate_kernel.rotate_openmx_H(mat_block, rotation_matrix, l_left, l_right))
    assert np.allclose(mat_block, mat_block_inversion * ((-1) ** (l_left + l_right)))
    print('ok')

    print('\n\n[compare] d-s orbital Hamiltonian')
    l_left, l_right = 2, 0
    mat_block = mat[14:19, 0:1]
    mat_block_rotate = mat_rotate[14:19, 0:1]
    mat_block_inversion = mat_inversion[14:19, 0:1]
    assert np.allclose(mat_block_rotate, rotate_kernel.rotate_openmx_H(mat_block, rotation_matrix, l_left, l_right))
    assert np.allclose(mat_block, mat_block_inversion * ((-1) ** (l_left + l_right)))
    print('ok')

    print('\n\n[compare] d-d orbital Hamiltonian')
    l_left, l_right = 2, 2
    mat_block = mat[9:14, 8:13]
    mat_block_rotate = mat_rotate[9:14, 8:13]
    mat_block_inversion = mat_inversion[9:14, 8:13]
    assert np.allclose(mat_block_rotate, rotate_kernel.rotate_openmx_H(mat_block, rotation_matrix, l_left, l_right))
    assert np.allclose(mat_block, mat_block_inversion * ((-1) ** (l_left + l_right)))
    print('ok')


def test_nn():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch_dtype = torch.float32
    torch.set_default_dtype(torch_dtype)
    rotate_kernel = Rotate(torch_dtype)

    # e3nn 中喂给 D_from_matrix 的旋转矩阵 R 作用 3 维空间矢量 r 的方式是 r @ R, 其中 r 和 R 按照 (y, z, x) 而不是 (x, y, z)顺序排列
    alpha, beta = 0.1, -0.4
    c, s = np.cos(alpha), np.sin(alpha)
    rotation_matrix = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    c, s = np.cos(beta), np.sin(beta)
    rotation_matrix = torch.tensor(rotation_matrix @ np.array(((c, 0, -s), (0, 1, 0), (s, 0, c))), dtype=torch.get_default_dtype())
    rotation_matrix_inv = torch.inverse(rotation_matrix)


    edge_Aij = True

    dataset = AijData(
        raw_data_dir="/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0923/processed",
        graph_dir="/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0923/graph_data",
        target="hamiltonian",
        dataset_name="test_MoS2",
        multiprocessing=False,
        # radius=5.0,
        radius=7.4,
        max_num_nbr=0,
        edge_Aij=edge_Aij,
        default_dtype_torch=torch.get_default_dtype()
    )
    num_species = len(dataset.info["index_to_Z"])

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)
    train_loader = DataLoader(dataset, batch_size=3,
                              shuffle=False, sampler=sampler,
                              collate_fn=Collater(edge_Aij))
    net = Net(
        num_species=num_species,
        irreps_embed_node="16x0e",
        irreps_sh='1x0e+1x1o+1x2e+1x3o+1x4e',
        irreps_mid_node='10x0e+8x1o',
        irreps_post_node="6x0e+6x1o+6x2e",
        irreps_out_node="2x1o+1x2e+1x0e",
        irreps_mid_edge="12x0e+7x1o",
        irreps_post_edge="5x0e+4x1o+3x2e",
        irreps_out_edge="2x0e+1x1o+1x2e",
        num_block=3,
        use_sc=True,
        r_max = 7.4,
    )

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("The model you built has: %d parameters" % params)
    print(net)

    for batch in train_loader:
        assert torch.allclose(batch.edge_attr[:, 0], torch.norm(batch.edge_attr[:, 1:], dim=-1))

        slice_edge_attr = batch._slice_dict['edge_index']
        slice_x = batch._slice_dict['x']

        output, output_edge = net(batch)

        original = batch.stru_id.index('openmx_test')
        rotate = batch.stru_id.index('openmx_test_rotate')
        inversion = batch.stru_id.index('openmx_test_inversion')

        print(batch)
        print(batch.stru_id)

        print(batch.pos[original * 3 : (original + 1) * 3])
        print(batch.pos[inversion * 3 : (inversion + 1) * 3])
        print(batch.pos[rotate * 3 : (rotate + 1) * 3])
        print(batch.pos[original * 3 : (original + 1) * 3] @ rotation_matrix)

        print('===========')
        print(output.shape)

        print(output[original * 3, 0:3])
        print(output[inversion * 3, 0:3])
        print(output[rotate * 3, 0:3])
        print(rotate_kernel.rotate_e3nn_v(output[original * 3, 0:3], rotation_matrix, 1))

        print('===========')

        print(output[original * 3, 3:6])
        print(output[inversion * 3, 3:6])
        print(output[rotate * 3, 3:6])
        print(rotate_kernel.rotate_e3nn_v(output[original * 3, 3:6], rotation_matrix, 1))

        print('===========')

        print(output[original * 3, 6:11])
        print(output[inversion * 3, 6:11])
        print(output[rotate * 3, 6:11])
        print(rotate_kernel.rotate_e3nn_v(output[original * 3, 6:11], rotation_matrix, 2))

        print('===========')

        print(output[original * 3, 11:12])
        print(output[inversion * 3, 11:12])
        print(output[rotate * 3, 11:12])
        print(rotate_kernel.rotate_e3nn_v(output[original * 3, 11:12], rotation_matrix, 0))

        print('===========')

        edge_index_dict = {}

        get_range = lambda x: range(slice_edge_attr[x], slice_edge_attr[x+1])
        for data_name in [original, inversion, rotate]:
            inv_lattice = torch.inverse(batch.lattice[data_name])
            for edge_index in get_range(data_name):
                i, j = batch.edge_index[:, edge_index]
                cart_coords_i = batch.pos[i]
                cart_coords_j_unit_cell = batch.pos[j]
                cart_coords_j = cart_coords_i + batch.edge_attr[edge_index, 1:4]
                R = (cart_coords_j - cart_coords_j_unit_cell) @ inv_lattice

                # (*R, i, j), i and j is 0-based index
                key = (*torch.round(R).int().tolist(),
                       (i - slice_x[data_name]).item() + 1,
                       (j - slice_x[data_name]).item() + 1)
                if key == (1, 0, 0, 1, 3):
                    print(f"found edge in {batch.stru_id[data_name]}")
                    edge_index_dict[data_name] = edge_index

        print(output_edge.shape)
        print(batch.edge_attr[edge_index_dict[original]])
        print(batch.edge_attr[edge_index_dict[inversion]])
        print(batch.edge_attr[edge_index_dict[rotate]])

        print('===========')

        print(output_edge[edge_index_dict[original], 0:1])
        print(output_edge[edge_index_dict[inversion], 0:1])
        print(output_edge[edge_index_dict[rotate], 0:1])
        print(rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 0:1], rotation_matrix, 0))

        print('===========')

        print(output_edge[edge_index_dict[original], 2:5])
        print(output_edge[edge_index_dict[inversion], 2:5])
        print(output_edge[edge_index_dict[rotate], 2:5])
        print(rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 2:5], rotation_matrix, 1))

        print('===========')

        print(output_edge[edge_index_dict[original], 5:10])
        print(output_edge[edge_index_dict[inversion], 5:10])
        print(output_edge[edge_index_dict[rotate], 5:10])
        print(rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 5:10], rotation_matrix, 2))


def test_Rij():
    device = torch.device("cpu")
    torch.autograd.set_detect_anomaly(True)
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch_dtype = torch.float64
    torch.set_default_dtype(torch_dtype)
    rotate_kernel = Rotate(torch_dtype)

    # e3nn 中喂给 D_from_matrix 的旋转矩阵 R 作用 3 维空间矢量 r 的方式是 r @ R, 其中 r 和 R 按照 (y, z, x) 而不是 (x, y, z)顺序排列
    alpha, beta = 0.1, -0.4
    c, s = np.cos(alpha), np.sin(alpha)
    rotation_matrix = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    c, s = np.cos(beta), np.sin(beta)
    rotation_matrix = torch.tensor(rotation_matrix @ np.array(((c, 0, -s), (0, 1, 0), (s, 0, c))),
                                   dtype=torch.get_default_dtype())
    rotation_matrix_inv = torch.inverse(rotation_matrix)

    edge_Aij = True

    dataset = AijData(
        raw_data_dir="/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0923/processed",
        graph_dir="/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0923/graph_data",
        target="hamiltonian",
        dataset_name="test_MoS2",
        multiprocessing=False,
        # radius=5.0,
        radius=7.4,
        max_num_nbr=0,
        edge_Aij=edge_Aij,
        default_dtype_torch=torch.get_default_dtype()
    )
    out_js_list = dataset.set_mask()
    num_species = len(dataset.info["index_to_Z"])

    train_loader = DataLoader(dataset, batch_size=1,
                              shuffle=False, sampler=SubsetRandomSampler([1]),
                              collate_fn=Collater(edge_Aij))
    test_loader = DataLoader(dataset, batch_size=1,
                              shuffle=False, sampler=SubsetRandomSampler([0, 2]),
                              collate_fn=Collater(edge_Aij))
    net = Net(
        num_species=num_species,
        irreps_embed_node="16x0e",
        irreps_sh='1x0e+1x1o+1x2e+1x3o+1x4e',
        # irreps_mid_node='16x0e+8x1o',
        # irreps_post_node="16x0e+8x1o",
        irreps_mid_node='64x0o+64x0e+8x1o+8x1e',
        irreps_post_node="64x0o+64x0e+8x1o+8x1e",
        irreps_out_node="1x0e",
        # irreps_mid_edge="12x0e+10x1o",
        # irreps_post_edge="12x0e+10x1o",
        # irreps_out_edge="2x1o+15x0e",
        irreps_mid_edge="80x0o+80x0e+10x1o+10x1e",
        irreps_post_edge="80x0o+80x0e+10x1o+10x1e",
        irreps_out_edge="1x1o+1x1e+15x0e",
        num_block=5,
        use_sc=False,
        r_max=7.4,
    )
    net.to(device)

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("The model you built has: %d parameters" % params)

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(model_parameters, lr=0.005, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    begin_time = time.time()
    net.train()
    for epoch in range(1000):
        learning_rate = optimizer.param_groups[0]['lr']
        for batch in train_loader:
            losses = LossRecord()
            output, output_edge = net(batch.to(device))

            l1 = output_edge[:, 0:3]
            l2 = output_edge[:, 3:6]

            intra_site_index = torch.where(abs(batch.edge_attr[:, 0]) < 1e-5)
            x_hat = batch.edge_attr[:, [2, 3, 1]] / batch.edge_attr[:, 0:1]
            x_hat[intra_site_index] = (l1 / torch.linalg.norm(l1, dim=-1, keepdim=True))[intra_site_index]
            y_hat_unnorm = torch.cross(x_hat, l2, dim=-1)
            y_hat = y_hat_unnorm / torch.linalg.norm(y_hat_unnorm, dim=-1, keepdim=True)
            z_hat = torch.cross(x_hat, y_hat, dim=-1)

            R = torch.stack([x_hat, y_hat, z_hat], dim=-2)

            output_H_prime = output_edge[:, 6:].reshape(-1, 3, 5)
            # output_H = rotate_kernel.rotate_openmx_H(output_H_prime, R, 1, 2, order_xyz=False)
            output_H = output_H_prime

            loss = criterion(output_H, batch.label)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(net.parameters(), 4.2, error_if_nonfinite=True)
            optimizer.step()
            losses.update(loss.item(), batch.num_edges)

            print(output_H[0])
            print(batch.label[0])

        print(f'Epoch #{epoch:01d} \t| '
              f'Learning rate: {learning_rate:0.2e} \t| '
              f'Batch time: {time.time() - begin_time:.2f} \t| '
              f'Train loss: {losses.avg:.8f}'
              )
        begin_time = time.time()


    mzjb

    target_key = (1, 0, 0, 1, 3)
    edge_index_dict = {}
    for batch in train_loader:
        print(batch)
        print(batch.stru_id)
        print(batch.label.shape)

        original = batch.stru_id.index('openmx_test')
        inversion = batch.stru_id.index('openmx_test_inversion')

        slice_edge_attr = batch._slice_dict['edge_index']
        slice_x = batch._slice_dict['x']
        get_range = lambda x: range(slice_edge_attr[x], slice_edge_attr[x + 1])
        for data_name in [original, inversion]:
            inv_lattice = torch.inverse(batch.lattice[data_name])
            for edge_index in get_range(data_name):
                i, j = batch.edge_index[:, edge_index]
                cart_coords_i = batch.pos[i]
                cart_coords_j_unit_cell = batch.pos[j]
                cart_coords_j = cart_coords_i + batch.edge_attr[edge_index, 1:4]
                R = (cart_coords_j - cart_coords_j_unit_cell) @ inv_lattice

                # (*R, i, j), i and j is 0-based index
                key = (*torch.round(R).int().tolist(),
                       (i - slice_x[data_name]).item() + 1,
                       (j - slice_x[data_name]).item() + 1)
                if key == target_key:
                    print(f"found edge in {batch.stru_id[data_name]}")
                    edge_index_dict[data_name] = edge_index

        output, output_edge = net(batch)
        l1 = output_edge[:, 0:3]
        l2 = output_edge[:, 3:6]

        print("***edge_attr***")
        print(batch.edge_attr[edge_index_dict[original]])
        print(batch.edge_attr[edge_index_dict[inversion]])

        print("***label***")
        print(batch.label[edge_index_dict[inversion]])
        print(batch.label[edge_index_dict[original]])
        print(rotate_kernel.rotate_openmx_H(batch.label[edge_index_dict[original]], rotation_matrix, 1, 2))

        intra_site_index = torch.where(abs(batch.edge_attr[:, 0]) < 1e-5)
        x_hat = batch.edge_attr[:, [2, 3, 1]] / batch.edge_attr[:, 0:1]
        x_hat[intra_site_index] = (l1 / torch.linalg.norm(l1, dim=-1, keepdim=True))[intra_site_index]
        y_hat_unnorm = torch.cross(x_hat, l2, dim=-1)
        y_hat = y_hat_unnorm / torch.linalg.norm(y_hat_unnorm, dim=-1, keepdim=True)
        z_hat = torch.cross(x_hat, y_hat, dim=-1)

        R = torch.stack([x_hat, y_hat, z_hat], dim=-2)

        output_H_prime = output_edge[:, 6:].reshape(-1, 3, 5)
        output_H = rotate_kernel.rotate_openmx_H(output_H_prime, R, 1, 2, order_xyz=False)

        print("***output***")
        print(output_edge.shape)
        print(output_edge[edge_index_dict[original], 0:6])
        print(output_edge[edge_index_dict[inversion], 0:6])
        print(rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 0:3], rotation_matrix, 1),
              rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 3:6], rotation_matrix, 1))

        print("***x_hat***")
        print(x_hat[edge_index_dict[original]])
        print(x_hat[edge_index_dict[inversion]])
        print(rotate_kernel.rotate_e3nn_v(x_hat[edge_index_dict[original]], rotation_matrix, 1))

        print("***y_hat***")
        print(y_hat[edge_index_dict[original]])
        print(y_hat[edge_index_dict[inversion]])
        print(rotate_kernel.rotate_e3nn_v(y_hat[edge_index_dict[original]], rotation_matrix, 1))

        print("***z_hat***")
        print(z_hat[edge_index_dict[original]])
        print(z_hat[edge_index_dict[inversion]])
        print(rotate_kernel.rotate_e3nn_v(z_hat[edge_index_dict[original]], rotation_matrix, 1))

        print("***output H***")
        print(output_H_prime[edge_index_dict[original]])
        print(output_H_prime[edge_index_dict[inversion]])
        print(output_H[edge_index_dict[original]])
        print(output_H[edge_index_dict[inversion]])
        print(rotate_kernel.rotate_openmx_H(output_H[edge_index_dict[original]], rotation_matrix, 1, 2))


    for batch in test_loader:
        print(batch.stru_id)

        rotate = batch.stru_id.index('openmx_test_rotate')
        slice_edge_attr = batch._slice_dict['edge_index']
        slice_x = batch._slice_dict['x']
        get_range = lambda x: range(slice_edge_attr[x], slice_edge_attr[x + 1])
        for data_name in [rotate]:
            inv_lattice = torch.inverse(batch.lattice[data_name])
            for edge_index in get_range(data_name):
                i, j = batch.edge_index[:, edge_index]
                cart_coords_i = batch.pos[i]
                cart_coords_j_unit_cell = batch.pos[j]
                cart_coords_j = cart_coords_i + batch.edge_attr[edge_index, 1:4]
                R = (cart_coords_j - cart_coords_j_unit_cell) @ inv_lattice

                # (*R, i, j), i and j is 0-based index
                key = (*torch.round(R).int().tolist(),
                       (i - slice_x[data_name]).item() + 1,
                       (j - slice_x[data_name]).item() + 1)
                if key == target_key:
                    print(f"found edge in {batch.stru_id[data_name]}")
                    edge_index_dict[data_name] = edge_index

        output, output_edge = net(batch)
        l1 = output_edge[:, 0:3]
        l2 = output_edge[:, 3:6]

        print("***edge_attr***")
        print(batch.edge_attr[edge_index_dict[rotate]])

        print("***label***")
        print(batch.label[edge_index_dict[rotate]])

        intra_site_index = torch.where(abs(batch.edge_attr[:, 0]) < 1e-5)
        x_hat = batch.edge_attr[:, [2, 3, 1]] / batch.edge_attr[:, 0:1]
        x_hat[intra_site_index] = (l1 / torch.linalg.norm(l1, dim=-1, keepdim=True))[intra_site_index]
        y_hat_unnorm = torch.cross(x_hat, l2, dim=-1)
        y_hat = y_hat_unnorm / torch.linalg.norm(y_hat_unnorm, dim=-1, keepdim=True)
        z_hat = torch.cross(x_hat, y_hat, dim=-1)

        R = torch.stack([x_hat, y_hat, z_hat], dim=-2)

        output_H_prime = output_edge[:, 6:].reshape(-1, 3, 5)
        output_H = rotate_kernel.rotate_openmx_H(output_H_prime, R, 1, 2, order_xyz=False)

        print("***output***")
        print(output_edge[edge_index_dict[rotate], 0:6])

        print("***x_hat***")
        print(x_hat[edge_index_dict[rotate]])

        print("***y_hat***")
        print(y_hat[edge_index_dict[rotate]])

        print("***z_hat***")
        print(z_hat[edge_index_dict[rotate]])

        print("***output H***")
        print(output_H_prime[edge_index_dict[rotate]])
        print(output_H[edge_index_dict[rotate]])


def test_Hij():
    device = torch.device('cpu')
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch_dtype = torch.float64
    torch.set_default_dtype(torch_dtype)
    rotate_kernel = Rotate(torch_dtype)

    # e3nn 中喂给 D_from_matrix 的旋转矩阵 R 作用 3 维空间矢量 r 的方式是 r @ R, 其中 r 和 R 按照 (y, z, x) 而不是 (x, y, z)顺序排列
    alpha, beta = 0.1, -0.4
    c, s = np.cos(alpha), np.sin(alpha)
    rotation_matrix = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    c, s = np.cos(beta), np.sin(beta)
    rotation_matrix = torch.tensor(rotation_matrix @ np.array(((c, 0, -s), (0, 1, 0), (s, 0, c))),
                                   dtype=torch.get_default_dtype())
    rotation_matrix_inv = torch.inverse(rotation_matrix)

    edge_Aij = True

    dataset = AijData(
        raw_data_dir="/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0923/processed",
        graph_dir="/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0923/graph_data",
        target="hamiltonian",
        dataset_name="test_MoS2_new",
        multiprocessing=False,
        # radius=5.0,
        radius=7.4,
        max_num_nbr=0,
        edge_Aij=edge_Aij,
        default_dtype_torch=torch.get_default_dtype()
    )
    out_js_list = dataset.set_mask()
    num_species = len(dataset.info["index_to_Z"])

    train_loader = DataLoader(dataset, batch_size=2,
                              shuffle=False, sampler=SubsetRandomSampler([1, 2]),
                              collate_fn=Collater(edge_Aij))
    test_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, sampler=SubsetRandomSampler([0]),
                             collate_fn=Collater(edge_Aij))
    
    begin = time.time()
    print('Building model...')
    net = Net(
        num_species=num_species,
        irreps_embed_node="32x0e",
        irreps_edge_init='64x0e',
        irreps_sh='1x0e + 1x1o + 1x2e + 1x3o + 1x4e',
        irreps_mid_node='16x0e + 16x0o + 8x1e + 8x1o',
        irreps_post_node='16x0e + 16x0o + 8x1e + 8x1o + 4x2e + 4x2o',
        irreps_out_node="1x0e",
        irreps_mid_edge='16x0e + 16x0o + 8x1e + 8x1o',
        irreps_post_edge='16x0e + 16x0o + 8x1e + 8x1o + 4x2e + 4x2o',
        irreps_out_edge="4x1o+4x2e",
        num_block=3,
        use_sc=False,
        r_max = 7.4,
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
    print(f'Finished building model, cost {time.time() - begin:.2f} seconds.')

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("The model you built has: %d parameters" % params)
    
    print(net)

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(model_parameters, lr=0.005, betas=(0.9, 0.999))
    criterion = nn.MSELoss()


    begin_time = time.time()
    net.train()
    for epoch in range(10000):
        learning_rate = optimizer.param_groups[0]['lr']
        for batch in train_loader:
            losses = LossRecord()
            output, output_edge = net(batch)

            H_pred = rotate_kernel.wiki2openmx_H(
                output_edge[:, 0:3][:, :, None] * output_edge[:, 12:17][:, None, :]
                + output_edge[:, 3:6][:, :, None] * output_edge[:, 17:22][:, None, :]
                + output_edge[:, 6:9][:, :, None] * output_edge[:, 22:27][:, None, :]
                + output_edge[:, 9:12][:, :, None] * output_edge[:, 27:32][:, None, :],
                1, 2
            )
            # H_pred = output_edge.unsqueeze(-1)

            loss = criterion(H_pred, batch.label)
            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(net.parameters(), 4.2, error_if_nonfinite=True)
            optimizer.step()
            losses.update(loss.item(), batch.num_edges)

        print(f'Epoch #{epoch:01d} \t| '
              f'Learning rate: {learning_rate:0.2e} \t| '
              f'Batch time: {time.time() - begin_time:.2f} \t| '
              f'Train loss: {losses.avg:.8f}'
              )
        begin_time = time.time()

        if epoch == 1000:
            # torch.save(net.state_dict(), '/home/gongxx/projects/DeepH/e3nn_DeepH/test_runs/0927/model1.pkl')
            mzjb




    target_key = (1, 0, 0, 1, 3)
    edge_index_dict = {}
    for batch in train_loader:
        print(batch)
        print(batch.stru_id)
        print(batch.label.shape)

        original = batch.stru_id.index('openmx_test')
        inversion = batch.stru_id.index('openmx_test_inversion')

        slice_edge_attr = batch._slice_dict['edge_index']
        slice_x = batch._slice_dict['x']
        get_range = lambda x: range(slice_edge_attr[x], slice_edge_attr[x + 1])
        for data_name in [original, inversion]:
            inv_lattice = torch.inverse(batch.lattice[data_name])
            for edge_index in get_range(data_name):
                i, j = batch.edge_index[:, edge_index]
                cart_coords_i = batch.pos[i]
                cart_coords_j_unit_cell = batch.pos[j]
                cart_coords_j = cart_coords_i + batch.edge_attr[edge_index, 1:4]
                R = (cart_coords_j - cart_coords_j_unit_cell) @ inv_lattice

                # (*R, i, j), i and j is 0-based index
                key = (*torch.round(R).int().tolist(),
                       (i - slice_x[data_name]).item() + 1,
                       (j - slice_x[data_name]).item() + 1)
                if key == target_key:
                    print(f"found edge in {batch.stru_id[data_name]}")
                    edge_index_dict[data_name] = edge_index

        output, output_edge = net(batch)
        H_pred = rotate_kernel.wiki2openmx_H(
            output_edge[:, 0:3][:, :, None] * output_edge[:, 12:17][:, None, :]
            + output_edge[:, 3:6][:, :, None] * output_edge[:, 17:22][:, None, :]
            + output_edge[:, 6:9][:, :, None] * output_edge[:, 22:27][:, None, :]
            + output_edge[:, 9:12][:, :, None] * output_edge[:, 27:32][:, None, :],
            1, 2
        )

        print("***edge_attr***")
        print(batch.edge_attr[edge_index_dict[original]])
        print(batch.edge_attr[edge_index_dict[inversion]])

        print("***label***")
        print(batch.label[edge_index_dict[inversion]])
        print(batch.label[edge_index_dict[original]])
        print(rotate_kernel.rotate_openmx_H(batch.label[edge_index_dict[original]], rotation_matrix, 1, 2))

        print("***output***")
        print(output_edge.shape)
        print(output_edge[edge_index_dict[original], 0:32])
        print(output_edge[edge_index_dict[inversion], 0:32])
        print(
            rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 0:3], rotation_matrix, 1),
            rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 3:6], rotation_matrix, 1),
            rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 6:9], rotation_matrix, 1),
            rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 9:12], rotation_matrix, 1),
            rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 12:17], rotation_matrix, 2),
            rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 17:22], rotation_matrix, 2),
            rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 22:27], rotation_matrix, 2),
            rotate_kernel.rotate_e3nn_v(output_edge[edge_index_dict[original], 27:32], rotation_matrix, 2),
        )
        print("***output H***")
        print(H_pred[edge_index_dict[original]])
        print(H_pred[edge_index_dict[inversion]])
        print(rotate_kernel.rotate_openmx_H(H_pred[edge_index_dict[original]], rotation_matrix, 1, 2))


    for batch in test_loader:
        print(batch.stru_id)

        rotate = batch.stru_id.index('openmx_test_rotate')
        slice_edge_attr = batch._slice_dict['edge_index']
        slice_x = batch._slice_dict['x']
        get_range = lambda x: range(slice_edge_attr[x], slice_edge_attr[x + 1])
        for data_name in [rotate]:
            inv_lattice = torch.inverse(batch.lattice[data_name])
            for edge_index in get_range(data_name):
                i, j = batch.edge_index[:, edge_index]
                cart_coords_i = batch.pos[i]
                cart_coords_j_unit_cell = batch.pos[j]
                cart_coords_j = cart_coords_i + batch.edge_attr[edge_index, 1:4]
                R = (cart_coords_j - cart_coords_j_unit_cell) @ inv_lattice

                # (*R, i, j), i and j is 0-based index
                key = (*torch.round(R).int().tolist(),
                       (i - slice_x[data_name]).item() + 1,
                       (j - slice_x[data_name]).item() + 1)
                if key == target_key:
                    print(f"found edge in {batch.stru_id[data_name]}")
                    edge_index_dict[data_name] = edge_index

        output, output_edge = net(batch)
        H_pred = rotate_kernel.wiki2openmx_H(
            output_edge[:, 0:3][:, :, None] * output_edge[:, 12:17][:, None, :]
            + output_edge[:, 3:6][:, :, None] * output_edge[:, 17:22][:, None, :]
            + output_edge[:, 6:9][:, :, None] * output_edge[:, 22:27][:, None, :]
            + output_edge[:, 9:12][:, :, None] * output_edge[:, 27:32][:, None, :],
            1, 2
        )

        print("***edge_attr***")
        print(batch.edge_attr[edge_index_dict[rotate]])

        print("***label***")
        print(batch.label[edge_index_dict[rotate]])

        print("***output***")
        print(output_edge[edge_index_dict[rotate], 0:32])

        print("***output H***")
        print(H_pred[edge_index_dict[rotate]])


if __name__ == '__main__':
    test_Hij()
    # test_Rij()
    # test_nn()
    # test_rotate_Hamiltonian()
