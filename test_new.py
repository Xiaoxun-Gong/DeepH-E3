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

from e3Aij import AijData, Collater, Net, LossRecord, Rotate
from e3Aij.utils import construct_H



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
        dataset_name="test_MoS2_new1",
        multiprocessing=False,
        # radius=5.0,
        radius=7.4,
        max_num_nbr=0,
        edge_Aij=edge_Aij,
        default_dtype_torch=torch.get_default_dtype()
    )
    #! out_js_list, net_out_irreps, save_dir
    out_js_list = dataset.set_mask([{"42 42": [5, 5], "42 16": [5, 4], "16 42": [4, 5], "16 16": [4, 4]}])
    # out_js_list = dataset.set_mask([{"42 42": [3, 5], "42 16": [3, 4], "16 42": [2, 5], "16 16": [2, 4]}])
    # out_js_list = dataset.set_mask([{"42 42": [0, 0], "42 16": [0, 0], "16 42": [0, 0], "16 16": [0, 0]}])
    net_out_irreps = '16x2e'
    num_species = len(dataset.info["index_to_Z"])

    train_loader = DataLoader(dataset, batch_size=2,
                              shuffle=False, sampler=SubsetRandomSampler([1, 2]),
                              collate_fn=Collater(edge_Aij))
    # test_loader = DataLoader(dataset, batch_size=1,
    #                          shuffle=False, sampler=SubsetRandomSampler([0]),
    #                          collate_fn=Collater(edge_Aij))
    
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
        irreps_out_edge=net_out_irreps,
        num_block=3,
        use_sc=False,
        r_max = 7.4,
    )
    net.to(device)
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
    
    # print(net)

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(model_parameters, lr=0.005, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    construct_kernel = construct_H(net_out_irreps, *out_js_list[0])

    begin_time = time.time()
    net.train()
    for epoch in range(10000):
        learning_rate = optimizer.param_groups[0]['lr']
        for batch in train_loader:
            losses = LossRecord()
            output, output_edge = net(batch.to(device))
            
            H_pred = construct_kernel.get_H(output_edge)

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
            torch.save(net.state_dict(), '/home/gongxx/projects/DeepH/e3nn_DeepH/test_runs/0929_l=2/model1.pkl')
            exit()



if __name__ == '__main__':
    test_Hij()
    # test_Rij()
    # test_nn()
    # test_rotate_Hamiltonian()
