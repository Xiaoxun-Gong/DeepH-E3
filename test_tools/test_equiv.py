import h5py
import numpy as np
from e3nn.o3 import Irreps, angles_to_matrix
import torch

import sys, os
sys.path.append('/home/gongxx/projects/DeepH/e3nn_DeepH/DeepH-E3')
from deephe3 import Rotate

# these need to be changed between different runs
ori = h5py.File('/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/1114_3xbi2se3/test_runs/2022-02-07_14-45-36_/test_out/original/hamiltonians_pred.h5')
inv = h5py.File('/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/1114_3xbi2se3/test_runs/2022-02-07_14-45-36_/test_out/inversed/hamiltonians_pred.h5')
rot = h5py.File('/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/1114_3xbi2se3/test_runs/2022-02-07_14-45-36_/test_out/rotated/hamiltonians_pred.h5')
processed_structure = '/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/1114_3xbi2se3/processed/original'
# print(np.array(ori['[0, 0, 0, 1, 3]']).max())
alpha, beta, gamma = torch.tensor([0.3, 0.4, 0.5])
rot_matrix = angles_to_matrix(alpha, beta, gamma).numpy()
spinful = True
detailed_output = False


if spinful:
    np_dtype = np.float32
    torch_dtype = torch.complex64
else:
    np_dtype = np.float32
    torch_dtype = torch.float32

rotate_kernel = Rotate(torch_dtype, spinful=spinful)

element = np.loadtxt(os.path.join(processed_structure, 'element.dat')).astype(int)
orbital_types_dir = os.path.join(processed_structure, 'orbital_types.dat')

orbital_types = []
with open(orbital_types_dir) as f:
    line = f.readline()
    while line:
        orbital_types.append(list(map(int, line.split())))
        line = f.readline()
atom_num_orbitals = [sum(map(lambda x: 2 * x + 1, atom_orbital_types)) for atom_orbital_types in orbital_types]

hopping_keys = []
for atom_i in element:
    for atom_j in element:
        hopping_key = f'{atom_i} {atom_j}'
        if hopping_key not in hopping_keys:
            hopping_keys.append(hopping_key)
    
dif_rot, dif_inv = {}, {}
for hopping_key in hopping_keys:
    atom_i, atom_j = hopping_key.split()
    index_i = np.where(element==int(atom_i))[0][0]
    index_j = np.where(element==int(atom_j))[0][0]
    dif_rot[hopping_key] = np.full((atom_num_orbitals[index_i] * (1 + spinful), 
                                    atom_num_orbitals[index_j] * (1 + spinful)), 0.0, dtype=np_dtype)
    dif_inv[hopping_key] = np.full((atom_num_orbitals[index_i] * (1 + spinful), 
                                    atom_num_orbitals[index_j] * (1 + spinful)), 0.0, dtype=np_dtype)

dif_num = {}
for key in dif_rot.keys():
    dif_num[key] = 0
    
for key in ori.keys():
    atom_i = eval(key)[3] - 1
    atom_j = eval(key)[4] - 1
    N_M_str = f'{element[atom_i]} {element[atom_j]}'
    H_ori = np.array(ori[key])
    rot_back = rotate_kernel.rotate_openmx_H_full(torch.tensor(rot[key]).to(torch_dtype), rot_matrix.T, orbital_types[atom_i], orbital_types[atom_j]).numpy()
    dif_rot[N_M_str] += np.abs(H_ori - rot_back)
    inv_back = rotate_kernel.rotate_openmx_H_full(torch.tensor(inv[key]).to(torch_dtype), -torch.eye(3), orbital_types[atom_i], orbital_types[atom_j]).numpy()
    dif_inv[N_M_str] += np.abs(H_ori - inv_back)
    dif_num[N_M_str] += 1


for key in dif_rot.keys():
    dif_rot[key] /= dif_num[key]
    dif_inv[key] /= dif_num[key]


print('===== Compare Rotation =====')
for k, v in dif_rot.items():
    print('----------')
    print(k)
    if detailed_output:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                print(f'{v[i, j]:8.6f}', end=" ")
            print()
    print('max:', np.amax(dif_rot[k]))


print('\n===== Compare Inversion =====')
for k, v in dif_inv.items():
    print('----------')
    print(k)
    if detailed_output:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                print(f'{v[i, j]:8.6f}', end=" ")
            print()
    print('max:', np.amax(dif_inv[k]))

ori.close()
inv.close()
rot.close()
    