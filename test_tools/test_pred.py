import json
import numpy as np
import h5py
import os


h_pred = h5py.File(
    "/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0627_deephCompare/3_bg/error_deeph/4-3/hamiltonians_pred.h5",
    "r",
)
processed_structure = '/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0627_deephCompare/3_bg/processed/twisted2/t-4-3' # containing hamiltonians.h5, element.dat, orbital_types.dat, info.json

simplified_output = True

rh = h5py.File(os.path.join(processed_structure, 'hamiltonians.h5'), "r")
element = np.loadtxt(os.path.join(processed_structure, 'element.dat')).astype(int)
orbital_types_dir = os.path.join(processed_structure, 'orbital_types.dat')
with open(os.path.join(processed_structure, 'info.json'), 'r') as f:
    spinful = json.load(f)['isspinful']

orbital_types = []
with open(orbital_types_dir) as f:
    line = f.readline()
    while line:
        orbital_types.append(list(map(int, line.split())))
        line = f.readline()
atom_num_orbitals = [sum(map(lambda x: 2 * x + 1, atom_orbital_types)) * (1 + spinful) for atom_orbital_types in orbital_types]

hopping_keys = []
for atom_i in element:
    for atom_j in element:
        hopping_key = f'{atom_i} {atom_j}'
        if hopping_key not in hopping_keys:
            hopping_keys.append(hopping_key)
    
dif = {}
for hopping_key in hopping_keys:
    atom_i, atom_j = hopping_key.split()
    index_i = np.where(element==int(atom_i))[0][0]
    index_j = np.where(element==int(atom_j))[0][0]
    dif[hopping_key] = np.full((atom_num_orbitals[index_i], atom_num_orbitals[index_j]), 0.0)

dif_num = {}
for key in dif.keys():
    dif_num[key] = 0
    
for key in h_pred.keys():
    atom_i = eval(key)[3] - 1
    atom_j = eval(key)[4] - 1
    N_M_str = f'{element[atom_i]} {element[atom_j]}'
    dif[N_M_str] += np.abs(np.array(rh[key]) - np.array(h_pred[key]))
    dif_num[N_M_str] += 1

dif_avg = {}
dif_min = {}
dif_max = {}
for key in dif.keys():
    dif[key] /= dif_num[key]
    dif_avg[key] = np.mean(dif[key])
    dif_min[key] = np.amin(dif[key])
    dif_max[key] = np.amax(dif[key])

for k, v in dif.items():
    print('----------')
    print(k)
    if not simplified_output:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                print(f'{v[i, j]:8.6f}', end=" ")
            print()
    print('max:', dif_max[k])
    print('avg:', dif_avg[k])
    print('min:', dif_min[k])

# mode = "mae"  # choices: max / mse / mae
# if mode == "mse":
#     difs = np.full((38, 38), 0.0)
#     num = 0
#     for key in rhp.keys():
#         difs += np.power(np.abs(np.array(rh[key]) - np.array(rhp[key])),2)
#         num += 1
#     dif = difs / num
# elif mode == "max":
#     difs = []
#     for key in rhp.keys():
#         difs.append(np.abs(np.array(rh[key]) - np.array(rhp[key])))
#     difs = np.stack(difs, axis=0)
#     dif = np.amax(difs, axis=0)
# elif mode == "mae":
#     difs = np.full((38, 38), 0.0)
#     num = 0
#     for key in rhp.keys():
#         difs += np.abs(np.array(rh[key]) - np.array(rhp[key]))
#         num += 1
#     dif = difs / num


# for i in range(dif.shape[0]):
#     for j in range(dif.shape[1]):
#         print(dif[i, j], end=" ")
#     print()

rh.close()
h_pred.close()