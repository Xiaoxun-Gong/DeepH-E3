import numpy as np
import json
from scipy import interpolate

tol = 0.1
spline_interpolate = True
fine_grid_size = 200

with open('band.json', 'r') as f:
    data = json.load(f)

energies = np.array(data['spin_up_energys']).T # [band, kpoint] -> [kpoint, band]
energies = np.sort(energies, axis=1)
nbnd = energies.shape[1]

for i in range(1, energies.shape[0]):
    num_shift = 0
    while True:
        diff = energies[i] - energies[i-1]
        diffmax_ix = np.argmax(np.abs(diff))
        diffmax = diff[diffmax_ix]
        if np.abs(diffmax) > tol:
            num_shift += 1
            if diffmax < 0:
                energies[i] = np.roll(energies[i], -1)
                energies[i, nbnd-num_shift:nbnd] = energies[i-1, nbnd-num_shift:nbnd]
            else:
                energies[i] = np.roll(energies[i], 1)
                energies[i, 0:num_shift] = energies[i-1, 0:num_shift]
        else:
            break
        
if spline_interpolate:
    kpoints_coords = data["kpoints_coords"]
    energies = energies.T # [kpoint, band] -> [band, kpoint] 
    
    # remove duplication
    dup_ixs = []
    for i in range(len(kpoints_coords)-1):
        if kpoints_coords[i] == kpoints_coords[i+1]:
            dup_ixs.append(i)
    for i in reversed(dup_ixs):
        kpoints_coords.pop(i)
        energies = np.delete(energies, i, axis=1)
        
    kpts_fine = np.linspace(kpoints_coords[0], kpoints_coords[-1], fine_grid_size)
    energies_fine = []
    for ibnd in range(len(energies)):
        s = interpolate.InterpolatedUnivariateSpline(kpoints_coords, energies[ibnd])
        energies_fine.append(s(kpts_fine))
    energies_fine = np.stack(energies_fine)
        
    data["kpoints_coords"] = kpts_fine.tolist()
    data["spin_up_energys"] = energies_fine.tolist()
    
else:
    data['spin_up_energys'] = energies.T.tolist()
    
with open('band_reconnect.json', 'w') as f:
    json.dump(data, f)
