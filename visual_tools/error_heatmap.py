#!/usr/bin/env python

import json
import sys
from functools import reduce
from operator import add
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as tfs

# plot kws
filename_json = sys.argv[1]
filename_plot = filename_json.split('.')[0]

fontsize = 18
tick_interval = 1

plt.rcParams.update({'font.size': fontsize,
                    'font.family': 'Arial',
                    'mathtext.fontset': 'cm'}
                    )

with open(filename_json, 'r') as f:
    error_info = json.load(f)

spinful = error_info['spinful']
chem_symbols = error_info['chem_symbols']
element_types = error_info['element_types'] # List[List], two sublists indicating the element types on each sides
element_num_y, element_num_x = len(element_types[0]), len(element_types[1]) # notice that 0 is for y, 1 is for x
orbital_types = error_info['orbital_types']
num_orbitals = [sum(map(lambda x: 2*x+1, o)) for o in orbital_types]
errors = np.array(error_info['errors']) # [[np.array(error_mat) for error_mat in a] for a in error_info['errors']] # [[None for _ in range(num_element)] for _ in range(num_element)]

fig_width = 6 # * (1 + spinful) 

# fig, axes = plt.subplots(element_num1, element_num2, figsize=(element_num1*fig_width, element_num2*fig_width), squeeze=False)
fig, ax = plt.subplots(figsize=(fig_width, fig_width))

yticks = range(0, errors.shape[0], tick_interval)
xticks = range(0, errors.shape[1], tick_interval)
ylabels = reduce(add, (list(range(1, 1+num_orbitals[i])) for i in element_types[0] for _ in range(1+spinful)))
xlabels = reduce(add, (list(range(1, 1+num_orbitals[i])) for i in element_types[1] for _ in range(1+spinful)))
ylabels = [ylabels[i] for i in yticks]
xlabels = [xlabels[i] for i in xticks]

im = ax.imshow(errors, cmap='Blues')
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.45) # label='(meV)'
cbar.ax.set_title('meV', fontsize=1.1*fontsize)
# cbar.ax.tick_params(labelsize=12)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
# xlabels = [xlabels[i] if i%tick_interval==0 else '' for i in range(len(xlabels))]
# ylabels = [ylabels[i] if i%tick_interval==0 else '' for i in range(len(ylabels))]
ax.set_xticklabels(xlabels, fontsize=0.8*fontsize)
ax.set_yticklabels(ylabels, fontsize=0.8*fontsize)
ax.set_xlabel(r'Orbital $\beta$', fontsize=1.1*fontsize)
ax.set_ylabel(r'Orbital $\alpha$', fontsize=1.1*fontsize)
ax.set_title(r'MAE of $H_{i\alpha, j\beta}$', pad=10, fontsize=1.3*fontsize)

plt.tight_layout()

num_orbitals_cumsum_y = np.cumsum([0] + [num_orbitals[i] for i in element_types[0] for _ in range(1+spinful)])
ymax = num_orbitals_cumsum_y[-1]
num_orbitals_cumsum_x = np.cumsum([0] + [num_orbitals[i] for i in element_types[1] for _ in range(1+spinful)])
xmax = num_orbitals_cumsum_x[-1]
    
arrow = [b'\xe2\x86\x91'.decode('utf8'), b'\xe2\x86\x93'.decode('utf8')]

# hopping species
species_padding = 1.7 * fontsize / 72 # 1inch = 72pt
labelpad = 1.9 * fontsize

plt.subplots_adjust(left=0.2)

ax.set_xlabel(ax.get_xlabel(), labelpad=labelpad)
ax.set_ylabel(ax.get_ylabel(), labelpad=labelpad)

# https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
# scale according to dpi_scale_trans, translate according to ax.transAxes
offset_inch_trans = fig.dpi_scale_trans + tfs.ScaledTranslation(0, 0, ax.transAxes)
xtrans = tfs.blended_transform_factory(ax.get_xaxis_transform(), offset_inch_trans)
ytrans = tfs.blended_transform_factory(offset_inch_trans, ax.get_yaxis_transform())

# y species
for i in range(element_num_y):
    if spinful:
        for s in (0, 1):
            ax.text(-species_padding - 0.05, 
                    (num_orbitals_cumsum_y[2*i+s]+num_orbitals_cumsum_y[2*i+s+1])/2-0.5, 
                    chem_symbols[element_types[0][i]] + arrow[s],
                    transform=ytrans, ha='right', va='center', rotation=90, fontsize=0.9*fontsize) 
    else:
        ax.text(-species_padding - 0.05, 
                (num_orbitals_cumsum_y[i]+num_orbitals_cumsum_y[i+1])/2-0.5, 
                chem_symbols[element_types[0][i]],
                transform=ytrans, ha='right', va='center', rotation=90, fontsize=0.9*fontsize)

# x species
for i in range(element_num_x):
    if spinful:
        for s in (0, 1):
            ax.text((num_orbitals_cumsum_x[2*i+s]+num_orbitals_cumsum_x[2*i+s+1])/2-0.5, 
                    -species_padding - 0.05, 
                    chem_symbols[element_types[1][i]] + arrow[s],
                    transform=xtrans, ha='center', va='top', rotation=0, fontsize=0.9*fontsize) # x species
    else:
        ax.text((num_orbitals_cumsum_x[i]+num_orbitals_cumsum_x[i+1])/2-0.5, 
                -species_padding - 0.05, 
                chem_symbols[element_types[1][i]],
                transform=xtrans, ha='center', va='top', rotation=0, fontsize=0.9*fontsize) # x species
        
# lines separating elements
for i in range(1, len(num_orbitals_cumsum_y)-1):
    ax.hlines(num_orbitals_cumsum_y[i]-0.5, -0.5, xmax-0.5, color='black', 
              linewidth=0.7 if spinful and i % 2 == 1 else 1)
for i in range(1, len(num_orbitals_cumsum_x)-1):
    ax.vlines(num_orbitals_cumsum_x[i]-0.5, -0.5, ymax-0.5, color='black', 
              linewidth=0.7 if spinful and i % 2 == 1 else 1)
    
# small-line x
for i in range(len(num_orbitals_cumsum_x)):
    ax.plot((num_orbitals_cumsum_x[i]-0.5, num_orbitals_cumsum_x[i]-0.5), 
            (-species_padding, -species_padding+0.05),
            transform=xtrans, clip_on=False, color='black', 
            linewidth=0.7 if spinful and i % 2 == 1 else 1)
# small-line y
for i in range(len(num_orbitals_cumsum_y)):
    ax.plot((-species_padding, -species_padding+0.05),
            (num_orbitals_cumsum_y[i]-0.5, num_orbitals_cumsum_y[i]-0.5),
            transform=ytrans, clip_on=False, color='black', 
            linewidth=0.7 if spinful and i % 2 == 1 else 1)        
    
# long-line x
ax.plot((-0.5, xmax-0.5), (-species_padding, -species_padding),
        transform=xtrans, clip_on=False, color='black', linewidth=1)
# long-line y
ax.plot((-species_padding, -species_padding), (-0.5, ymax-0.5),
        transform=ytrans, clip_on=False, color='black', linewidth=1)
        
# lines of equivariant blocks
if spinful:
    orbital_types_sp = [ot * 2 for ot in orbital_types]
else:
    orbital_types_sp = orbital_types
width = 0.25 if spinful else 0.5
# horizontal lines
orbital_sizes = 2 * np.concatenate(list(map(np.array, (orbital_types_sp[i] for i in element_types[0])))) + 1
orbital_sizes_cumsum = np.cumsum(orbital_sizes)
for i in range(len(orbital_sizes_cumsum)-1):
    ax.hlines(orbital_sizes_cumsum[i]-0.5, -0.5, xmax-0.5, color='black', linewidth=width, linestyles='dashed')
    
# vertial lines
orbital_sizes = 2 * np.concatenate(list(map(np.array, (orbital_types_sp[i] for i in element_types[1])))) + 1
orbital_sizes_cumsum = np.cumsum(orbital_sizes)
for i in range(len(orbital_sizes_cumsum)-1):
    ax.vlines(orbital_sizes_cumsum[i]-0.5, -0.5, ymax-0.5, color='black', linewidth=width, linestyles='dashed')
    

plt.savefig(f'{filename_plot}.png', dpi=800)
plt.savefig(f'{filename_plot}.svg')
