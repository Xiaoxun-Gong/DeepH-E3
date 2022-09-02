#!/usr/bin/env python
# requires band_deeph.json and band_dft.json created by plot_bands.py

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

min_plot_energy = -3
max_plot_energy = 3

trim_gap = False
trim_min = -0.7
trim_max = 0.72

fontsize = 18
legend = False
legend_kw = {'loc': 'center left', 'bbox_to_anchor': (0.5, 0.53)}

efermi_dft = 0
efermi_deeph = 0

file_tag = 'band'
plot_format = 'png'
plot_dpi = 400


# load dft
with open('band_dft.json', 'r') as f:
    data_dft = json.load(f)
for key, val in data_dft.items():
    if type(val) is list:
        data_dft[key] = np.array(val)
        
# load deeph
with open('band_deeph.json', 'r') as f:
    data_deeph = json.load(f)
for key, val in data_deeph.items():
    if type(val) is list:
        data_deeph[key] = np.array(val)
        
        
hsk_coords = data_dft["hsk_coords"]
plot_hsk_symbols = data_dft["plot_hsk_symbols"]
kpath_num = data_dft["kpath_num"]
spin_num = data_dft["spin_num"]

band_num_each_spin_dft = data_dft["band_num_each_spin"]
kpoints_coords_dft = data_dft["kpoints_coords"]
spin_up_energys_dft = data_dft["spin_up_energys"]
spin_dn_energys_dft = data_dft["spin_dn_energys"]

band_num_each_spin_deeph = data_deeph["band_num_each_spin"]
kpoints_coords_deeph = data_deeph["kpoints_coords"]
spin_up_energys_deeph = data_deeph["spin_up_energys"]
spin_dn_energys_deeph = data_deeph["spin_dn_energys"]


## Design the Figure
# For GUI less server
plt.switch_backend('agg') 
# Set the Fonts
# plt.rcParams.update({'font.size': 14,
#                      'font.family': 'STIXGeneral',
#                      'mathtext.fontset': 'stix'})
plt.rcParams.update({'font.size': fontsize,
                    'font.family': 'Arial',
                    'mathtext.fontset': 'cm'})
# Set the spacing between the axis and labels
plt.rcParams['xtick.major.pad'] = '6'
plt.rcParams['ytick.major.pad'] = '6'
# Set the ticks 'inside' the axis
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

x_min = 0.0
x_max = hsk_coords[-1]

# Create the figure and axis object
if trim_gap:
    fig, (ax_cond, ax_val) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': (max_plot_energy-trim_max, trim_min-min_plot_energy)}, figsize=(5.5, 5.5))
    
    # fig.tight_layout(pad=0.3)
else:
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
# Set the range of plot
if trim_gap:
    ax_val.set_xlim(x_min, x_max)
else:
    ax.set_xlim(x_min, x_max)
if trim_gap:
    ax_val.set_ylim(min_plot_energy, trim_min)
    ax_cond.set_ylim(trim_max, max_plot_energy)
else:
    ax.set_ylim(min_plot_energy, max_plot_energy)
# Set the label of x and y axis
if trim_gap:
    ax_val.tick_params('y', labelsize=0.8*fontsize)
    ax_cond.tick_params('y', labelsize=0.8*fontsize)
    ax_val.set_xlabel('')
    ax_val.set_ylabel('Energy (eV)')
    ax_val.yaxis.set_label_coords(0.055, 0.5, transform=fig.transFigure)
else:
    ax.set_xlabel('')
    ax.set_ylabel('Energy (eV)')
    ax.tick_params('y', labelsize=0.8*fontsize)
# Set the Ticks of x and y axis
if trim_gap:
    ax_val.set_xticks(hsk_coords)
    ax_val.set_xticklabels(plot_hsk_symbols)
    ax_val.tick_params('x', labelsize=fontsize)
else:
    ax.set_xticks(hsk_coords)
    ax.set_xticklabels(plot_hsk_symbols)
    ax.tick_params('x', labelsize=fontsize)

# Plot the solid lines for High symmetic k-points
for kpath_i in range(kpath_num+1):
    if trim_gap:
        ax_val.vlines(hsk_coords[kpath_i], min_plot_energy, trim_min, colors="black", linewidth=0.7)
        ax_cond.vlines(hsk_coords[kpath_i], trim_max, max_plot_energy, colors="black", linewidth=0.7)
    else:
        ax.vlines(hsk_coords[kpath_i], min_plot_energy, max_plot_energy, colors="black", linewidth=0.7)
# Plot the fermi energy surface with a dashed line
if not trim_gap:
    ax.hlines(0.0, x_min, x_max, colors="black", 
            linestyles="dashed", linewidth=0.7)


# Plot the DFT Band Structure
for band_i in range(band_num_each_spin_dft):
    x = kpoints_coords_dft
    y = spin_up_energys_dft[band_i] - efermi_dft
    if trim_gap:
        if y[0] <= 0:
            ax_val.plot(x[y<0], y[y<0], 'r-', linewidth=1.5, zorder=3)
        else:
            ax_cond.plot(x[y>0], y[y>0], 'r-', linewidth=1.5, zorder=3)
    else:
        ax.plot(x, y, 'r-', linewidth=1.5, zorder=3)
if spin_num == 2:
    raise NotImplementedError
    # for band_i in range(band_num_each_spin_dft):
    #     x = kpoints_coords_dft
    #     y = spin_dn_energys_dft[band_i]
    #     ax.plot(x, y, '-', color='#0564c3', linewidth=1)

# Plot the DeepH band structure
for band_i in range(band_num_each_spin_deeph):
    x = kpoints_coords_deeph
    y = spin_up_energys_deeph[band_i] - efermi_deeph
    if trim_gap:
        ax_val.scatter(x[y<0], y[y<0], c='b', s=5, zorder=4)
        ax_cond.scatter(x[y>0], y[y>0], c='b', s=5, zorder=4)
    else:
        ax.scatter(x, y, c='b', s=5, zorder=4)
if spin_num == 2:
    raise NotImplementedError
    # for band_i in range(band_num_each_spin_deeph):
    #     x = kpoints_coords_deeph
    #     y = spin_dn_energys_deeph[band_i]
    #     ax.plot(x, y, '-', color='#0564c3', linewidth=1)

# create legends
if legend:
    band_dft_proxy = mlines.Line2D([], [], linestyle='-', color='r', linewidth=1.5)
    band_deeph_proxy = mlines.Line2D([], [], linestyle=':', color='b', linewidth=1.5)
    fig.legend((band_dft_proxy, band_deeph_proxy), ('DFT', 'DeepH-E3'), prop={'size': 0.8*fontsize}, **legend_kw)

if trim_gap:
    ax_cond.spines.bottom.set_visible(False)
    ax_val.spines.top.set_visible(False)
    ax_cond.xaxis.tick_top()
    ax_cond.tick_params(labeltop=False)  # don't put tick labels at the top
    ax_val.xaxis.tick_bottom()
    
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_cond.plot([0, 1], [0, 0], transform=ax_cond.transAxes, **kwargs)
    ax_val.plot([0, 1], [1, 1], transform=ax_val.transAxes, **kwargs)
            
# Save the figure
plot_filename = "%s.%s" %(file_tag, plot_format)
plt.tight_layout()
fig.subplots_adjust(hspace=0.05)
plt.savefig(plot_filename, format=plot_format, dpi=plot_dpi, transparent=False)
plt.savefig('band.svg', transparent=True)
