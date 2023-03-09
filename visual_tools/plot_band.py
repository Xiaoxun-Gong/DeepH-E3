#!/usr/bin/env python
# Author: LiYang@cmt.tsinghua
# Date: 2021.04.08
# Descripution: This python code is designed for calculate the fermi energy
#                and plot the band structure use the *.Band file.
#              ::Input File::
#                - *.Band
#              ::Output File::
#                - *.Band.png
#                - *.Band.conv
#
# Usage: python3 plot_openmx_band.py -h

import sys
import argparse
import math
import numpy as np
import re
import matplotlib.pyplot as plt
import json
import copy

from_json = True

BOHR = 0.529177210903 #A
HARTREE = 27.21138602 #eV
GREEK_SYMBOLS = ['Gamma','Delta','Theta','Lambda','Xi',
                 'Pi','Sigma','Phi','Psi','Omega']

class BandData():
  def __init__(self, data_type):
    self.data_type = data_type
    self.band_data = {}
    return

  def file_read(self, filename):
    with open(filename) as frp:
      self.file_lines = frp.readlines()
    return
  
  def __read_openmx_basic_info(self):
    '''Read int the basic information of the band'''
    # Read in the basic info.
    line = self.file_lines[0].split()
    band_num_each_spin = int(line[0])
    spin_num = int(line[1]) + 1
    band_num = band_num_each_spin * spin_num
    fermi_energy = float(line[2]) * HARTREE
    
    # Record the info.
    self.band_data["band_num_each_spin"] = band_num_each_spin
    self.band_data["spin_num"] = spin_num
    self.band_data["band_num"] = band_num
    self.band_data["fermi_energy"] = fermi_energy
    
    # Variours check
    if spin_num not in [1, 2]:
      print("[error] Spin Number ERROR!")
      sys.exit(1)
    return
  
  def __read_openmx_rlv(self):
    '''Read in the openmx reciprocal lattice vector (rlv)'''
    # Reciprocal lattice vector :: rlv
    # Each row of the 'rlv' matrix is a basis vector
    line = self.file_lines[1].split()
    rlv = np.zeros((3, 3))
    rlv[0, 0] = float(line[0])
    rlv[0, 1] = float(line[1])
    rlv[0, 2] = float(line[2])
    rlv[1, 0] = float(line[3])
    rlv[1, 1] = float(line[4])
    rlv[1, 2] = float(line[5])
    rlv[2, 0] = float(line[6])
    rlv[2, 1] = float(line[7])
    rlv[2, 2] = float(line[8])
    self.band_data["rlv"] = rlv

  def __read_openmx_kpath(self):
    '''Read the openmx band kpath info'''
    # Read in the kpath number
    line = self.file_lines[2]
    kpath_num = int(line)
    self.band_data["kpath_num"] = kpath_num
    # Read in the kpath block
    kpath_densities = [1 for _ in range(kpath_num)]
    kpath_vectors = [[[], []] for _ in range(kpath_num)]
    kpath_symbols = [[] for _ in range(kpath_num)]
    line_i = 2
    for kpath_i in range(kpath_num):
      line_i += 1
      line = self.file_lines[line_i].split()
      kpath_densities[kpath_i] = int(line[0])
      kpath_vectors[kpath_i][0] = [float(line[1]), float(line[2]), 
                                   float(line[3])]
      kpath_vectors[kpath_i][1] = [float(line[4]), float(line[5]), 
                                   float(line[6])]
      kpath_symbols[kpath_i] = [line[7], line[8]]
    kpoints_num = sum(kpath_densities)
    # Record the data
    self.band_data["kpath_densities"] = kpath_densities
    self.band_data["kpath_vectors"] = kpath_vectors
    self.band_data["kpath_symbols"] = kpath_symbols
    self.band_data["kpoints_num"] = kpoints_num
    return

  def __sort_energys(self, energys):
    '''sort the energys'''
    kpoints_num = len(energys[0])
    band_num = len(energys)
    sorted_energys = np.zeros((band_num, kpoints_num))
    for kpoints_i in range(kpoints_num):
        current_k_column = energys[:, kpoints_i]
        sorted_current_k_column = np.sort(current_k_column)
        sorted_energys[:, kpoints_i] = sorted_current_k_column
    return sorted_energys

  def __read_openmx_energys(self):
    '''Read in the OPENMX energys'''
    band_num_each_spin = self.band_data["band_num_each_spin"]
    kpoints_num = self.band_data["kpoints_num"]
    spin_num = self.band_data["spin_num"]
    ### Prepare the data array
    spin_up_energys = np.zeros((band_num_each_spin, kpoints_num))
    if spin_num == 2:
      spin_dn_energys = np.zeros((band_num_each_spin, kpoints_num))
    else:
      spin_dn_energys = []
    kpoint_vectors = [[] for _ in range(kpoints_num)]
    ### Read in the data
    line_i = 2 + self.band_data["kpath_num"]
    for kpoint_i in range(kpoints_num):
      # The kpoints line
      line_i += 1
      line = self.file_lines[line_i].split()
      kpoint_vectors[kpoint_i] = [float(line[1]), 
                                  float(line[2]), 
                                  float(line[3])]
      # The (spin-up) energys line
      line_i += 1
      line = self.file_lines[line_i].split()
      for band_i in range(band_num_each_spin):
        spin_up_energys[band_i, kpoint_i] = float(line[band_i]) * HARTREE
      # The spin-down energys line
      if spin_num == 2:
        line_i += 2
        line = self.file_lines[line_i].split()
        for band_i in range(band_num_each_spin):
          spin_dn_energys[band_i, kpoint_i] = float(line[band_i]) * HARTREE
    ### Post Process the Band energys
    sorted_energys = spin_up_energys
    if spin_num == 2:
      sorted_energys = np.concatenate((spin_up_energys, spin_up_energys),
                                      axis=0)
    # Sort the band
    sorted_energys = self.__sort_energys(sorted_energys)
    ### Record the data
    self.band_data["kpoint_vectors"] = kpoint_vectors
    self.band_data["spin_up_energys"] = spin_up_energys
    self.band_data["spin_dn_energys"] = spin_dn_energys
    self.band_data["sorted_energys"] = sorted_energys
    return

  def __cal_k_distance(self, rlv, beg_kpt_frac, end_kpt_frac, distance_unit=1):
    '''Calcualte the k-cooridnate in Cartesian indicator'''
    beg_kpt_cart = np.array([0.0, 0.0, 0.0])
    end_kpt_cart = np.array([0.0, 0.0, 0.0])
    #
    # k_cart = k_frac * rlv
    #                               __              __
    #                               | b_1x b_1y b_1z |
    #        = (kf_1, kf_2, kf_3) * | b_2x b_2y b_2z |
    #                               | b_3x b_3y b_3z |
    #                               --              --
    #        = (kc_x, kc_y, kc_z)
    #
    for xyz in range(3):
      beg_kpt_cart[xyz] = 0.0
      for b_i in range(3):
        beg_kpt_cart[xyz] += beg_kpt_frac[b_i] * rlv[b_i, xyz]
        end_kpt_cart[xyz] += end_kpt_frac[b_i] * rlv[b_i, xyz]
    # Calculate the k distance of the two kpoints
    k_distance = math.sqrt(sum((beg_kpt_cart-end_kpt_cart) ** 2))
    k_distance /= distance_unit
    return k_distance

  def __get_kpt_coords(self, distance_unit):
    '''Get the coords of each kpoints in k-space'''
    kpath_num = self.band_data["kpath_num"]
    kpath_vectors = self.band_data["kpath_vectors"]
    kpoint_vectors = self.band_data["kpoint_vectors"]
    kpath_densities = self.band_data["kpath_densities"]
    rlv = self.band_data["rlv"]
    kpoints_num = self.band_data["kpoints_num"]
    ### Prepare the data list
    hsk_distance_list = np.zeros(kpath_num)
    sum_hsk_distance_list = np.zeros(kpath_num)
    kpoints_coords = np.zeros(kpoints_num)
    hsk_coords = np.zeros(kpath_num+1)
    ### Get the distance for high-symmetry kpoints
    for kpath_i in range(kpath_num):
      start_hsk = kpath_vectors[kpath_i][0]
      end_hsk = kpath_vectors[kpath_i][1]
      hsk_distance_list[kpath_i] = \
        self.__cal_k_distance(rlv, start_hsk, end_hsk, distance_unit)
      sum_hsk_distance_list[kpath_i] = \
        sum(hsk_distance_list[0:kpath_i+1])
    hsk_coords[1:] = sum_hsk_distance_list
    ### Get the distance in k-space of k-points on the k-path
    kpoints_i = -1
    for kpath_i in range(kpath_num):
      # Count the Previous kpath distance
      pre_path_distance = hsk_coords[kpath_i]
      # Calculate the kpoints' distance in current kpath
      for _ in range(kpath_densities[kpath_i]):
        kpoints_i += 1
        start_hsk = kpath_vectors[kpath_i][0]
        end_hsk = kpoint_vectors[kpoints_i]
        # The total distance equals to (pre_dis + curr_dis)
        kpoints_coords[kpoints_i] = pre_path_distance + \
          self.__cal_k_distance(rlv, start_hsk, end_hsk, distance_unit)
    ### Record the data
    self.band_data["hsk_coords"] = hsk_coords
    self.band_data["kpoints_coords"] = kpoints_coords
    return

  def __refine_fermi_energy(self):
    '''Refine the fermi energy and the center of HOMO and LUMO'''
    fermi_energy = self.band_data["fermi_energy"]
    energys = self.band_data["sorted_energys"]
    band_num = self.band_data["band_num"]
    kpoints_num = self.band_data["kpoints_num"]
    spin_num = self.band_data["spin_num"]
    # find the LUMO and HOMO
    min_homo_ediff = fermi_energy - energys[0, 0]
    homo_band_index = 0
    homo_kpt_index = 0
    min_lumo_ediff = energys[band_num-1, 0] - fermi_energy
    lumo_band_index = band_num-1
    lumo_kpt_index = 0
    for band_i in range(band_num):
      for kpoint_i in range(kpoints_num):
        curr_energy = energys[band_i, kpoint_i]
        lumo_ediff = curr_energy - fermi_energy 
        homo_ediff = fermi_energy - curr_energy
        if (lumo_ediff >= 0) and (lumo_ediff < min_lumo_ediff):
          min_lumo_ediff = lumo_ediff
          lumo_band_index = band_i
          lumo_kpt_index = kpoint_i
        elif (homo_ediff > 0) and (homo_ediff < min_homo_ediff):
          min_homo_ediff = homo_ediff
          homo_band_index = band_i
          homo_kpt_index = kpoint_i
    lumo_energy = energys[lumo_band_index, lumo_kpt_index]
    homo_energy = energys[homo_band_index, homo_kpt_index]
    refined_fermi_energy = (lumo_energy + homo_energy) / 2

    # lihe: no need to do refine
    refined_fermi_energy = fermi_energy
    
    # Shift the zero energy to the fermi level
    self.band_data["origin_spin_up_energys"] = self.band_data["spin_up_energys"]
    self.band_data["origin_spin_dn_energys"] = self.band_data["spin_dn_energys"]
    self.band_data["origin_sorted_energys"] = self.band_data["sorted_energys"]
    self.band_data["spin_up_energys"] -= refined_fermi_energy
    self.band_data["sorted_energys"] -= refined_fermi_energy
    if spin_num == 2:
      self.band_data["spin_up_energys"] -= refined_fermi_energy
    # Record the data
    self.band_data["refined_fermi_energy"] = refined_fermi_energy
    self.band_data["lumo_energy"] = lumo_energy
    self.band_data["homo_energy"] = homo_energy
    self.band_data["lumo_band_index"] = lumo_band_index
    self.band_data["lumo_kpt_index"] = lumo_kpt_index
    self.band_data["homo_band_index"] = homo_band_index
    self.band_data["homo_kpt_index"] = homo_kpt_index
    self.band_data["min_lumo_ediff"] = min_lumo_ediff
    self.band_data["min_homo_ediff"] = min_homo_ediff
    return

  def __prepare_plot_kpt_symbol(self):
    '''Prepare the kpoints symbols for the plot'''
    kpath_num = self.band_data["kpath_num"]
    kpath_symbols = self.band_data["kpath_symbols"]
    # Prepare the symbol of k-axis (xtics)
    hsk_symbols = ['' for _ in range(kpath_num+1)]
    # Set 
    hsk_symbols[0] = kpath_symbols[0][0]
    for kpath_i in range(1, kpath_num):
      if kpath_symbols[kpath_i][0] == kpath_symbols[kpath_i-1][1]:
        hsk_symbols[kpath_i] = kpath_symbols[kpath_i][0]
      else:
        hsk_symbols[kpath_i] = "%s|%s" %(kpath_symbols[kpath_i - 1][1],
                                         kpath_symbols[kpath_i][0])
    hsk_symbols[-1] = kpath_symbols[-1][1]
    ## Plot the Band
    plot_hsk_symbols = []
    for symbol in hsk_symbols:
      symbol = symbol.replace("\\", "")
      for greek_symbol in GREEK_SYMBOLS:
        if greek_symbol == 'Gamma':
          latex_greek_symbol = 'Γ'
        else:
          latex_greek_symbol = "$\\" + greek_symbol + "$"
        symbol = re.sub(greek_symbol, "orz", symbol, 
                        flags=re.I)
        symbol = symbol.replace("orz", latex_greek_symbol)
      symbol = re.sub(r'_\d+', lambda x:'$'+x[0]+'$', symbol)
      plot_hsk_symbols.append(symbol)
    ## Record the data
    self.band_data["hsk_symbols"] = hsk_symbols
    self.band_data["plot_hsk_symbols"] = plot_hsk_symbols
    return
 
  def get_band_data(self):
    '''Get the band data'''
    # Read in data
    if self.data_type == 'openmx':
      distance_unit = 2 * math.pi * BOHR
      self.__read_openmx_basic_info()
      self.__read_openmx_rlv()
      self.__read_openmx_kpath()
      self.__read_openmx_energys()
    elif self.data_type == 'vasp':
      print("[TODO]")
      sys.exit(0)
    else:
      print("[TODO]")
      sys.exit(0)
    # Post process
    self.__get_kpt_coords(distance_unit)
    self.__refine_fermi_energy()
    self.__prepare_plot_kpt_symbol()
    return


def get_command_line_input():
  '''Read in the command line parameters'''
  parser = argparse.ArgumentParser("Basic band plot parameters")
  parser.add_argument('-t', '--type', dest='data_type', 
                      default='openmx', type=str, choices=['openmx', 'vasp'],
                      help='Type of the band calculation.')
  parser.add_argument('-d', '--ymin', dest='min_plot_energy', 
                      default=-6, type=float,
                      help='Minimal plot energy windows.')
  parser.add_argument('-u', '--ymax', dest='max_plot_energy', 
                      default=6, type=float,
                      help='Maximal plot energy windows.')
  parser.add_argument('-f', '--format', dest='plot_format', 
                      default='png', type=str, choices=['png', 'eps', 'pdf'],
                      help='Plot format.')
  parser.add_argument('-i', '--dpi', dest='plot_dpi', 
                      default=400, type=int,
                      help='Plot resolution (dpi).')
  parser.add_argument('-a', '--datafile', dest='data_filename', 
                      default='openmx.Band', type=str,
                      help='Input data filename.')
  parser.add_argument('-o', '--output', dest='file_tag', 
                      default='band', type=str,
                      help='Output file tag name.')
  parser.add_argument('-x', '--no-plot', dest='no_plot', action='store_const',
                      const=True, default=False,
                      help='Do not plot the band.')
  args = parser.parse_args()
  plot_args = {"data_type"       : args.data_type ,
               "min_plot_energy" : args.min_plot_energy,
               "max_plot_energy" : args.max_plot_energy,
               "plot_format"     : args.plot_format,
               "plot_dpi"        : args.plot_dpi,
               "file_tag"        : args.file_tag,
               "data_filename"   : args.data_filename,
               "no_plot"         : args.no_plot}
  return plot_args


def band_save_to_json(data, file_tag):
  '''Save the band data to json file'''
  filename = "%s.json" %file_tag
  data_save = copy.deepcopy(data)
  for key, vals in data_save.items():
    if type(vals) is np.ndarray:
      data_save[key] = vals.tolist()
  with open(filename, 'w') as jfwp:
    json.dump(data_save, jfwp)
  return


def band_plot(band_data_obj, plot_args):
  '''band plot function'''
  hsk_coords = band_data_obj.band_data["hsk_coords"]
  plot_hsk_symbols = band_data_obj.band_data["plot_hsk_symbols"]
  kpath_num = band_data_obj.band_data["kpath_num"]
  band_num_each_spin = band_data_obj.band_data["band_num_each_spin"]
  kpoints_coords = band_data_obj.band_data["kpoints_coords"]
  spin_num = band_data_obj.band_data["spin_num"]
  spin_up_energys = band_data_obj.band_data["spin_up_energys"]
  spin_dn_energys = band_data_obj.band_data["spin_dn_energys"]
  min_plot_energy = plot_args["min_plot_energy"]
  max_plot_energy = plot_args["max_plot_energy"]
  file_tag = plot_args["file_tag"]
  plot_format = plot_args["plot_format"]
  plot_dpi = plot_args["plot_dpi"]
  ## Design the Figure
  # For GUI less server
  plt.switch_backend('agg') 
  # Set the Fonts
  # plt.rcParams.update({'font.size': 14,
  #                      'font.family': 'STIXGeneral',
  #                      'mathtext.fontset': 'stix'})
  plt.rcParams.update({'font.size': 22,
                     'font.family': 'Arial',
                     'mathtext.fontset': 'cm'})
  # Set the spacing between the axis and labels
  plt.rcParams['xtick.major.pad'] = '6'
  plt.rcParams['ytick.major.pad'] = '6'
  # Set the ticks 'inside' the axis
  plt.rcParams['xtick.direction'] = 'in'
  plt.rcParams['ytick.direction'] = 'in'
  # Create the figure and axis object
  fig = plt.figure(figsize=(5.5, 5.5))
  band_plot = fig.add_subplot(1, 1, 1)
  # Set the range of plot
  x_min = 0.0
  x_max = hsk_coords[-1]
  y_min = min_plot_energy
  y_max = max_plot_energy
  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)
  # Set the label of x and y axis
  plt.xlabel('')
  plt.ylabel('Energy (eV)')
  # Set the Ticks of x and y axis
  plt.xticks(hsk_coords)
  band_plot.set_xticklabels(plot_hsk_symbols)
  plt.yticks(size=14)
  # Plot the solid lines for High symmetic k-points
  for kpath_i in range(kpath_num+1):
    plt.vlines(hsk_coords[kpath_i], y_min, y_max, colors="black", linewidth=0.7)
  # Plot the fermi energy surface with a dashed line
  plt.hlines(0.0, x_min, x_max, colors="black", 
             linestyles="dashed", linewidth=0.7)
  # Plot the Band Structure
  for band_i in range(band_num_each_spin):
      x = kpoints_coords
      y = spin_up_energys[band_i]
      band_plot.plot(x, y, 'r-', linewidth=1.5)
  if spin_num == 2:
      for band_i in range(band_num_each_spin):
          x = kpoints_coords
          y = spin_dn_energys[band_i]
          band_plot.plot(x, y, '-', color='#0564c3', linewidth=1)
  # Save the figure
  plot_filename = "%s.%s" %(file_tag, plot_format)
  plt.tight_layout()
  plt.savefig(plot_filename, format=plot_format, dpi=plot_dpi, transparent=False)
  plt.savefig('band.svg', transparent=False)
  return 


def band():
  '''band functions'''
  plot_args = get_command_line_input()
  band_data_obj = BandData(plot_args["data_type"])
  if not from_json:
    band_data_obj.file_read(plot_args["data_filename"])
    band_data_obj.get_band_data()
    band_save_to_json(band_data_obj.band_data, plot_args["file_tag"])
  else:
    with open('band.json', 'r') as f:
      data_new = json.load(f)
    for key, val in data_new.items():
      if type(val) is list:
        data_new[key] = np.array(val)
    band_data_obj.band_data = data_new
  # ==
  if not plot_args["no_plot"]:
    band_plot(band_data_obj, plot_args)


#+----------------+
#|  Main Process  |
#+----------------+

def main():
  band()
  return

if __name__=='__main__':
  main()