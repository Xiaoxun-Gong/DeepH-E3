#!/usr/bin/env python

import os
import argparse
import warnings
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process data from openmx output.')
parser.add_argument('--input_dir', type=str, default='/home/lihe/hdd/materials_data/MoS2/md_openmx/configuration/', help='Every folder under input_dir containing openmx.scfout will be recognized as a structure folder.')
parser.add_argument('--output_dir', type=str, default='/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/1004_MoS2/processed/', help='Processed structure information will be stored here.')
parser.add_argument('--simpout', action='store_true', help='Supress the output of each data processor.')
parser.add_argument('--olp', action='store_true', help='Output overlaps.h5.')
args = parser.parse_args()

supress_output = args.simpout

datajl_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.split(datajl_dir)[-1] == 'DeepH-E3':
    datajl_dir = os.path.join(datajl_dir, 'deephe3/process_data_tools')
datajl_dir = os.path.join(datajl_dir, 'openmx_get_data.jl')

# = find structures
stru_path_list = []
print(f'Looking for DFT calculated data under: {args.input_dir}')
for root, dirs, files in os.walk(args.input_dir):
    if 'openmx.scfout' in files:
        stru_path_list.append(os.path.abspath(root))

assert len(stru_path_list) > 0, 'cannot find any structure'
print(f'Found {len(stru_path_list)} structure(s).')      
        
# = process structures
os.makedirs(args.output_dir, exist_ok=True)
print('Processing...')
stru_path_list_iter = tqdm(stru_path_list) if supress_output else stru_path_list
for stru_input_path in stru_path_list_iter:
    relpath = os.path.split(stru_input_path)[-1]
    stru_output_path = os.path.join(args.output_dir, relpath)
    if os.path.isdir(stru_output_path):
        warnings.warn('Processed structures might already be existing under output_dir')
    os.makedirs(stru_output_path, exist_ok=True)
    # TODO might need modification
    cmd = f'julia {datajl_dir} --input_dir {stru_input_path} --output_dir {stru_output_path}' + \
           (' --if_OLP true' if args.olp else '') + \
           (' > /dev/null 2>&1' if supress_output else '')
    return_code = os.system(cmd)
    assert return_code == 0, f'Error occured in executing command "{cmd}". Try not to include --simpout to see error messages.'
print(f'All processed data successfully saved to {args.output_dir}')