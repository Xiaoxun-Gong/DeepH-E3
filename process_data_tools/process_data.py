import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process data from openmx output.')
parser.add_argument('--input_dir', type=str, default='/home/lihe/hdd/materials_data/MoS2/md_openmx/configuration/', help='Every folder under input_dir containing openmx.scfout will be recognized as a structure folder.')
parser.add_argument('--output_dir', type=str, default='/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/1004_MoS2/processed/', help='Processed structure information will be stored here.')
args = parser.parse_args()

supress_output = True

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# = find structures
stru_path_list = []
for root, dirs, files in os.walk(args.input_dir):
    if 'openmx.scfout' in files:
        stru_path_list.append(os.path.abspath(root))

print(f'Found {len(stru_path_list)} structures.')      
        
# = process structures
os.makedirs(args.output_dir, exist_ok=True)
print('Processing...')
stru_path_list_iter = tqdm(stru_path_list) if supress_output else stru_path_list
for stru_input_path in stru_path_list_iter:
    relpath = os.path.split(stru_input_path)[-1]
    stru_output_path = os.path.join(args.output_dir, relpath)
    os.makedirs(stru_output_path, exist_ok=True)
    # TODO might need modification
    cmd = f'julia get_data.jl --input_dir {stru_input_path} --output_dir {stru_output_path}' + \
           (' > /dev/null 2>&1' if supress_output else '')
    os.system(cmd)