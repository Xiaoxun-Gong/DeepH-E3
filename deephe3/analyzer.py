import json
import re
import os
import time
import h5py
import numpy as np
import csv

from .kernel import NetOutInfo, DeepHE3Kernel

class testResultAnalyzer:
    
    def __init__(self, data_dir, output_dir):
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        self.kernel = DeepHE3Kernel()
        self.kernel.load_config(train_config_path=os.path.join(data_dir, 'src/train.ini'))
        # self.kernel.load_dataset_info(os.path.join(training_dir, 'src/dataset_info.json'))
        # self.kernel.config_set_target()
        # self.kernel.register_constructor()
        self.kernel.net_out_info = NetOutInfo.from_json(os.path.join(data_dir, 'src'))
        self.kernel.dataset_info = self.kernel.net_out_info.dataset_info
        
        self.dataset_info = self.kernel.dataset_info
        
        self.h5file = os.path.join(data_dir, 'test_result.h5')
        
    def view_structures(self, detailed=False):
        with h5py.File(self.h5file, 'r') as f:
            if detailed:
                print(f.visit(lambda x: print(x)))
            else:
                for name in f:
                    print(name, end=' ')
                print()
        return 0
    
    def error_plot(self, include=[], exclude=[], mode='mae'):
        if mode not in ['mae', 'mse', 'rmse']:
            print(f'Unknown mode: {mode}')
            return 1
        
        spinful = self.dataset_info.spinful
        num_element = len(self.dataset_info.index_to_Z)
        
        error_info = {}
        error_info['spinful'] = spinful
        error_info['chem_symbols'] = [Z_to_chemsymbol(Z.item()) for Z in self.dataset_info.index_to_Z]
        error_info['orbital_types'] = self.dataset_info.orbital_types
        
        with h5py.File(self.h5file, 'r') as f:
            included = select_structures(list(f.keys()), include, exclude)
            if len(included) == 0:
                print('No structures included')
                return 1
            
            output_dir = os.path.join(self.output_dir, str(time.strftime(mode.upper() + '_' + '%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
            os.makedirs(output_dir, exist_ok=True)
            print('Outputs will be saved under', output_dir)
            
            with open(os.path.join(output_dir, 'structures_included.txt'), 'w') as f1:
                f1.write(' '.join(included))
                
            errors = [[None for _ in range(num_element)] for _ in range(num_element)]
            num_orbitals = [None for _ in range(num_element)]
            
            n = 0
            for s in included:
                g = f[s]
                
                H_pred = np.array(g['H_pred'])
                label = np.array(g['label'])
                node_attr = np.array(g['node_attr'])
                edge_index = np.array(g['edge_index'])
                
                # mask = np.array(g['mask'])
                
                error_pre = np.abs(H_pred - label)
                if mode in ('mse', 'rmse'):
                    error_pre = np.power(error_pre, 2)
                n += 1
        
                for i in range(num_element):
                    for j in range(num_element):
                        i_j = np.array([i, j])
                        selector = np.all(i_j[:, None] == node_attr[edge_index], axis=0)
                        error_pre_s = error_pre[selector]
                        error = np.mean(error_pre_s, axis=0)
                        # if mode == 'mae':
                        #     error = np.mean(abs_error_s, axis=0)
                        # elif mode == 'mse':
                        #     error = np.mean(np.power(abs_error_s, 2), axis=0)
                        # elif mode == 'maxe':
                        #     error = np.max(abs_error_s, axis=0)
                        # else:
                        #     raise ValueError()
                        
                        dummy_ix = np.where(selector)[0][0]
                        error = self.kernel.update_hopping({}, error[None, ...], node_attr, edge_index[:, dummy_ix, None], np.array([0]), debug=True)['0']
                        # error = self.kernel.update_hopping({}, mask[None, dummy_ix, ...], node_attr, edge_index[:, dummy_ix, None], np.array([0]), debug=True)['0']
                        if self.dataset_info.spinful:
                            error = error.real
                        
                        if errors[i][j] is None:
                            errors[i][j] = error
                        else:
                            errors[i][j] += error
                
        for i in range(num_element):
            for j in range(num_element):
                errors[i][j] /= n
                if mode == 'rmse':
                    errors[i][j] = np.sqrt(errors[i][j])
                errors[i][j] *= 1000 # convert to meV
        
        summary_info = f'Summary of prediction error\nType: {mode}, unit: meV\n\n'
        for i in range(num_element): 
            for j in range(num_element):
                Z1 = Z_to_chemsymbol(self.dataset_info.index_to_Z[i].item())
                Z2 = Z_to_chemsymbol(self.dataset_info.index_to_Z[j].item())
                error = errors[i][j]
                    
                summary_info += '\n'.join(('----------',
                                           'Atomic types: ' + Z1 + '-' + Z2,
                                           'Avg: ' + str(np.mean(error)),
                                           'Max: ' + str(np.amax(error)),
                                           'Min: ' + str(np.amin(error)),
                                           '\n'))
                
                savepath = os.path.join(output_dir, f'{mode.upper()}_{Z1}-{Z2}')
                save_error(savepath + '.csv', error)
                
                error_info.update(element_types=[[i], [j]],
                                  errors=error.tolist())
                with open(os.path.join(output_dir, f'error_info_{i}_{j}.json'), 'w') as f:
                    json.dump(error_info, f)
                
            num_orbitals[i] = error.shape[0] // (1+spinful)
        
        # = combined error plot =
        for i in range(num_element):
            errors[i] = np.concatenate(errors[i], axis=1)
        errors = np.concatenate(errors, axis=0)
        
        error_info.update(element_types=[list(range(num_element)) for _ in range(2)],
                          errors=errors.tolist())
        with open(os.path.join(output_dir, f'error_info_combined.json'), 'w') as f:
            json.dump(error_info, f)
        
        savepath = os.path.join(output_dir, f'{mode.upper()}_Summary')
        save_error(savepath + '.csv', errors)
        
        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write(summary_info)
        print('Written summary to summary.txt')
        
        return 0
        
    def error_analysis(self, include=[], exclude=[]):
        with h5py.File(self.h5file, 'r') as f:
            structures = list(f.keys())
            included = select_structures(structures, include, exclude)
            if len(included) == 0:
                print('No structures included')
                return 1
                         
            total_mse, total_mae, n = 0, 0, 0
            for s in included:
                g = f[s]
                H_pred = np.array(g['H_pred'])
                label = np.array(g['label'])
                mask = np.array(g['mask'])
                mse = maskmse(H_pred, label, mask)
                mae = maskmae(H_pred, label, mask)
                total_mse += mse
                total_mae += mae
                n += 1
                
            print(f'MSE is {total_mse / n}')
            print(f'MAE is {total_mae / n}')
            
        return 0
            
    def get_hamiltonian(self, stru_name):
        with h5py.File(self.h5file, 'r') as f:
            if not stru_name in f:
                print(f'Structure "{stru_name}" not found')
                return 1
            output_dir = os.path.join(self.output_dir, stru_name)
            os.makedirs(output_dir, exist_ok=True)
            
            g = f[stru_name]
            # stru = g['structure']

            # np.savetxt(os.path.join(output_dir, 'lat.dat'), np.transpose(stru['lat']))
            # np.savetxt(os.path.join(output_dir, 'element.dat'), stru['element'], fmt='%-3i')
            # np.savetxt(os.path.join(output_dir, 'site_positions.dat'), np.transpose(stru['sites']))
            # np.savetxt(os.path.join(output_dir, 'rlat.dat'), 2*np.pi*np.linalg.inv(stru['lat']))
            
            # with open(os.path.join(output_dir, 'orbital_types.dat'), 'w') as f:
            #     for i in g['node_attr']:
            #         f.write(' '.join(map(str, self.kernel.dataset_info.orbital_types[i])))
            #         f.write('\n')

            # info = {"isspinful": self.kernel.dataset_info.spinful}
            # with open(os.path.join(output_dir, 'info.json'), 'w') as f:
            #     json.dump(info, f)
                
            H_pred = np.array(g['H_pred'])
            # label = np.array(g['label'])
            node_attr = np.array(g['node_attr'])
            edge_index = np.array(g['edge_index'])
            edge_key = np.array(g['edge_key'])
            
        H_dict = self.kernel.update_hopping({}, H_pred, node_attr, edge_index, edge_key)
        with h5py.File(os.path.join(output_dir, f'hamiltonians_pred.h5'), 'w') as f:
            for k, v in H_dict.items():
                f[k] = v
                    
        return 0

def select_structures(all_structures, include, exclude):
    assert not (len(include) > 0 and len(exclude) > 0)
    included = []
    if len(include) > 0:
        for s in all_structures:
            for inc in include:
                if re.match(inc, s):
                    included.append(s)
    else:
        included = all_structures
        if len(exclude) > 0:
            for ix in range(len(included)-1, -1, -1):
                for exc in exclude:
                    if re.match(exc, included[ix]):
                        included.pop(ix)
                    
    print(f'Selected {len(included)} structures')
    # assert len(included) > 0, 'No structures included'
    
    return included

def maskmse(input, target, mask):
    assert input.shape == target.shape == mask.shape
    mse = np.power(np.abs(input - target), 2)
    mse = mse[mask].mean()
    return mse

def maskmae(input, target, mask):
    assert input.shape == target.shape == mask.shape
    mae = np.abs(input - target)
    mae = mae[mask].mean()
    return mae

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'process_data_tools/periodic_table.json'), 'r') as f:
    elem = json.load(f)
def Z_to_chemsymbol(Z: int):
    for k, v in elem.items():
        if v['Atomic no'] == Z:
            return k
        
def save_error(csv_file, error):
    # savepath.csv, savepath.png, savepath.svg
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('', *range(1, error.shape[1] + 1)))
        for ix, row in enumerate(error):
            writer.writerow((ix + 1, *row))
    print(f'Written error to csv file: {os.path.basename(csv_file)}')
    
    # https://www.matplotlib.org.cn/gallery/images_contours_and_fields/image_annotated_heatmap.html
    
def line_outside_axes(ax, start, end, **kwargs):
    # ref:
    # https://stackoverflow.com/questions/47597534/how-to-add-horizontal-lines-as-annotations-outside-of-axes-in-matplotlib
    # https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
    # if ax.transData is passed into transform, then the axes will automatically extend to include the datapoints, so we must use ax.transAxes instead
    kwargs.update(clip_on=False)
    transfunc = lambda x: ax.transAxes.inverted().transform(ax.transData.transform(x))
    ax.plot(*transfunc((start, end)).T, transform=ax.transAxes, **kwargs) 
    # clip_on=False, color='black', linewidth=1)
    