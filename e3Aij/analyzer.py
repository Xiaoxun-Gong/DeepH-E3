import os
import h5py
import numpy as np
import csv
import matplotlib.pyplot as plt

from .kernel import NetOutInfo, e3AijKernel

class testResultAnalyzer:
    
    def __init__(self, data_dir, output_dir):
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        self.kernel = e3AijKernel()
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
    
    def error_plot(self, stru_name, mode='mae'):
        os.makedirs(self.output_dir, exist_ok=True)
        
        with h5py.File(self.h5file, 'r') as f:
            if not stru_name in f:
                print(f'Structure "{stru_name}" not found')
                return
            g = f[stru_name]
            
            H_pred = np.array(g['H_pred'])
            label = np.array(g['label'])
            node_attr = np.array(g['node_attr'])
            edge_index = np.array(g['edge_index'])
            
            mask = np.array(g['mask'])
            
        abs_error = np.abs(H_pred - label)
        num_element = len(self.dataset_info.index_to_Z)
        for i in range(num_element):
            for j in range(num_element):
                Z1 = self.dataset_info.index_to_Z[i].item()
                Z2 = self.dataset_info.index_to_Z[j].item()
                i_j = np.array([i, j])
                selector = np.all(i_j[:, None] == node_attr[edge_index], axis=0)
                abs_error_s = abs_error[selector]
                if mode == 'mae':
                    error = np.mean(abs_error_s, axis=0)
                elif mode == 'mse':
                    error = np.mean(np.power(abs_error_s, 2), axis=0)
                elif mode == 'maxe':
                    error = np.max(abs_error_s, axis=0)
                else:
                    raise NotImplementedError(f'Unknown mode: {mode}')
                
                dummy_ix = np.where(selector)[0][0]
                error = self.kernel.update_hopping({}, error[None, ...], node_attr, edge_index[:, dummy_ix, None], np.array([0]), debug=True)['0']
                # error = self.kernel.update_hopping({}, mask[None, dummy_ix, ...], node_attr, edge_index[:, dummy_ix, None], np.array([0]), debug=True)['0']
                if self.dataset_info.spinful:
                    error = error.real
                    
                csv_file = f'{mode.upper()}_{Z1}-{Z2}.csv' 
                with open(os.path.join(self.output_dir, csv_file), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(('', *range(1, error.shape[1] + 1)))
                    for ix, row in enumerate(error):
                        writer.writerow((ix + 1, *row))
                print(f'Written error to csv file: {csv_file}')
                
                # from lihe
                plt.figure(figsize=(6, 6))
                plt.rcParams.update({'font.size': 16,
                                    'font.family': 'Arial',
                                    'mathtext.fontset': 'cm'}
                                    )
                im = plt.imshow(error * 1e3, cmap='Blues')
                plt.colorbar(im, shrink=0.45, label='(meV)')
                plt.xticks(range(error.shape[1]), range(1, 1 + error.shape[1]), fontsize=8)
                plt.yticks(range(error.shape[0]), range(1, 1 + error.shape[0]), fontsize=8)
                plt.xlabel(r'Orbital $\beta$')
                plt.ylabel(r'Orbital $\alpha$')
                plt.title(f'{mode.upper()}' + r' of $H^\prime_{i\alpha, j\beta}$' + f' ({Z1}-{Z2})')

                plt.tight_layout()

                figname = os.path.join(self.output_dir, f'{mode.upper()}_{Z1}-{Z2}')
                plt.savefig(f'{figname}.png', dpi=800)
                print(f'Saved figure {os.path.basename(figname)}.png')
                plt.savefig(f'{figname}.svg')
                print(f'Saved figure {os.path.basename(figname)}.svg')
            