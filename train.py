import random
import shutil
import time
import os
import sys
import argparse
from configparser import ConfigParser

import torch
import numpy as np
from torch import optim, nn
from torch.autograd.grad_mode import F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import SubsetRandomSampler, DataLoader

from torch.utils.tensorboard import SummaryWriter

from e3Aij import AijData, Collater, Net, LossRecord
from e3Aij.utils import Logger, RevertDecayLR, MaskMSELoss, TrainConfig
from e3Aij.e3modules import construct_H, e3TensorDecomp


parser = argparse.ArgumentParser(description='Train e3Aij network')
parser.add_argument('--config', type=str, help='Config file for training')
args = parser.parse_args()

# torch.autograd.set_detect_anomaly(True)

config = TrainConfig(args.config)

# = random seed =
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
random.seed(config.seed)


# = default dtype =
torch.set_default_dtype(config.torch_dtype)


# = save to time folder =
config.save_dir = os.path.join(config.save_dir, str(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
config.save_dir = config.save_dir + '_' + config.additional_folder_name
assert not os.path.exists(config.save_dir)
os.makedirs(config.save_dir)


# = record output =
sys.stdout = Logger(os.path.join(config.save_dir, "result.txt"))
sys.stderr = Logger(os.path.join(config.save_dir, "stderr.txt"))


print('\n------- e3Aij model training begins -------')
print(f'Output will be stored under: {config.save_dir}')

# = save e3Aij script =
src = os.path.dirname(os.path.abspath(__file__))
dst = os.path.join(config.save_dir, 'src')
if config.checkpoint_dir:
    old_dir = os.path.dirname(config.checkpoint_dir)
    shutil.copytree(os.path.join(old_dir, 'src'), os.path.join(config.save_dir, 'src'))
    shutil.copytree(os.path.join(old_dir, 'tensorboard'), os.path.join(config.save_dir, 'tensorboard'))
    shutil.copyfile(os.path.join(old_dir, 'best_model.pkl'), os.path.join(config.save_dir, 'best_model.pkl'))
    dst = os.path.join(config.save_dir, 'src_restart')
os.makedirs(dst)
shutil.copyfile(os.path.join(src, 'train.py'), os.path.join(dst, 'train.py'))
shutil.copyfile(config.config_file, os.path.join(dst, 'train.ini'))
shutil.copytree(os.path.join(src, 'e3Aij'), os.path.join(dst, 'e3Aij_1'))
print('Saved e3Aij source code to output dir')


# = get graph data =
print('\n------- Preparation of graph data -------')
edge_Aij = True
dataset = AijData(
    raw_data_dir=config.processed_data_dir,
    graph_dir=config.graph_dir,
    target=config.target,
    dataset_name=config.dataset_name,
    multiprocessing=False,
    radius=config.cutoff_radius,
    max_num_nbr=0,
    edge_Aij=edge_Aij,
    only_ij=config.only_ij,
    default_dtype_torch=torch.get_default_dtype()
)
config.set_target(dataset.info['orbital_types'], dataset.info['index_to_Z'], os.path.join(config.save_dir, 'src/targets.txt'))
out_js_list, out_slices = dataset.set_mask(config.target_blocks)
construct_kernel = e3TensorDecomp(config.net_out_irreps, out_js_list, default_dtype_torch=torch.get_default_dtype(), device_torch=config.device)
# construct_kernel = construct_H(config.net_out_irreps, *out_js_list[0])


print('\n------- Data loader for training -------')
# = data loader =
dataset_size = len(dataset)
train_size = int(config.train_ratio * dataset_size)
val_size = int(config.val_ratio * dataset_size)
test_size = int(config.test_ratio * dataset_size)
assert train_size + val_size + test_size <= dataset_size

indices = list(range(dataset_size))
np.random.shuffle(indices)
print(f'size of train set: {len(indices[:train_size])}')
print(f'size of val set: {len(indices[train_size:train_size + val_size])}')
print(f'size of test set: {len(indices[train_size + val_size:train_size + val_size + test_size])}')

train_loader = DataLoader(dataset, 
                          batch_size=config.batch_size,
                          shuffle=False, 
                          sampler=SubsetRandomSampler(indices[:train_size]),
                          collate_fn=Collater(edge_Aij))
val_loader = DataLoader(dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        sampler=SubsetRandomSampler(indices[train_size:train_size + val_size]),
                        collate_fn=Collater(edge_Aij))
test_loader = DataLoader(dataset,
                         batch_size=config.batch_size,
                         shuffle=False,
                         sampler=SubsetRandomSampler(indices[train_size + val_size:train_size + val_size + test_size]),
                         collate_fn=Collater(edge_Aij))


# = build net =
begin = time.time()
print('\n------- Build model -------')
print('Building model...')
if config.checkpoint_dir:
    sys.path.append(os.path.join(os.path.dirname(config.checkpoint_dir), 'src'))
    from build_model import net
else:
    num_species = len(dataset.info["index_to_Z"])
    net = Net(
        num_species=num_species,
        irreps_embed_node=config.irreps_embed_node,
        irreps_edge_init=config.irreps_edge_init,
        irreps_sh=config.irreps_sh,
        irreps_mid_node=config.irreps_mid_node,
        irreps_post_node=config.irreps_post_node,
        irreps_out_node=config.irreps_out_node,
        irreps_mid_edge=config.irreps_mid_edge,
        irreps_post_edge=config.irreps_post_edge,
        irreps_out_edge=config.net_out_irreps,
        num_block=config.num_blocks,
        r_max=config.cutoff_radius,
        use_sc=True,
        only_ij=config.only_ij,
        if_sort_irreps=False
    )
    with open(os.path.join(config.save_dir, 'src/build_model.py'), 'w') as f:
        print(f'''from e3Aij_1 import Net
net = Net(
    num_species={num_species},
    irreps_embed_node='{config.irreps_embed_node}',
    irreps_edge_init='{config.irreps_edge_init}',
    irreps_sh='{config.irreps_sh}',
    irreps_mid_node='{config.irreps_mid_node}',
    irreps_post_node='{config.irreps_post_node}',
    irreps_out_node='{config.irreps_out_node}',
    irreps_mid_edge='{config.irreps_mid_edge}',
    irreps_post_edge='{config.irreps_post_edge}',
    irreps_out_edge='{config.net_out_irreps}',
    num_block={config.num_blocks},
    r_max={config.cutoff_radius},
    use_sc={True},
    only_ij={config.only_ij},
    if_sort_irreps={False}
)''', file=f)
print(f'Finished building model, cost {time.time() - begin:.2f} seconds.')
net.to(config.device)
model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("The model you built has %d parameters." % params)
print(net)


print('\n------- Preparation for training -------')
# = select optimizer =
model_parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(model_parameters, lr=config.lr, betas=(0.9, 0.999))
# model_parameters = filter(lambda p: p.requires_grad, net.parameters())
# optimizer_sgd = optim.SGD(model_parameters, lr=config.lr)
print('Using optimizer Adam')
criterion = MaskMSELoss()
print('Loss type: MSE over all matrix elements')
# = tensorboard =
tb_writer = SummaryWriter(os.path.join(config.save_dir, 'tensorboard'))
print('Tensorboard recorder initialized')

# = LR scheduler =
scheduler = RevertDecayLR(net, optimizer, config.save_dir, config.revert_decay_epoch, config.revert_decay_gamma, config.torch_scheduler)

if config.checkpoint_dir:
    print(f'Loading from checkpoint at {config.checkpoint_dir}')
    checkpoint = torch.load(config.checkpoint_dir)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch'] + 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = config.lr
    scheduler.decay_epoch = config.revert_decay_epoch
    scheduler.decay_gamma = config.revert_decay_gamma
    print(f'Starting from epoch {checkpoint["epoch"]} with validation loss {checkpoint["val_loss"]}')
else:
    epoch = 0
    print('Starting new training process')
    

print('\n------- Begin training -------')
# = train and validation =
begin_time = time.time()
epoch_begin_time = time.time()
while epoch < config.num_epoch:
    # if epoch == 500:
    #     scheduler.optimizer = optimizer_sgd
    #     optimizer = optimizer_sgd
    
    # = TRAIN =
    net.train()
    learning_rate = optimizer.param_groups[0]['lr']
    train_losses = LossRecord()
    for batch in train_loader:
        # get predicted H
        output, output_edge = net(batch.to(device=config.device))
        H_pred = construct_kernel.get_H(output_edge)
        # get loss
        # loss = criterion(H_pred, batch.label.to(device=config.device))
        # loss = sum([criterion(H_pred[:, slice(out_slices[i], out_slices[i + 1])], 
        #                       batch.label[:, slice(out_slices[i], out_slices[i + 1])].to(device=config.device), 
        #                       batch.mask[:, i])
        #             for i in range(len(out_js_list))])
        loss = criterion(H_pred, batch.label.to(device=config.device), batch.mask)
        train_losses.update(loss.item(), batch.num_edges)
        # loss backward
        optimizer.zero_grad()
        loss.backward()
        # TODO clip_grad_norm_(net.parameters(), 4.2, error_if_nonfinite=True)
        optimizer.step()
        
    
    # = VALIDATION =
    val_losses = LossRecord()
    if len(out_js_list) > 1:
        loss_record_list = [LossRecord() for _ in range(len(out_js_list))]
    with torch.no_grad():
        net.eval()
        for batch in val_loader:
            # get predicted H
            output, output_edge = net(batch.to(device=config.device))
            H_pred = construct_kernel.get_H(output_edge)
            # get loss
            # loss = criterion(H_pred, batch.label.to(device=config.device))
            val_loss = criterion(H_pred, batch.label.to(device=config.device), batch.mask)
            val_losses.update(val_loss.item(), batch.num_edges)
            if len(out_js_list) > 1:
                for i in range(len(out_js_list)):
                    target_loss = criterion(H_pred[:, slice(out_slices[i], out_slices[i + 1])], 
                                            batch.label[:, slice(out_slices[i], out_slices[i + 1])].to(device=config.device),
                                            batch.mask[:, slice(out_slices[i], out_slices[i + 1])])
                    loss_record_list[i].update(target_loss, batch.mask[:, out_slices[i]].sum())
    
    # = PRINT LOSS =
    time_r = round(time.time() - begin_time)
    d, h, m, s = time_r//86400, time_r%86400//3600, time_r%3600//60, time_r%60
    out_info = (f'Epoch #{epoch:<5d}  | '
                f'Time: {d:02d}d {h:02d}h {m:02d}m  | '
                f'LR: {learning_rate:0.2e}  | '
                f'Batch time: {time.time() - epoch_begin_time:6.2f}  | '
                f'Train loss: {train_losses.avg:11.8f}  | '
                f'Val loss: {val_losses.avg:11.8f}'
                )
    if len(out_js_list) > 1:
        out_info = '====================\n' + out_info + '\n'
        if config.simp_out:
            loss_list = [loss_record_list[i].avg for i in range(len(out_js_list))]
            max_loss = max(loss_list)
            min_loss = min(loss_list)
            out_info += f'Target {loss_list.index(max_loss):03} has maximum loss {max_loss:.8f}; '
            out_info += f'Target {loss_list.index(min_loss):03} has minimum loss {min_loss:.8f}'
        else:
            i = 0
            while i < len(out_js_list):
                out_info += f'Target {i:03} loss: {loss_record_list[i].avg:.8f}'
                if i % 4 == 3:
                    out_info += '\n'
                else:
                    out_info += ' \t|'
                i += 1
    print(out_info)
    
    # = TENSORBOARD =
    tb_writer.add_scalar('Learning rate', learning_rate, global_step=epoch)
    tb_writer.add_scalars('loss', {'Train loss': train_losses.avg}, global_step=epoch)
    tb_writer.add_scalars('loss', {'Validation loss': val_losses.avg}, global_step=epoch)
    if len(out_js_list) > 1:
        if config.simp_out:
            tb_writer.add_scalars('loss', {'Max loss': max_loss}, global_step=epoch)
            tb_writer.add_scalars('loss', {'Min loss': min_loss}, global_step=epoch)
        else:
            for i in range(len(out_js_list)):
                tb_writer.add_scalars('loss', {f'Target {i} loss': loss_record_list[i].avg}, global_step=epoch)
    
    # = save model, revert, etc. =
    scheduler.step(epoch, val_losses.avg)
    
    epoch_begin_time = time.time()
    epoch += 1