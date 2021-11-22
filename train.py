import random
import shutil
import time
import os
import sys
import argparse
import json

import torch
import numpy as np
from torch import optim, nn
from torch.autograd.grad_mode import F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import SubsetRandomSampler, DataLoader

from torch.utils.tensorboard import SummaryWriter

from e3Aij import AijData, Collater, Net, LossRecord
from e3Aij.utils import Logger, RevertDecayLR, MaskMSELoss
from e3Aij.parse_configs import TrainConfig
from e3Aij.e3modules import e3TensorDecomp

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


# = record output =
os.makedirs(config.save_dir)
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
# print('\n------- Preparation of graph data -------')
edge_Aij = True
# dataset = AijData(
#     raw_data_dir=config.processed_data_dir,
#     graph_dir=config.graph_dir,
#     target=config.target,
#     dataset_name=config.dataset_name,
#     multiprocessing=False,
#     radius=config.cutoff_radius,
#     max_num_nbr=0,
#     edge_Aij=edge_Aij,
#     only_ij=config.only_ij,
#     default_dtype_torch=torch.get_default_dtype()
# )
dataset = AijData.from_existing_graph(config.graph_dir, config.torch_dtype)
spinful = dataset.info['spinful']
config.set_target(dataset.info['orbital_types'], dataset.info['index_to_Z'], spinful, os.path.join(config.save_dir, 'src/targets.txt'))
out_js_list, out_slices = dataset.set_mask(config.target_blocks, convert_to_net=config.convert_net_out)
construct_kernel = e3TensorDecomp(config.net_out_irreps, out_js_list, default_dtype_torch=torch.get_default_dtype(), spinful=spinful, if_sort=config.convert_net_out, device_torch=config.device)


print('\n------- Data loader for training -------')
# = data loader =
indices = list(range(len(dataset)))

if config.extra_val:
    extra_val_indices = []
    for extra_val_id in config.extra_val:
        ind = dataset.data.stru_id.index(extra_val_id)
        extra_val_indices.append(ind)
        indices.remove(ind)
    
dataset_size = len(indices)
train_size = int(config.train_ratio * dataset_size)
val_size = int(config.val_ratio * dataset_size)
test_size = int(config.test_ratio * dataset_size)
assert train_size + val_size + test_size <= dataset_size

np.random.shuffle(indices)
print(f'size of train set: {len(indices[:train_size])}')
print(f'size of val set: {len(indices[train_size:train_size + val_size])}')
print(f'size of test set: {len(indices[train_size + val_size:train_size + val_size + test_size])}')
print(f'Batch size: {config.batch_size}')
if config.extra_val:
    print(f'Additionally validating on {len(extra_val_indices)} structure(s)')

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
if config.extra_val:
    extra_val_loader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                sampler=SubsetRandomSampler(extra_val_indices),
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
        spinful=False,
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
    spinful={False},
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
scheduler = RevertDecayLR(net, optimizer, config.save_dir, config.revert_decay_patience, config.revert_decay_rate, config.torch_scheduler)

if config.checkpoint_dir:
    print(f'Loading from checkpoint at {config.checkpoint_dir}')
    checkpoint = torch.load(config.checkpoint_dir)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    with open(os.path.join(config.save_dir, 'tensorboard/info.json'), 'r') as f:
        global_step = json.load(f)['global_step'] + 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = config.lr
    # scheduler.decay_epoch = config.revert_decay_epoch
    # scheduler.decay_gamma = config.revert_decay_gamma
    print(f'Starting from epoch {checkpoint["epoch"]} with validation loss {checkpoint["val_loss"]}')
else:
    global_step = 0
    print('Starting new training process')
    

print('\n------- Begin training -------')
# = train and validation =
begin_time = time.time()
epoch_begin_time = time.time()
best_loss = 1e10

def val(val_loader):
    val_losses = LossRecord()
    loss_record_list = None
    if len(out_js_list) > 1:
        loss_record_list = [LossRecord() for _ in range(len(out_js_list))]
    with torch.no_grad():
        net.eval()
        for batch in val_loader:
            # get predicted H
            output, output_edge = net(batch.to(device=config.device))
            if config.convert_net_out:
                H_pred = output_edge
            else:
                H_pred = construct_kernel.get_H(output_edge)
            # get loss
            val_loss = criterion(H_pred, batch.label.to(device=config.device), batch.mask)
            val_losses.update(val_loss.item(), batch.num_edges)
            if len(out_js_list) > 1:
                for i in range(len(out_js_list)):
                    target_loss = criterion(H_pred[..., slice(out_slices[i], out_slices[i + 1])], 
                                            batch.label[..., slice(out_slices[i], out_slices[i + 1])].to(device=config.device),
                                            batch.mask[..., slice(out_slices[i], out_slices[i + 1])])
                    if spinful:
                        if config.convert_net_out:
                            num_hoppings = batch.mask[:, out_slices[i] * 4].sum() # ! this is not correct
                        else:
                            num_hoppings = batch.mask[:, 0, out_slices[i]].sum()
                    else:
                        num_hoppings = batch.mask[:, out_slices[i]].sum()
                    loss_record_list[i].update(target_loss.item(), num_hoppings)
    return val_losses, loss_record_list

epoch = scheduler.next_epoch # todo: this is not correct
learning_rate = optimizer.param_groups[0]['lr']
while epoch < config.num_epoch and learning_rate > config.min_lr:
    
    # = TRAIN =
    net.train()
    learning_rate = optimizer.param_groups[0]['lr']
    train_losses = LossRecord()
    for batch in train_loader:
        # get predicted H
        output, output_edge = net(batch.to(device=config.device))
        if config.convert_net_out:
            H_pred = output_edge
        else:
            H_pred = construct_kernel.get_H(output_edge)
        # get loss/home/gongxx/projects/DeepH/e3nn_DeepH
        loss = criterion(H_pred, batch.label.to(device=config.device), batch.mask)
        train_losses.update(loss.item(), batch.num_edges)
        # loss backward
        optimizer.zero_grad()
        loss.backward()
        # TODO clip_grad_norm_(net.parameters(), 4.2, error_if_nonfinite=True)
        optimizer.step()
        
    
    # = VALIDATION =
    val_losses, loss_record_list = val(val_loader)
    if config.extra_val:
        extra_val_losses, extra_loss_record_list = val(extra_val_loader)
    
    # = PRINT LOSS =
    time_r = round(time.time() - begin_time)
    d, h, m, s = time_r//86400, time_r%86400//3600, time_r%3600//60, time_r%60
    out_info = (f'Epoch #{epoch:<5d}  | '
                f'Time: {d:02d}d {h:02d}h {m:02d}m  | '
                f'LR: {learning_rate:.2e}  | '
                f'Batch time: {time.time() - epoch_begin_time:6.2f}  | '
                f'Train loss: {train_losses.avg:.2e}  | ' # :11.8f
                f'Val loss: {val_losses.avg:.2e}'
                )
    if config.extra_val:
        out_info += f'  | Extra val: {extra_val_losses.avg:0.2e}'
    if len(out_js_list) > 1:
        out_info = '====================\n' + out_info + '\n'
        loss_list = [loss_record_list[i].avg for i in range(len(out_js_list))]
        max_loss = max(loss_list)
        min_loss = min(loss_list)
        out_info += f'Target {loss_list.index(max_loss):03} has maximum loss {max_loss:.2e}; '
        out_info += f'Target {loss_list.index(min_loss):03} has minimum loss {min_loss:.2e}'
        if not config.simp_out:
            out_info += '\n'
            i = 0
            while i < len(out_js_list):
                out_info += f'Target {i:03}: {loss_record_list[i].avg:.2e}'
                if i % 5 == 4:
                    out_info += '\n'
                else:
                    out_info += ' \t|'
                i += 1
    print(out_info)
    
    # = TENSORBOARD =
    tb_writer.add_scalar('Learning rate', learning_rate, global_step=global_step)
    tb_writer.add_scalars('Loss', {'Train loss': train_losses.avg}, global_step=global_step)
    tb_writer.add_scalars('Loss', {'Validation loss': val_losses.avg}, global_step=global_step)
    if config.extra_val:
        tb_writer.add_scalars('Loss', {'Extra Validation': extra_val_losses.avg}, global_step=global_step)
    if len(out_js_list) > 1:
        tb_writer.add_scalars('Loss', {'Max loss': max_loss}, global_step=global_step)
        tb_writer.add_scalars('Loss', {'Min loss': min_loss}, global_step=global_step)
        tb_writer.add_scalars('Target losses', {'Validation loss': val_losses.avg}, global_step=global_step)
        for i in range(len(out_js_list)):
            tb_writer.add_scalars('Target losses', {f'Target {i} loss': loss_record_list[i].avg}, global_step=global_step)
    with open(os.path.join(config.save_dir, 'tensorboard/info.json'), 'w') as f:
        json.dump({'global_step': global_step}, f)
                
    # = write report =
    if val_losses.avg < best_loss:
        if len(out_js_list) > 1:
            best_loss = val_losses.avg
            target_loss_list = [(loss_record_list[i].avg, i) for i in range(len(loss_record_list))]
            target_loss_list.sort(key=lambda x: x[0], reverse=True)
            report = open(os.path.join(config.save_dir, 'report.txt'), 'w')
            print(f'Best model:', file=report)
            print(out_info, file=report)
            print('\n------- Detailed losses of each target -------', file=report)
            print('Losses are sorted in descending order', file=report)
            for i in range(len(out_js_list)):
                index_target = target_loss_list[i][1]
                print(f'\n======= No.{i:03}: Target {index_target:03} =======', file=report)
                print('Validation loss:           ', target_loss_list[i][0], file=report)
                print('Angular quantum numbers:   ', out_js_list[index_target], file=report)
                print('Target blocks:             ', config.target_blocks[index_target], file=report)
                print('Position in H matrix:      ', dataset.equivariant_blocks[index_target], file=report)
            report.close()
    
    # = save model, revert, etc. =
    scheduler.step(val_losses.avg)
    
    epoch_begin_time = time.time()
    epoch = scheduler.next_epoch
    global_step += 1