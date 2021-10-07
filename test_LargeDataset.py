import random
import shutil
import time
import os
import sys

import torch
import numpy as np
from torch import optim, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import SubsetRandomSampler, DataLoader

from torch.utils.tensorboard import SummaryWriter

from e3Aij import AijData, Collater, Net, LossRecord, Rotate
from e3Aij.utils import e3TensorDecomp, Logger, RevertDecayLR



device = torch.device('cuda')
seed = 42
torch_dtype = torch.float32
# net_out_irreps = '4x1o+4x2o+4x3o'
net_out_irreps = '4x0e+4x1e+4x2e'
batch_size = 4
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2
lr = 0.001
num_epoch = 3000
save_dir = '/home/gongxx/projects/DeepH/e3nn_DeepH/test_runs/1006_first_largeData'
additional_folder_name = 'continue'
# ! set_mask
checkpoint_dir = '/home/gongxx/projects/DeepH/e3nn_DeepH/test_runs/1006_first_largeData/2021-10-06_20-10-07_1x1/best_model.pkl'


# = random seed =
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)


# = default dtype =
torch.set_default_dtype(torch_dtype)


# = save to time folder =
save_dir = os.path.join(save_dir, str(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
save_dir = save_dir + '_' + additional_folder_name
assert not os.path.exists(save_dir)
os.makedirs(save_dir)


# = record output =
sys.stdout = Logger(os.path.join(save_dir, "result.txt"))
sys.stderr = Logger(os.path.join(save_dir, "stderr.txt"))


# = get graph data =
edge_Aij = True
dataset = AijData(
    raw_data_dir="/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/1004_MoS2/processed",
    graph_dir="/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/1004_MoS2/graph_data",
    target="hamiltonian",
    dataset_name="MoS2",
    multiprocessing=False,
    # radius=5.0,
    radius=7.2,
    max_num_nbr=0,
    edge_Aij=edge_Aij,
    default_dtype_torch=torch.get_default_dtype()
)
# out_js_list = dataset.set_mask([{"42 42": [3, 5], "42 16": [3, 4], "16 42": [2, 5], "16 16": [2, 4]}])
out_js_list = dataset.set_mask([{"42 42": [3, 3], "42 16": [3, 2], "16 42": [2, 3], "16 16": [2, 2]}])
construct_kernel = e3TensorDecomp(net_out_irreps, *out_js_list[0], default_dtype_torch=torch.get_default_dtype(), device_torch=device)


# = save e3Aij =
src = os.path.dirname(os.path.abspath(__file__))
dst = os.path.join(save_dir, 'src')
os.makedirs(dst)
shutil.copyfile(os.path.join(src, 'test_LargeDataset.py'), os.path.join(dst, 'test_LargeDataset.py'))
shutil.copytree(os.path.join(src, 'e3Aij'), os.path.join(dst, 'e3Aij'))


# = tensorboard =
tb_writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))


# = data loader =
dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = int(test_ratio * dataset_size)
assert train_size + val_size + test_size <= dataset_size

indices = list(range(dataset_size))
np.random.shuffle(indices)
print(f'size of train set: {len(indices[:train_size])}')
print(f'size of val set: {len(indices[train_size:train_size + val_size])}')
print(f'size of test set: {len(indices[train_size + val_size:train_size + val_size + test_size])}')

train_loader = DataLoader(dataset, 
                          batch_size=batch_size,
                          shuffle=False, 
                          sampler=SubsetRandomSampler(indices[:train_size]),
                          collate_fn=Collater(edge_Aij))
val_loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        sampler=SubsetRandomSampler(indices[train_size:train_size + val_size]),
                        collate_fn=Collater(edge_Aij))
test_loader = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         sampler=SubsetRandomSampler(indices[train_size + val_size:train_size + val_size + test_size]),
                         collate_fn=Collater(edge_Aij))


# = build net =
num_species = len(dataset.info["index_to_Z"])

begin = time.time()
print('Building model...')
net = Net(
    num_species=num_species,
    irreps_embed_node="32x0e",
    irreps_edge_init="64x0e",
    irreps_sh='1x0e + 1x1o + 1x2e + 1x3o + 1x4e + 1x5o + 1x6e',
    irreps_mid_node='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o',#'16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+4x4e+4x4o+4x5e+4x5o+4x6e+4x6o',
    irreps_post_node='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o',
    irreps_out_node="1x0e",
    irreps_mid_edge='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o',
    irreps_post_edge='16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o',
    irreps_out_edge=net_out_irreps,
    num_block=3,
    use_sc=False,
    r_max = 7.4,
    if_sort_irreps=False
)
net.to(device)
print(f'Finished building model, cost {time.time() - begin:.2f} seconds.')

model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("The model you built has %d parameters." % params)


# = select optimizer =
model_parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(model_parameters, lr=lr, betas=(0.9, 0.999))
criterion = nn.MSELoss()

# = LR scheduler =
# scheduler = RevertDecayLR(net, optimizer,  save_dir, [250], [0.2])
scheduler = RevertDecayLR(net, optimizer,  save_dir, [250], [0.2])

if checkpoint_dir:
    print(f'Load from checkpoint at {checkpoint_dir}')
    checkpoint = torch.load(checkpoint_dir)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print(f'Starting from epoch {epoch} with validation loss {checkpoint["val_loss"]}.')
else:
    epoch = 0
    

print('Begin training...')
# = train and validation =
begin_time = time.time()
while epoch < num_epoch:    
    # = TRAIN =
    net.train()
    learning_rate = optimizer.param_groups[0]['lr']
    train_losses = LossRecord()
    for batch in train_loader:
        # get predicted H
        output, output_edge = net(batch.to(device=device))
        H_pred = construct_kernel.get_H(output_edge)
        # get loss
        loss = criterion(H_pred, batch.label.to(device=device))
        train_losses.update(loss.item(), batch.num_edges)
        # loss backward
        optimizer.zero_grad()
        loss.backward()
        # TODO clip_grad_norm_(net.parameters(), 4.2, error_if_nonfinite=True)
        optimizer.step()
        
    
    # = VALIDATION =
    val_losses = LossRecord()
    with torch.no_grad():
        # TODO: loss_each_out if out_fea_len > 1
        net.eval()
        for batch in val_loader:
            # get predicted H
            output, output_edge = net(batch.to(device=device))
            H_pred = construct_kernel.get_H(output_edge)
            # get loss
            loss = criterion(H_pred, batch.label.to(device=device))
            val_losses.update(loss.item(), batch.num_edges)
    
    # = PRINT LOSS =
    print(f'Epoch #{epoch:01d} \t| '
          f'Learning rate: {learning_rate:0.2e} \t| '
          f'Batch time: {time.time() - begin_time:.2f} \t| '
          f'Train loss: {train_losses.avg:.8f} \t| '
          f'Val loss: {val_losses.avg:.8f} \t| '
         )
    
    # = TENSORBOARD =
    tb_writer.add_scalar('Learning rate', learning_rate, global_step=epoch)
    tb_writer.add_scalars('loss', {'Train loss': train_losses.avg}, global_step=epoch)
    tb_writer.add_scalars('loss', {'Validation loss': val_losses.avg}, global_step=epoch)
    
    # = save model, revert, etc. =
    scheduler.step(epoch, val_losses.avg)
    
    begin_time = time.time()
    epoch += 1

# ! torch.save(net.state_dict(), '/home/gongxx/projects/DeepH/e3nn_DeepH/test_runs/0930_e3td_1x1/model1.pkl')