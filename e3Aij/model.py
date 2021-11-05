import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import degree
from e3nn.nn import FullyConnectedNet, Gate, BatchNorm, NormActivation, Extract
from e3nn.o3 import TensorProduct, Irrep, Irreps, Linear, SphericalHarmonics, FullyConnectedTensorProduct
from .from_nequip.cutoffs import PolynomialCutoff
from .from_nequip.radial_basis import BesselBasis
from .from_nequip.tp_utils import tp_path_exists
from .from_schnetpack.acsf import GaussianBasis
from .e3modules import sort_irreps, e3LayerNorm, e3ElementWise


epsilon = 1e-8

def get_gate_nonlin(irreps_in1, irreps_in2, irreps_out, 
                    act = {1: torch.nn.functional.silu, -1: torch.tanh}, 
                    act_gates  = {1: torch.sigmoid, -1: torch.tanh}
                    ):
    # get gate nonlinearity after tensor product
    # irreps_in1 and irreps_in2 are irreps to be multiplied in tensor product
    # irreps_out is desired irreps after gate nonlin
    # notice that nonlin.irreps_out might not be exactly equal to irreps_out
    
    irreps_scalars = Irreps([
        (mul, ir)
        for mul, ir in irreps_out
        if ir.l == 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ]).simplify()
    irreps_gated = Irreps([
        (mul, ir)
        for mul, ir in irreps_out
        if ir.l > 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ]).simplify()
    if irreps_gated.dim > 0:
        if tp_path_exists(irreps_in1, irreps_in2, "0e"):
            ir = "0e"
        elif tp_path_exists(irreps_in1, irreps_in2, "0o"):
            ir = "0o"
        else:
            raise ValueError(
                f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} is unable to produce gates needed for irreps_gated={irreps_gated}")
    else:
        ir = None
    irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

    gate_nonlin = Gate(
        irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
        irreps_gated  # gated tensors
    )
    
    return gate_nonlin


class EquiConv(nn.Module):
    def __init__(self, fc_len_in, irreps_in1, irreps_in2, irreps_out, norm='', nonlin=True,
                 weight_layers=1, weight_neurons=32,
                 act = {1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates = {1: torch.sigmoid, -1: torch.tanh}
                 ):
        super(EquiConv, self).__init__()
        
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)
        
        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out, act, act_gates)
            irreps_tp_out = self.nonlin.irreps_in
        else:
            irreps_tp_out = Irreps([(mul, ir) for mul, ir in irreps_out if tp_path_exists(irreps_in1, irreps_in2, ir)])
        
        self.tp = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_tp_out, 
                                              shared_weights=True, internal_weights=True)
        
        if nonlin:
            self.cfconv = e3ElementWise(self.nonlin.irreps_out)
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.cfconv = e3ElementWise(irreps_tp_out)
            self.irreps_out = irreps_tp_out
        
        # fully connected net to create tensor product weights
        self.fc = nn.Sequential(nn.Linear(fc_len_in, 64),
                                nn.SiLU(),
                                nn.Linear(64, 64),
                                nn.SiLU(),
                                nn.Linear(64, self.cfconv.len_weight)
                                )

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.cfconv.irreps_in)
            else:
                raise ValueError(f'unknown norm: {norm}')

    def forward(self, fea_in1, fea_in2, fea_weight, batch_edge):
        z = self.tp(fea_in1, fea_in2)

        if self.nonlin is not None:
            z = self.nonlin(z)

        weight = self.fc(fea_weight)
        z = self.cfconv(z, weight)

        if self.norm is not None:
            z = self.norm(z, batch_edge)

        # TODO self-connection here
        return z


class NodeUpdateBlock(nn.Module):
    def __init__(self, num_species, fc_len_in, irreps_sh, irreps_in_node, irreps_out_node, irreps_in_edge,
                 act, act_gates, use_sc=True, concat=True, only_ij=False, nonlin=False, norm='e3LayerNorm', if_sort_irreps=False):
        super(NodeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_sh = Irreps(irreps_sh)
        irreps_out_node = Irreps(irreps_out_node)
        irreps_in_edge = Irreps(irreps_in_edge)

        if concat:
            irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
            if if_sort_irreps:
                self.sort = sort_irreps(irreps_in1)
                irreps_in1 = self.sort.irreps_out
        else:
            irreps_in1 = irreps_in_node
        irreps_in2 = irreps_sh

        self.lin_pre = Linear(irreps_in=irreps_in_node, irreps_out=irreps_in_node)
        
        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_node)
            irreps_conv_out = self.nonlin.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_node
            conv_nonlin = True
            
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_nonlin, act=act, act_gates=act_gates)
        self.lin_post = Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out)
        
        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out
        
        self.sc = None
        if use_sc:
            self.sc = FullyConnectedTensorProduct(irreps_in_node, f'{num_species}x0e', self.conv.irreps_out)
            # self.sc = FullyConnectedTensorProduct(irreps_in_node, f'{num_species}x0e', self.irreps_out)
            
        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')

        self.irreps_in_node = irreps_in_node
        self.use_sc = use_sc
        self.concat = concat
        self.only_ij = only_ij
        self.if_sort_irreps = if_sort_irreps


    def forward(self, node_fea, node_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch, selfloop_edge):
        if self.use_sc:
            node_self_connection = self.sc(node_fea, node_one_hot)

        node_fea = self.lin_pre(node_fea)

        index_i = edge_index[0]
        index_j = edge_index[1]
        if self.concat:
            fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
            if self.if_sort_irreps:
                fea_in = self.sort(fea_in)
            edge_update = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])
        else:
            edge_update = self.conv(node_fea[index_j], edge_sh, edge_length_embedded, batch[edge_index[0]])
        
        # reduce=mean to normalize according to number of node neighbors
        node_fea = scatter(edge_update, index_i, dim=0, dim_size=node_fea.shape[0], reduce='mean')
        if self.only_ij:
            node_fea = node_fea + scatter(edge_update[~selfloop_edge], index_j[~selfloop_edge], dim=0, dim_size=node_fea.shape[0], reduce='mean')
            
        node_fea = self.lin_post(node_fea)
            
        if self.use_sc:
            node_fea = node_fea + node_self_connection
            
        if self.nonlin is not None:
            node_fea = self.nonlin(node_fea)
            
        # TODO another linear layer here
        
        if self.norm is not None:
            node_fea = self.norm(node_fea, batch)
            
        # TODO another nonlin here

        return node_fea


class EdgeUpdateBlock(nn.Module):
    def __init__(self, num_species, fc_len_in, irreps_sh, irreps_in_node, irreps_in_edge, irreps_out_edge,
                 act, act_gates, use_sc=True, init_edge=False, nonlin=False, norm='e3LayerNorm', if_sort_irreps=False):
        super(EdgeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_in_edge = Irreps(irreps_in_edge)
        irreps_out_edge = Irreps(irreps_out_edge)

        irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
        if if_sort_irreps:
            self.sort = sort_irreps(irreps_in1)
            irreps_in1 = self.sort.irreps_out
        irreps_in2 = irreps_sh

        self.lin_pre = Linear(irreps_in=irreps_in_edge, irreps_out=irreps_in_edge)
        
        self.nonlin = None
        self.lin_post = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_edge)
            irreps_conv_out = self.nonlin.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_edge
            conv_nonlin = True
            
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_nonlin, act=act, act_gates=act_gates)
        self.lin_post = Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out)
        
        if use_sc:
            self.sc = FullyConnectedTensorProduct(irreps_in_edge, f'{num_species**2}x0e', self.conv.irreps_out)
            # self.sc = FullyConnectedTensorProduct(irreps_in_edge, f'{num_species**2}x0e', self.irreps_out)

        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')
            
        self.use_sc = use_sc
        self.init_edge = init_edge
        self.if_sort_irreps = if_sort_irreps
        self.irreps_in_edge = irreps_in_edge


    def forward(self, node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch):
        if not self.init_edge:
            if self.use_sc:
                edge_self_connection = self.sc(edge_fea, edge_one_hot)
            edge_fea = self.lin_pre(edge_fea)
            
        index_i = edge_index[0]
        index_j = edge_index[1]
        fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
        if self.if_sort_irreps:
            fea_in = self.sort(fea_in)
        edge_fea = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])
        
        edge_fea = self.lin_post(edge_fea)

        if self.use_sc:
            edge_fea = edge_fea + edge_self_connection
            
        if self.nonlin is not None:
            edge_fea = self.nonlin(edge_fea)

        if self.norm is not None:
            edge_fea = self.norm(edge_fea, batch[edge_index[0]])

        return edge_fea


class Net(nn.Module):
    def __init__(self, num_species, irreps_embed_node, irreps_edge_init, irreps_sh, irreps_mid_node, 
                 irreps_post_node, irreps_out_node,irreps_mid_edge, irreps_post_edge, irreps_out_edge, 
                 num_block, r_max, use_sc=True, only_ij=False, num_basis=128,
                 act={1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates={1: torch.sigmoid, -1: torch.tanh},
                 if_sort_irreps=False):
        super(Net, self).__init__()
        self.num_species = num_species
        self.only_ij = only_ij
        
        irreps_embed_node = Irreps(irreps_embed_node)
        assert irreps_embed_node == Irreps(f'{irreps_embed_node.dim}x0e')
        self.embedding = Linear(irreps_in=f"{num_species}x0e", irreps_out=irreps_embed_node)

        # edge embedding for tensor product weight
        # self.basis = BesselBasis(r_max, num_basis=num_basis, trainable=True)
        # self.cutoff = PolynomialCutoff(r_max, p=6)
        self.basis = GaussianBasis(start=0.0, stop=r_max, n_gaussians=num_basis, trainable=False)
        
        # distance expansion to initialize edge feature
        irreps_edge_init = Irreps(irreps_edge_init)
        assert irreps_edge_init == Irreps(f'{irreps_edge_init.dim}x0e')
        self.distance_expansion = GaussianBasis(
            start=0.0, stop=6.0, n_gaussians=irreps_edge_init.dim, trainable=False
        )

        self.irreps_sh = irreps_sh
        self.sh = SphericalHarmonics(
            irreps_out=irreps_sh,
            normalize=True,
            normalization='component',
        )

        # self.edge_update_block_init = EdgeUpdateBlock(num_basis, irreps_sh, self.embedding.irreps_out, None, irreps_mid_edge, act, act_gates, False, init_edge=True)
        irreps_node_prev = self.embedding.irreps_out
        irreps_edge_prev = irreps_edge_init

        self.node_update_blocks = nn.ModuleList([])
        self.edge_update_blocks = nn.ModuleList([])
        for index_block in range(num_block):
            if index_block == num_block - 1:
                node_update_block = NodeUpdateBlock(num_species, num_basis, irreps_sh, irreps_node_prev, irreps_post_node, irreps_edge_prev, act, act_gates, use_sc, only_ij=only_ij, if_sort_irreps=if_sort_irreps)
                edge_update_block = EdgeUpdateBlock(num_species, num_basis, irreps_sh, node_update_block.irreps_out, irreps_edge_prev, irreps_post_edge, act, act_gates, use_sc, if_sort_irreps=if_sort_irreps)
            else:
                node_update_block = NodeUpdateBlock(num_species, num_basis, irreps_sh, irreps_node_prev, irreps_mid_node, irreps_edge_prev, act, act_gates, use_sc, only_ij=only_ij, if_sort_irreps=if_sort_irreps)
                edge_update_block = EdgeUpdateBlock(num_species, num_basis, irreps_sh, node_update_block.irreps_out, irreps_edge_prev, irreps_mid_edge, act, act_gates, use_sc, if_sort_irreps=if_sort_irreps)
            irreps_node_prev = node_update_block.irreps_out
            irreps_edge_prev = edge_update_block.irreps_out
            self.node_update_blocks.append(node_update_block)
            self.edge_update_blocks.append(edge_update_block)
        
        for _, ir in Irreps(irreps_out_edge):
            assert ir in irreps_edge_prev, f'required ir {ir} cannot be produced by convolution'

        self.irreps_out_node = irreps_out_node
        self.irreps_out_edge = irreps_out_edge
        self.lin_node = Linear(irreps_in=irreps_node_prev, irreps_out=irreps_out_node)
        self.lin_edge = Linear(irreps_in=irreps_edge_prev, irreps_out=irreps_out_edge)

    def forward(self, data):
        node_one_hot = F.one_hot(data.x, num_classes=self.num_species).type(torch.get_default_dtype())
        edge_one_hot = F.one_hot(self.num_species * data.x[data.edge_index[0]] + data.x[data.edge_index[1]],
                                 num_classes=self.num_species**2).type(torch.get_default_dtype()) # ! might not be good if dataset has many elements
        
        node_fea = self.embedding(node_one_hot)

        edge_vec = data["edge_attr"][:, [2, 3, 1]] # (y, z, x) order

        edge_sh = self.sh(edge_vec).type(torch.get_default_dtype())
        # edge_length_embedded = (self.basis(data["edge_attr"][:, 0] + epsilon) * self.cutoff(data["edge_attr"][:, 0])[:, None]).type(torch.get_default_dtype())
        edge_length_embedded = self.basis(data['edge_attr'][:, 0])
        
        selfloop_edge = None
        if self.only_ij:
            selfloop_edge = torch.abs(data["edge_attr"][:, 0]) < 1e-7

        # edge_fea = self.edge_update_block_init(node_fea, edge_sh, None, edge_length_embedded, data["edge_index"])
        edge_fea = self.distance_expansion(data['edge_attr'][:, 0]).type(torch.get_default_dtype())
        # for edge_fea_sing, distance_sing in zip(edge_fea, data['edge_attr'][:, 0]):
        #     print(distance_sing)
        #     print(edge_fea_sing)
        # exit()
        for node_update_block, edge_update_block in zip(self.node_update_blocks, self.edge_update_blocks):
            node_fea = node_update_block(node_fea, node_one_hot, edge_sh, edge_fea, edge_length_embedded, data["edge_index"], data.batch, selfloop_edge)
            edge_fea = edge_update_block(node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, data["edge_index"], data.batch)

        node_fea = self.lin_node(node_fea)
        edge_fea = self.lin_edge(edge_fea)
        return node_fea, edge_fea

    def __repr__(self):
        info = '===== e3Aij model structure: ====='
        info += f'\nusing spherical harmonics: {self.irreps_sh}'
        for index, (nupd, eupd) in enumerate(zip(self.node_update_blocks, self.edge_update_blocks)):
            info += f'\n=== layer {index} ==='
            info += f'\nnode update: ({nupd.irreps_in_node} -> {nupd.irreps_out})'
            info += f'\nedge update: ({eupd.irreps_in_edge} -> {eupd.irreps_out})'
        info += '\n=== output info ==='
        info += f'\noutput node: ({self.irreps_out_node})'
        info += f'\noutput edge: ({self.irreps_out_edge})'
        
        return info