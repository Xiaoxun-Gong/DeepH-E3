import warnings
import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
from torch_scatter import scatter
from e3nn.nn import FullyConnectedNet, Gate, BatchNorm, NormActivation, Extract
from e3nn.o3 import TensorProduct, Irrep, Irreps, Linear, SphericalHarmonics, FullyConnectedTensorProduct
from .from_nequip.cutoffs import PolynomialCutoff
from .from_nequip.radial_basis import BesselBasis
from .from_nequip.tp_utils import tp_path_exists
from .from_schnetpack.acsf import GaussianBasis
from torch_geometric.nn.models.dimenet import BesselBasisLayer
from .e3modules import sort_irreps, e3LayerNorm, e3ElementWise
from .e3modules import cplxLinear, cplxE3Linear, cplxFullyConnectedTensorProduct, get_complex_activation, CSiLU_module
from .utils import flt2cplx


epsilon = 1e-8


def get_gate_nonlin(irreps_in1, irreps_in2, irreps_out, is_complex=False,
                    act={1: torch.nn.functional.silu, -1: torch.tanh}, 
                    act_gates={1: torch.sigmoid, -1: torch.tanh}
                    ):
    # get gate nonlinearity after tensor product
    # irreps_in1 and irreps_in2 are irreps to be multiplied in tensor product
    # irreps_out is desired irreps after gate nonlin
    # notice that nonlin.irreps_out might not be exactly equal to irreps_out
    if is_complex:
        act_tmp, act_gates_tmp = act, act_gates
        act, act_gates = {}, {}
        for k, v in act_tmp.items():
            act[k] = get_complex_activation(v)
        for k, v in act_gates_tmp.items():
            act_gates[k] = get_complex_activation(v)
            
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
            warnings.warn('Using odd representations as gates')
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


class SkipConnection(nn.Module):
    def __init__(self, irreps_in, irreps_out, is_complex=False):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)
        self.sc = None
        if irreps_in == irreps_out:
            self.sc = None
        else:
            self.sc = cplxE3Linear(irreps_in=irreps_in, irreps_out=irreps_out, is_complex=is_complex)
    
    def forward(self, old, new):
        if self.sc is not None:
            old = self.sc(old)
        
        return old + new
        

class EquiConv(nn.Module):
    def __init__(self, fc_len_in, irreps_in1, irreps_in2, irreps_out, norm='', nonlin=True, is_complex=False,
                 act = {1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates = {1: torch.sigmoid, -1: torch.tanh}
                 ):
        super(EquiConv, self).__init__()
        
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)
        
        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out, is_complex, act, act_gates)
            irreps_tp_out = self.nonlin.irreps_in
        else:
            irreps_tp_out = Irreps([(mul, ir) for mul, ir in irreps_out if tp_path_exists(irreps_in1, irreps_in2, ir)])
        
        self.tp = cplxFullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_tp_out, is_complex=is_complex,
                                              shared_weights=True, internal_weights=True)
        
        if nonlin:
            self.cfconv = e3ElementWise(self.nonlin.irreps_out)
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.cfconv = e3ElementWise(irreps_tp_out)
            self.irreps_out = irreps_tp_out
        
        # fully connected net to create tensor product weights
        if is_complex:
            linear_act = CSiLU_module()
        else:
            linear_act = nn.SiLU()
        self.fc = nn.Sequential(cplxLinear(fc_len_in, 64, is_complex=is_complex),
                                linear_act,
                                cplxLinear(64, 64, is_complex=is_complex),
                                linear_act,
                                cplxLinear(64, self.cfconv.len_weight, is_complex=is_complex)
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
                 act, act_gates, use_sc=True, concat=True, only_ij=False, nonlin=False, norm='e3LayerNorm', is_complex=False, if_sort_irreps=False):
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

        self.lin_pre = cplxE3Linear(irreps_in=irreps_in_node, irreps_out=irreps_in_node, is_complex=is_complex)
        
        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_node, is_complex, act, act_gates)
            irreps_conv_out = self.nonlin.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_node
            conv_nonlin = True
            
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_nonlin, is_complex=is_complex, act=act, act_gates=act_gates)
        self.lin_post = cplxE3Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, is_complex=is_complex)
        
        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out
        
        self.sc = None
        if use_sc:
            self.sc = cplxFullyConnectedTensorProduct(irreps_in_node, f'{num_species}x0e', self.conv.irreps_out, is_complex=is_complex)
            # self.sc = FullyConnectedTensorProduct(irreps_in_node, f'{num_species}x0e', self.irreps_out)
            
        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')
        
        self.skip_connect = SkipConnection(irreps_in_node, self.irreps_out, is_complex)

        self.irreps_in_node = irreps_in_node
        self.use_sc = use_sc
        self.concat = concat
        self.only_ij = only_ij
        self.if_sort_irreps = if_sort_irreps
        self.is_complex = is_complex


    def forward(self, node_fea, node_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch, selfloop_edge, edge_length):
        if self.is_complex:
            node_fea = node_fea.type(flt2cplx(torch.get_default_dtype()))
            node_one_hot = node_one_hot.type(flt2cplx(torch.get_default_dtype()))
            edge_sh = edge_sh.type(flt2cplx(torch.get_default_dtype()))
            edge_fea = edge_fea.type(flt2cplx(torch.get_default_dtype()))
            edge_length_embedded = edge_length_embedded.type(flt2cplx(torch.get_default_dtype()))
            
        node_fea_old = node_fea
        
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
        
        # sigma = 3
        # n = 2
        # edge_update = edge_update * torch.exp(- edge_length ** n / sigma ** n / 2).view(-1, 1)
        
        # todo: reduce=mean to normalize according to number of node neighbors
        node_fea = scatter(edge_update, index_i, dim=0, dim_size=node_fea.shape[0], reduce='add')
        if self.only_ij:
            node_fea = node_fea + scatter(edge_update[~selfloop_edge], index_j[~selfloop_edge], dim=0, dim_size=node_fea.shape[0], reduce='add')
            
        node_fea = self.lin_post(node_fea)
            
        if self.use_sc:
            node_fea = node_fea + node_self_connection
            
        if self.nonlin is not None:
            node_fea = self.nonlin(node_fea)
            
        # TODO another linear layer here
        
        if self.norm is not None:
            node_fea = self.norm(node_fea, batch)
            
        # TODO another nonlin here
        node_fea = self.skip_connect(node_fea_old, node_fea)
        
        return node_fea


class EdgeUpdateBlock(nn.Module):
    def __init__(self, num_species, fc_len_in, irreps_sh, irreps_in_node, irreps_in_edge, irreps_out_edge,
                 act, act_gates, use_sc=True, init_edge=False, nonlin=False, norm='e3LayerNorm', 
                 is_complex=False, if_sort_irreps=False):
        super(EdgeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_in_edge = Irreps(irreps_in_edge)
        irreps_out_edge = Irreps(irreps_out_edge)

        irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
        if if_sort_irreps:
            self.sort = sort_irreps(irreps_in1)
            irreps_in1 = self.sort.irreps_out
        irreps_in2 = irreps_sh

        self.lin_pre = cplxE3Linear(irreps_in=irreps_in_edge, irreps_out=irreps_in_edge, is_complex=is_complex)
        
        self.nonlin = None
        self.lin_post = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_edge, is_complex, act, act_gates)
            irreps_conv_out = self.nonlin.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_edge
            conv_nonlin = True
            
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_nonlin, is_complex=is_complex, act=act, act_gates=act_gates)
        self.lin_post = cplxE3Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, is_complex=is_complex)
        
        if use_sc:
            self.sc = cplxFullyConnectedTensorProduct(irreps_in_edge, f'{num_species**2}x0e', self.conv.irreps_out, is_complex=is_complex)
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
        
        self.skip_connect = SkipConnection(irreps_in_edge, self.irreps_out, is_complex)
            
        self.use_sc = use_sc
        self.init_edge = init_edge
        self.if_sort_irreps = if_sort_irreps
        self.irreps_in_edge = irreps_in_edge
        self.is_complex = is_complex


    def forward(self, node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch):
        if self.is_complex:
            node_fea = node_fea.type(flt2cplx(torch.get_default_dtype()))
            edge_one_hot = edge_one_hot.type(flt2cplx(torch.get_default_dtype()))
            edge_sh = edge_sh.type(flt2cplx(torch.get_default_dtype()))
            edge_fea = edge_fea.type(flt2cplx(torch.get_default_dtype()))
            edge_length_embedded = edge_length_embedded.type(flt2cplx(torch.get_default_dtype()))
        
        edge_fea_old = edge_fea
        
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
        
        edge_fea = self.skip_connect(edge_fea_old, edge_fea)

        return edge_fea


class Net(nn.Module):
    def __init__(self, num_species, irreps_embed_node, irreps_edge_init, irreps_sh, irreps_mid_node, 
                 irreps_post_node, irreps_out_node,irreps_mid_edge, irreps_post_edge, irreps_out_edge, 
                 num_block, r_max, use_sc=True, only_ij=False, spinful=False, num_basis=128,
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
        # self.basis = BesselBasis(r_max, num_basis=num_basis, trainable=False)
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
                node_update_block = NodeUpdateBlock(num_species, num_basis, irreps_sh, irreps_node_prev, irreps_post_node, irreps_edge_prev, act, act_gates, use_sc, is_complex=spinful, only_ij=only_ij, if_sort_irreps=if_sort_irreps)
                edge_update_block = EdgeUpdateBlock(num_species, num_basis, irreps_sh, node_update_block.irreps_out, irreps_edge_prev, irreps_post_edge, act, act_gates, use_sc, is_complex=spinful, if_sort_irreps=if_sort_irreps)
            else:
                node_update_block = NodeUpdateBlock(num_species, num_basis, irreps_sh, irreps_node_prev, irreps_mid_node, irreps_edge_prev, act, act_gates, use_sc, only_ij=only_ij, if_sort_irreps=if_sort_irreps)
                edge_update_block = EdgeUpdateBlock(num_species, num_basis, irreps_sh, node_update_block.irreps_out, irreps_edge_prev, irreps_mid_edge, act, act_gates, use_sc, if_sort_irreps=if_sort_irreps)
            irreps_node_prev = node_update_block.irreps_out
            irreps_edge_prev = edge_update_block.irreps_out
            self.node_update_blocks.append(node_update_block)
            self.edge_update_blocks.append(edge_update_block)
        
        irreps_out_edge = Irreps(irreps_out_edge)
        for _, ir in irreps_out_edge:
            assert ir in irreps_edge_prev, f'required ir {ir} in irreps_out_edge cannot be produced by convolution in the last edge update block ({edge_update_block.irreps_in_edge} -> {edge_update_block.irreps_out})'
            # if irreps_out_edge.count(ir) > irreps_edge_prev.count(ir):
            #     msg = f'multiplicity of {ir} in irreps {irreps_edge_prev} produced by the last edge update block is smaller than the multiplicity of that in irreps_out_edge, which is {irreps_out_edge.count(ir)}'
            #     warnings.warn(msg)

        self.irreps_out_node = irreps_out_node
        self.irreps_out_edge = irreps_out_edge
        self.lin_node = cplxE3Linear(irreps_in=irreps_node_prev, irreps_out=irreps_out_node, is_complex=spinful)
        self.lin_edge = cplxE3Linear(irreps_in=irreps_edge_prev, irreps_out=irreps_out_edge, is_complex=spinful)

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
        for node_update_block, edge_update_block in zip(self.node_update_blocks, self.edge_update_blocks):
            node_fea = node_update_block(node_fea, node_one_hot, edge_sh, edge_fea, edge_length_embedded, data["edge_index"], data.batch, selfloop_edge, data["edge_attr"][:, 0])
            edge_fea = edge_update_block(node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, data["edge_index"], data.batch)

        node_fea = self.lin_node(node_fea)
        edge_fea = self.lin_edge(edge_fea)
        return node_fea, edge_fea

    def __repr__(self):
        info = '===== e3Aij model structure: ====='
        info += f'\nusing spherical harmonics: {self.irreps_sh}'
        for index, (nupd, eupd) in enumerate(zip(self.node_update_blocks, self.edge_update_blocks)):
            info += f'\n=== layer {index} ==='
            info += f'\n{"complex " if nupd.is_complex else ""}node update: ({nupd.irreps_in_node} -> {nupd.irreps_out})'
            info += f'\n{"complex " if eupd.is_complex else ""}edge update: ({eupd.irreps_in_edge} -> {eupd.irreps_out})'
        info += '\n=== output ==='
        info += f'\noutput node: ({self.irreps_out_node})'
        info += f'\noutput edge: ({self.irreps_out_edge})'
        
        return info