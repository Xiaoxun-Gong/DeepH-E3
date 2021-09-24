import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from e3nn.nn import FullyConnectedNet, Gate, BatchNorm, NormActivation, Extract
from e3nn.o3 import TensorProduct, Irrep, Irreps, Linear, SphericalHarmonics, FullyConnectedTensorProduct
from nequip.nn.cutoffs import PolynomialCutoff
from nequip.nn.radial_basis import BesselBasis
from nequip.utils.tp_utils import tp_path_exists


epsilon = 1e-8


class EquiConv(nn.Module):
    def __init__(self, fc_len_in, irreps_in1, irreps_in2, irreps_out,
                 instructions=None, weight_layers=1, weight_neurons=16):
        super(EquiConv, self).__init__()
        if instructions == None:
            self.tp = FullyConnectedTensorProduct(
                irreps_in1, irreps_in2, irreps_out, shared_weights=False, internal_weights=False,
            )
        else:
            self.tp = TensorProduct(
                irreps_in1, irreps_in2, irreps_out, instructions,
                shared_weights=False,
                internal_weights=False,
            )

        self.fc = FullyConnectedNet(
            [fc_len_in]
            + weight_layers * [weight_neurons]
            + [self.tp.weight_numel],
            nn.functional.silu,
        )

    def forward(self, fea_in1, fea_in2, fea_weight):
        weight = self.fc(fea_weight)
        edge_update = self.tp(fea_in1, fea_in2, weight)
        return edge_update


class NodeUpdateBlock(nn.Module):
    def __init__(self, fc_len_in, irreps_sh, irreps_in_node, irreps_out_node, irreps_in_edge,
                 act, act_gates, use_sc, concat=True):
        super(NodeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_sh = Irreps(irreps_sh)
        irreps_out_node = Irreps(irreps_out_node)
        irreps_in_edge = Irreps(irreps_in_edge)

        if concat:
            irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
        else:
            irreps_in1 = irreps_in_node
        irreps_in2 = irreps_sh

        irreps_scalars = Irreps([
            (mul, ir)
            for mul, ir in irreps_out_node
            if ir.l == 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
        ]).simplify()
        irreps_gated = Irreps([
            (mul, ir)
            for mul, ir in irreps_out_node
            if ir.l > 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
        ])
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

        self.nonlin = Gate(
            irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
            irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
            irreps_gated  # gated tensors
        )

        self.lin_pre = Linear(irreps_in=irreps_in_node, irreps_out=irreps_in_node)
        if use_sc:
            self.sc = FullyConnectedTensorProduct(irreps_in_node, irreps_in_node, self.nonlin.irreps_in)
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, self.nonlin.irreps_in)
        self.bn = BatchNorm(self.nonlin.irreps_in, instance=True)
        self.lin_post = Linear(irreps_in=self.nonlin.irreps_in, irreps_out=self.nonlin.irreps_in)
        self.irreps_out = self.nonlin.irreps_out
        self.use_sc = use_sc
        self.concat = concat


    def forward(self, node_fea, edge_sh, edge_fea, edge_length_embedded, edge_index):
        node_fea = self.lin_pre(node_fea)

        if self.use_sc:
            node_self_connection = self.sc(node_fea, node_fea)
        index_i = edge_index[0]
        index_j = edge_index[1]
        if self.concat:
            edge_update = self.conv(torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1), edge_sh, edge_length_embedded)
        else:
            edge_update = self.conv(node_fea[index_j], edge_sh, edge_length_embedded)
        node_fea = scatter(edge_update, index_i, dim=0, dim_size=node_fea.shape[0])
        # node_fea = self.bn(node_fea)
        if self.use_sc:
            node_fea = node_fea + node_self_connection

        node_fea = self.lin_post(node_fea)
        node_fea = self.nonlin(node_fea)
        return node_fea


class EdgeUpdateBlock(nn.Module):
    def __init__(self, fc_len_in, irreps_sh, irreps_in_node, irreps_in_edge, irreps_out_edge,
                 act, act_gates, use_sc, init_edge=False):
        super(EdgeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_out_edge = Irreps(irreps_out_edge)

        irreps_in1 = irreps_in_node + irreps_in_node
        irreps_in2 = irreps_sh

        irreps_scalars = Irreps([
            (mul, ir)
            for mul, ir in irreps_out_edge
            if ir.l == 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
        ]).simplify()
        irreps_gated = Irreps([
            (mul, ir)
            for mul, ir in irreps_out_edge
            if ir.l > 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
        ])
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

        self.nonlin = Gate(
            irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
            irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
            irreps_gated  # gated tensors
        )

        self.lin_pre = Linear(irreps_in=irreps_in_edge, irreps_out=irreps_in_edge)
        if use_sc:
            self.sc = FullyConnectedTensorProduct(irreps_in_edge, irreps_in_edge, self.nonlin.irreps_in)
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, self.nonlin.irreps_in)
        self.bn = BatchNorm(self.nonlin.irreps_in, instance=True)
        self.lin_post = Linear(irreps_in=self.nonlin.irreps_in, irreps_out=self.nonlin.irreps_in)
        self.irreps_out = self.nonlin.irreps_out
        self.use_sc = use_sc
        self.init_edge = init_edge

    def forward(self, node_fea, edge_sh, edge_fea, edge_length_embedded, edge_index):
        if not self.init_edge:
            edge_fea = self.lin_pre(edge_fea)
            if self.use_sc:
                edge_self_connection = self.sc(edge_fea, edge_fea)
        index_i = edge_index[0]
        index_j = edge_index[1]
        edge_fea = self.conv(torch.cat([node_fea[index_i], node_fea[index_j]], dim=-1), edge_sh, edge_length_embedded)
        # edge_fea = self.bn(edge_fea)
        if self.use_sc:
            edge_fea = edge_fea + edge_self_connection

        edge_fea = self.lin_post(edge_fea)
        edge_fea = self.nonlin(edge_fea)
        return edge_fea


class Net(nn.Module):
    def __init__(self, num_species, irreps_embed, irreps_sh, irreps_mid_node, irreps_post_node, irreps_out_node,
                 irreps_mid_edge, irreps_post_edge, irreps_out_edge, num_block, use_sc, r_max,
                 act={1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates={1: torch.sigmoid, -1: torch.tanh},
                 num_basis=32):
        super(Net, self).__init__()
        self.num_species = num_species

        self.embedding = Linear(
            irreps_in=f"{num_species}x0e", irreps_out=irreps_embed
        )

        self.basis = BesselBasis(r_max, num_basis=num_basis, trainable=True)
        self.cutoff = PolynomialCutoff(r_max, p=6)

        self.sh = SphericalHarmonics(
            irreps_out=irreps_sh,
            normalize=True,
            normalization='component',
        )

        self.edge_update_block_init = EdgeUpdateBlock(num_basis, irreps_sh, self.embedding.irreps_out, None, irreps_mid_edge, act, act_gates, False, init_edge=True)
        irreps_node_prev = self.embedding.irreps_out
        irreps_edge_prev = self.edge_update_block_init.irreps_out

        self.node_update_blocks = nn.ModuleList([])
        self.edge_update_blocks = nn.ModuleList([])
        for index_block in range(num_block):
            if index_block == num_block - 1:
                node_update_block = NodeUpdateBlock(num_basis, irreps_sh, irreps_node_prev, irreps_post_node, irreps_edge_prev, act, act_gates, use_sc)
                edge_update_block = EdgeUpdateBlock(num_basis, irreps_sh, node_update_block.irreps_out, irreps_edge_prev, irreps_post_edge, act, act_gates, use_sc)
            else:
                node_update_block = NodeUpdateBlock(num_basis, irreps_sh, irreps_node_prev, irreps_mid_node, irreps_edge_prev, act, act_gates, use_sc)
                edge_update_block = EdgeUpdateBlock(num_basis, irreps_sh, node_update_block.irreps_out, irreps_edge_prev, irreps_mid_edge, act, act_gates, use_sc)
            irreps_node_prev = node_update_block.irreps_out
            irreps_edge_prev = edge_update_block.irreps_out
            self.node_update_blocks.append(node_update_block)
            self.edge_update_blocks.append(edge_update_block)

        self.lin_node = Linear(irreps_in=irreps_post_node, irreps_out=irreps_out_node)
        self.lin_edge = Linear(irreps_in=irreps_post_edge, irreps_out=irreps_out_edge)

    def forward(self, data):
        node_fea = F.one_hot(data.x, num_classes=self.num_species).type(torch.get_default_dtype())
        node_fea = self.embedding(node_fea)

        edge_vec = data["edge_attr"][:, [2, 3, 1]] # (y, z, x) order

        edge_sh = self.sh(edge_vec)
        edge_length_embedded = self.basis(data["edge_attr"][:, 0] + epsilon) * self.cutoff(data["edge_attr"][:, 0])[:, None]

        edge_fea = self.edge_update_block_init(node_fea, edge_sh, None, edge_length_embedded, data["edge_index"])
        for node_update_block, edge_update_block in zip(self.node_update_blocks, self.edge_update_blocks):
            node_fea = node_update_block(node_fea, edge_sh, edge_fea, edge_length_embedded, data["edge_index"])
            edge_fea = edge_update_block(node_fea, edge_sh, edge_fea, edge_length_embedded, data["edge_index"])

        node_fea = self.lin_node(node_fea)
        edge_fea = self.lin_edge(edge_fea)
        return node_fea, edge_fea
