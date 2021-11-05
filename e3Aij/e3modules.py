import torch
from torch import nn
from torch_scatter import scatter

from e3nn.o3 import Irrep, Irreps, wigner_3j
from e3nn.nn import Extract


class Rotate:
    def __init__(self, default_dtype_torch, device_torch='cpu'):
        sqrt_2 = 1.4142135623730951
        # openmx的实球谐函数基组变复球谐函数
        self.Us_openmx = {
            0: torch.tensor([1], dtype=torch.cfloat, device=device_torch),
            1: torch.tensor([[-1 / sqrt_2, 1j / sqrt_2, 0], [0, 0, 1], [1 / sqrt_2, 1j / sqrt_2, 0]], dtype=torch.cfloat, device=device_torch),
            2: torch.tensor([[0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0]], dtype=torch.cfloat, device=device_torch),
            3: torch.tensor([[0, 0, 0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [0, 0, 0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, -1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2, 0, 0],
                             [0, 0, 0, 0, 0, 1 / sqrt_2, 1j / sqrt_2]], dtype=torch.cfloat, device=device_torch),
        }
        # openmx的实球谐函数基组变wiki的实球谐函数 https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        self.Us_openmx2wiki = {
            0: torch.eye(1, dtype=default_dtype_torch).to(device=device_torch),
            1: torch.eye(3, dtype=default_dtype_torch)[[1, 2, 0]].to(device=device_torch),
            2: torch.eye(5, dtype=default_dtype_torch)[[2, 4, 0, 3, 1]].to(device=device_torch),
            3: torch.eye(7, dtype=default_dtype_torch)[[6, 4, 2, 0, 1, 3, 5]].to(device=device_torch)
        }
        self.Us_wiki2openmx = {k: v.T for k, v in self.Us_openmx2wiki.items()}

    def rotate_e3nn_v(self, v, R, l, order_xyz=True):
        if order_xyz:
            # R是(x, y, z)顺序
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn是(y, z, x)顺序
        else:
            # R是(y, z, x)顺序
            R_e3nn = R
        return v @ Irrep(l, 1).D_from_matrix(R_e3nn)

    def rotate_openmx_H(self, H, R, l_left, l_right, order_xyz=True):
        if order_xyz:
            # R是(x, y, z)顺序
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn是(y, z, x)顺序
        else:
            # R是(y, z, x)顺序
            R_e3nn = R
        return self.Us_openmx2wiki[l_left].T @ Irrep(l_left, 1).D_from_matrix(R_e3nn).transpose(-1, -2) @ self.Us_openmx2wiki[l_left] @ H \
               @ self.Us_openmx2wiki[l_right].T @ Irrep(l_right, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_right]
    
    def rotate_openmx_H_full(self, H, R, orbital_types_left, orbital_types_right, order_xyz=True):
        if order_xyz:
            # R是(x, y, z)顺序
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn是(y, z, x)顺序
        else:
            # R是(y, z, x)顺序
            R_e3nn = R
        irreps_left = Irreps([(1, (l, (- 1) ** l)) for l in orbital_types_left])
        irreps_right = Irreps([(1, (l, (- 1) ** l)) for l in orbital_types_right])
        openmx2wiki_left = torch.block_diag(*[self.Us_openmx2wiki[l] for l in orbital_types_left])
        openmx2wiki_right = torch.block_diag(*[self.Us_openmx2wiki[l] for l in orbital_types_right])
        return openmx2wiki_left.T @ irreps_left.D_from_matrix(R_e3nn).transpose(-1, -2) @ openmx2wiki_left @ H \
               @ openmx2wiki_right.T @ irreps_right.D_from_matrix(R_e3nn) @ openmx2wiki_right

    def wiki2openmx_H(self, H, l_left, l_right):
        return self.Us_openmx2wiki[l_left].T @ H @ self.Us_openmx2wiki[l_right]

    def openmx2wiki_H(self, H, l_left, l_right):
        return self.Us_openmx2wiki[l_left] @ H @ self.Us_openmx2wiki[l_right].T

    def rotate_matrix_convert(self, R):
        # (x, y, z)顺序排列的旋转矩阵转换为(y, z, x)顺序(see e3nn.o3.spherical_harmonics() and https://docs.e3nn.org/en/stable/guide/change_of_basis.html)
        return torch.eye(3)[[1, 2, 0]] @ R @ torch.eye(3)[[1, 2, 0]].T # todo: cuda


class sort_irreps(torch.nn.Module):
    def __init__(self, irreps_in):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        sorted_irreps = irreps_in.sort()
        irreps_out_list = [((mul, ir),) for mul, ir in sorted_irreps.irreps]
        instructions = [(i,) for i in sorted_irreps.inv]
        self.extr = Extract(irreps_in, irreps_out_list, instructions)
        
        self.irreps_in = irreps_in
        self.irreps_out = sorted_irreps.irreps
    
    def forward(self, x):
        extracted = self.extr(x)
        return torch.cat(extracted, dim=-1)


class construct_H:
    def __init__(self, net_irreps_out, l1, l2):
        net_irreps_out = Irreps(net_irreps_out)
        
        self.l1, self.l2 = l1, l2
        
        # # = check angular momentum quantum number =
        # if l1 == l2:
        #     assert len(net_irreps_out) == 1, 'It is recommended to combine the irreps together if the two output angular momentum quantum number are the same'
        #     assert net_irreps_out[0].mul % 2 == 0
        #     self.mul = net_irreps_out[0].mul // 2
        # elif l1 * l2 == 0:
        #     assert len(net_irreps_out) == 1, 'Only need one irrep if one angular momentum is 0'
        #     assert net_irreps_out[0].mul == 1, 'Only need multiplicity one if one angular momentum is 0'
        #     self.mul = 1
        #     if l1 == 0:
        #         assert net_irreps_out[0].ir.l == l2
        #     elif l2 == 0:
        #         assert net_irreps_out[0].ir.l == l1
        # else:
        #     assert net_irreps_out[0].ir.l == l1
        #     assert net_irreps_out[1].ir.l == l2
        #     assert net_irreps_out[0].mul == net_irreps_out[1].mul
        #     self.mul = net_irreps_out[0].mul
        #     # assert self.mul != 1, 'Too few multiplicities'
               
        # # = check parity
        # for mul_ir in net_irreps_out:
        #     assert mul_ir.ir.p == (- 1) ** mul_ir.ir.l
        self.mul = 4 # ! temporary
            
        self.rotate_kernel = Rotate(torch.get_default_dtype())

    def get_H(self, net_out):
        r''' get openmx type H from net output '''
        if self.l1 == 0:
            H_pred = net_out.unsqueeze(-2)
        elif self.l2 == 0:
            H_pred = net_out.unsqueeze(-1)
        else:
            vec1 = net_out[:, :(self.mul * (2 * self.l1 + 1))].reshape(-1, self.mul, 2 * self.l1 + 1)
            vec2 = net_out[:, (self.mul * (2 * self.l1 + 1)):].reshape(-1, self.mul, 2 * self.l2 + 1)
            H_pred = torch.sum(vec1[:, :, :, None] * vec2[:, :, None, :], dim=-3)

        H_pred = self.rotate_kernel.wiki2openmx_H(H_pred, self.l1, self.l2)
        
        return H_pred.reshape(net_out.shape[0], -1)


class e3TensorDecomp:
    def __init__(self, net_irreps_out, out_js_list, default_dtype_torch, device_torch='cpu'):
        self.dtype = default_dtype_torch
        self.device = device_torch
        self.out_js_list = out_js_list
        net_irreps_out = Irreps(net_irreps_out)

        required_irreps_out = Irreps(None)
        in_slices = [0]
        out_slices = [0]
        wigner_multipliers = []
        for H_l1, H_l2 in out_js_list:
            # = construct required_irreps_out =
            p = (- 1) ** (H_l1 + H_l2) # required parity
            required_ls = range(abs(H_l1 - H_l2), H_l1 + H_l2 + 1)
            if len(net_irreps_out) < len(required_irreps_out) + 1:
                raise ValueError('Net irreps out and target does not match')
            mul = net_irreps_out[len(required_irreps_out)].mul
            required_irreps_out += Irreps([(mul, (l, p)) for l in required_ls])
            
            # = construct slices =
            in_slices.append(required_irreps_out.dim)
            out_slices.append(out_slices[-1] + (2 * H_l1 + 1) * (2 * H_l2 + 1))
            
            # = get CG coefficients multiplier to act on net_out =
            wigner_multiplier = []
            for l in required_ls:
                for i in range(mul):
                    wigner_multiplier.append(wigner_3j(H_l1, H_l2, l, dtype=default_dtype_torch, device=device_torch))
            wigner_multiplier = torch.cat(wigner_multiplier, dim=-1)
            wigner_multipliers.append(wigner_multiplier)

        # = check net irreps out =
        assert net_irreps_out == required_irreps_out, f'requires {required_irreps_out} but got {net_irreps_out}'
        
        self.in_slices = in_slices
        self.out_slices = out_slices
        self.wigner_multipliers = wigner_multipliers

        # = register rotate kernel =
        self.rotate_kernel = Rotate(default_dtype_torch, device_torch)
    
    def get_H(self, net_out):
        r''' get openmx type H from net output '''
        out = torch.zeros(net_out.shape[0], self.out_slices[-1], dtype=self.dtype, device=self.device)
        for i in range(len(self.out_js_list)):
            in_slice = slice(self.in_slices[i], self.in_slices[i + 1])
            out_slice = slice(self.out_slices[i], self.out_slices[i + 1])
            # H_block = torch.einsum('ijk,uk->uij', self.wigner_multiplier, net_out)
            H_block = torch.sum(self.wigner_multipliers[i][None, :, :, :] * net_out[:, None, None, in_slice], dim=-1)
            H_block = self.rotate_kernel.wiki2openmx_H(H_block, *self.out_js_list[i])
            out[:, out_slice] = H_block.reshape(net_out.shape[0], -1)
        return out


class e3LayerNorm(nn.Module):
    def __init__(self, irreps_in, eps=1e-5, affine=True, normalization='component', subtract_mean=True, divide_norm=False):
        super().__init__()
        
        self.irreps_in = Irreps(irreps_in)
        self.eps = eps
        
        if affine:          
            ib, iw = 0, 0
            weight_slices, bias_slices = [], []
            for mul, ir in irreps_in:
                if ir.is_scalar(): # bias only to 0e
                    bias_slices.append(slice(ib, ib + mul))
                    ib += mul
                else:
                    bias_slices.append(None)
                weight_slices.append(slice(iw, iw + mul))
                iw += mul
            self.weight = nn.Parameter(torch.ones([iw]))
            self.bias = nn.Parameter(torch.zeros([ib]))
            self.bias_slices = bias_slices
            self.weight_slices = weight_slices
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.subtract_mean = subtract_mean
        self.divide_norm = divide_norm
        assert normalization in ['component', 'norm']
        self.normalization = normalization
            
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.weight is not None:
            self.weight.data.fill_(1)
            # nn.init.uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0)
            # nn.init.uniform_(self.bias)

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        # input x must have shape [num_node(edge), dim]
        # if first dimension of x is node index, then batch should be batch.batch
        # if first dimension of x is edge index, then batch should be batch.batch[batch.edge_index[0]]
        
        if batch is None:
            batch = torch.full([x.shape[0]], 0, dtype=torch.int64)

        # from torch_geometric.nn.norm.LayerNorm

        batch_size = int(batch.max()) + 1 
        
        out = []
        ix = 0
        for index, (mul, ir) in enumerate(self.irreps_in):        
            field = x[:, ix: ix + mul * ir.dim].reshape(-1, mul, ir.dim) # [node, mul, repr]
            
            # compute and subtract mean
            if self.subtract_mean or ir.l == 0: # do not subtract mean for l>0 irreps if subtract_mean=False
                mean = scatter(field, batch, dim=0, dim_size=batch_size,
                            reduce='mean').mean(dim=1, keepdim=True)
                field = field - mean[batch]
                
            # compute and divide norm
            if self.divide_norm or ir.l == 0: # do not divide norm for l>0 irreps if subtract_mean=False
                norm = scatter(field.pow(2), batch, dim=0, dim_size=batch_size,
                            reduce='mean').mean(dim=[1,2], keepdim=True)
                if self.normalization == 'norm':
                    norm = norm * ir.dim
                field = field / (norm.sqrt()[batch] + self.eps)
            
            # affine
            if self.weight is not None:
                weight = self.weight[self.weight_slices[index]]
                field = field * weight[None, :, None]
            if self.bias is not None and ir.is_scalar():
                bias = self.bias[self.bias_slices[index]]
                field = field + bias[None, :, None]
            
            out.append(field.reshape(-1, mul * ir.dim))
            ix += mul * ir.dim
            
        out = torch.cat(out, dim=-1)
                
        return out

class e3ElementWise:
    def __init__(self, irreps_in):
        self.irreps_in = Irreps(irreps_in)
        
        len_weight = 0
        for mul, ir in self.irreps_in:
            len_weight += mul
        
        self.len_weight = len_weight
    
    def __call__(self, x: torch.Tensor, weight: torch.Tensor):
        # x should have shape [edge/node, channels]
        # weight should have shape [edge/node, self.len_weight]
        
        ix = 0
        iw = 0
        out = []
        for mul, ir in self.irreps_in:
            field = x[:, ix: ix + mul * ir.dim]
            field = field.reshape(-1, mul, ir.dim)
            field = field * weight[:, iw: iw + mul][:, :, None]
            field = field.reshape(-1, mul * ir.dim)
            
            ix += mul * ir.dim
            iw += mul
            out.append(field)
        
        return torch.cat(out, dim=-1)