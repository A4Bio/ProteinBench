from ast import Global
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean
import numpy as np
from ..tools.design_utils import gather_nodes
import math


class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


activation_maps = {
    'leakyrelu': nn.LeakyReLU(),
    'relu': nn.ReLU(),
    'silu': nn.SiLU(),
    'mish': nn.Mish(),
    'gelu': nn.GELU()
}

def permute_final_dims(tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

from scipy.stats import truncnorm

def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape
    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")
    return f

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))

class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.
    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        init="default",
        init_fn=None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:
                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0
                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                trunc_normal_init_(self.weight, scale=1.0)
            elif init == "relu":
                trunc_normal_init_(self.weight, scale=2.0)
            elif init == "glorot":
                nn.init.xavier_uniform_(self.weight, gain=1)
            elif init == "gating":
                with torch.no_grad():
                    self.weight.fill_(0.0)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
            elif init == "final":
                 with torch.no_grad():
                    self.weight.fill_(0.0)
            else:
                raise ValueError("Invalid init string.")


class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.c_in = (c_in,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        out = nn.functional.layer_norm(x, self.c_in,
            self.weight, self.bias, self.eps)
        return out


def graph2matrix(x, chunks):
    matrix = torch.stack(torch.chunk(x, chunks), dim=0)
    return matrix

class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, c_in, c_hidden, _outgoing=False):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = Linear(self.c_in, self.c_hidden)
        self.linear_a_g = Linear(self.c_in, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_in, self.c_hidden)
        self.linear_b_g = Linear(self.c_in, self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_in, self.c_hidden, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_hidden, init="final")

        self.layer_norm_in = LayerNorm(self.c_in)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self, a, b):
        idxs = ((2, 0, 1), (2, 1, 0))
        idx_a, idx_b = (idxs[1], idxs[0]) if self._outgoing else (idxs[0], idxs[1])
        p = torch.matmul(
            permute_final_dims(a, idx_a),
            permute_final_dims(b, idx_b),
        )
        return permute_final_dims(p, (1, 2, 0))

    def forward(self, z, src_idx, dst_idx):
        chunks = src_idx.shape[0] // len(src_idx[src_idx == 0])
        z = graph2matrix(z, chunks)
        z = self.layer_norm_in(z)
        a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z))
        # a: [*, N_res, N_res, C_z]
        b = self.linear_b_p(z) * self.sigmoid(self.linear_b_g(z))
        # b: [*, N_res, N_res, C_z]
        # tri_mul_out and tri_mul_in are different here
        x = self._combine_projections(a, b)

        ndst_idx = torch.stack(torch.chunk(dst_idx, chunks), dim=0)
        ndst_idx = ndst_idx.repeat(self.c_hidden, 1, 1).permute(1, 2, 0)
        x = torch.gather(x, 1, ndst_idx)

        x = self.layer_norm_out(x)
        # x: [*, N_res, N_res, C_z]
        x = self.linear_z(x)
        # g: [*, N_res, N_res, C_z]
        g = self.sigmoid(self.linear_g(z))
        # z: [*, N_res, N_res, C_z]

        z = x * g
        return z.reshape(-1, z.shape[-1])


"""============================================================================================="""
""" Graph Encoder """
"""============================================================================================="""

def get_attend_mask(idx, mask):
    mask_attend = gather_nodes(mask.unsqueeze(-1), idx).squeeze(-1) # 一阶邻居节点的mask: 1代表节点存在, 0代表节点不存在
    mask_attend = mask.unsqueeze(-1) * mask_attend # 自身的mask*邻居节点的mask
    return mask_attend

#################################### node modules ###############################
class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, edge_drop=0.0, output_mlp=True):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        self.output_mlp = output_mlp
        
        # self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        # self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        # self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Sequential(nn.Linear(num_in, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, num_hidden)
        )
        self.Bias = nn.Sequential(
                                nn.Linear(num_hidden*3, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_heads)
                                )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, h_V, h_E, center_id, batch_id, dst_idx=None):
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        
        w = self.Bias(torch.cat([h_V[center_id], h_E],dim=-1)).view(E, n_heads, 1) 
        attend_logits = w/np.sqrt(d) 

        V = self.W_V(h_E).view(-1, n_heads, d) 
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        h_V = scatter_sum(attend*V, center_id, dim=0).view([-1, self.num_hidden])

        if self.output_mlp:
            h_V_update = self.W_O(h_V)
        else:
            h_V_update = h_V
        return h_V_update

class GCN(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(GCN, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden*3, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

        from .proteinmpnn_module import PositionWiseFeedForward
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, src_idx, batch_id, dst_idx):
        """ Parallel computation of full transformer layer """
        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        dh = scatter_mean(h_message, src_idx, dim=0) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        return h_V

class GAT(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(GAT, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden*3, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

        self.A = nn.Parameter(torch.empty(size=(num_hidden + num_in, 1)))

        from .proteinmpnn_module import PositionWiseFeedForward
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, src_idx, batch_id, dst_idx):
        """ Parallel computation of full transformer layer """
        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        e = F.sigmoid(F.leaky_relu(torch.matmul(h_EV, self.A))).squeeze(-1).exp()
        e = e / e.sum(-1).unsqueeze(-1)
        h_message = h_message * e.unsqueeze(-1)

        dh = scatter_sum(h_message, src_idx, dim=0) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        return h_V

class QKV(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, edge_drop=0.0):
        super(QKV, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.Bias = nn.Sequential(
                                nn.Linear(num_hidden*3, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_heads)
                                )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, h_V, h_E, center_id, batch_id, dst_idx):
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)

        Q = self.W_Q(h_V).view(N, n_heads, 1, d)[center_id]
        K = self.W_K(h_E).view(E, n_heads, d, 1)
        attend_logits = torch.matmul(Q, K).view(E, n_heads, 1)
        attend_logits = attend_logits / np.sqrt(d)

        V = self.W_V(h_E).view(-1, n_heads, d) 
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        h_V = scatter_sum(attend*V, center_id, dim=0).view([N, self.num_hidden])

        h_V_update = self.W_O(h_V)
        return h_V_update


#################################### edge modules ###############################
class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EdgeMLP, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E

class DualEGraph(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DualEGraph, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        E_agg = scatter_mean(h_E, dst_idx, dim=0)
        h_EV = torch.cat([E_agg[src_idx], h_E, E_agg[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E

#################################### context modules ###############################
class Context(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, node_context = False, edge_context = False):
        super(Context, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.node_context = node_context
        self.edge_context = edge_context

        self.V_MLP = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                )
        
        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

        self.E_MLP = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden)
                                )
        
        self.E_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        if self.node_context:
            c_V = scatter_mean(h_V, batch_id, dim=0)
            h_V = h_V * self.V_MLP_g(c_V[batch_id])
            # h_V = h_V + h_V * self.V_MLP_g(c_V[batch_id])
            # h_V = self.V_MLP(h_V) * self.V_MLP_g(c_V[batch_id])
            # h_V = h_V + self.V_MLP(h_V) * self.V_MLP_g(c_V[batch_id])
        
        # if self.edge_context:
        #     c_V = scatter_mean(h_V, batch_id, dim=0)
        #     h_E = h_E * self.E_MLP_g(c_V[batch_id[edge_idx[0]]])

        return h_V, h_E


class GeneralGNN(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, node_net = 'AttMLP', edge_net = 'EdgeMLP', node_context = 0, edge_context = 0):
        super(GeneralGNN, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(3)])
        self.node_net = node_net
        self.edge_net = edge_net
        if node_net == 'AttMLP':
            self.attention = NeighborAttention(num_hidden, num_in, num_heads=4) 
        if node_net == 'GCN':
            self.attention = GCN(num_hidden, num_in, num_heads=4) 
        if node_net == 'GAT':
            self.attention = GAT(num_hidden, num_in, num_heads=4) 
        if node_net == 'QKV':
            self.attention = QKV(num_hidden, num_in, num_heads=4) 
        
        if edge_net == 'None':
            pass
        if edge_net == 'EdgeMLP':
            self.edge_update = EdgeMLP(num_hidden, num_in, num_heads=4)
        if edge_net == 'DualEGraph':
            self.edge_update = DualEGraph(num_hidden, num_in, num_heads=4)
        
        self.context = Context(num_hidden, num_in, num_heads=4, node_context=node_context, edge_context=edge_context)

        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        if self.node_net == 'AttMLP' or self.node_net == 'QKV':
            dh = self.attention(h_V, torch.cat([h_E, h_V[dst_idx]], dim=-1), src_idx, batch_id, dst_idx)
        else:
            dh = self.attention(h_V, h_E, src_idx, batch_id, dst_idx)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if self.edge_net=='None':
            pass
        else:
            h_E = self.edge_update( h_V, h_E, edge_idx, batch_id )

        h_V, h_E = self.context(h_V, h_E, edge_idx, batch_id)
        return h_V, h_E

class GNNModule(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0):
        super(GNNModule, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(2)])
        self.attention = NeighborAttention(num_hidden, num_in, num_heads, edge_drop=0.0) # TODO: edge_drop
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        center_id = edge_idx[0]
        dh = self.attention(h_V, h_E, center_id, batch_id)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        return h_V


class GNNModule_E1(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, att_output_mlp=True, node_output_mlp=True):
        super(GNNModule_E1, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.node_output_mlp = node_output_mlp
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(3)])
        self.attention = NeighborAttention(num_hidden, num_in, num_heads=4, edge_drop=0.0, output_mlp=att_output_mlp) # TODO: edge_drop
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        # self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        # self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        # self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        dh = self.attention(h_V, torch.cat([h_E, h_V[dst_idx]], dim=-1), src_idx, batch_id)
        h_V = self.norm[0](h_V + self.dropout(dh))
        if self.node_output_mlp:
            dh = self.dense(h_V)
            h_V = self.norm[1](h_V + self.dropout(dh))

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm[2](h_E + self.dropout(h_message))
        return h_V, h_E


class GNNModule_E2(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(GNNModule_E2, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

        from .proteinmpnn_module import PositionWiseFeedForward
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        """ Parallel computation of full transformer layer """
        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        dh = scatter_sum(h_message, src_idx, dim=0) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class GNNModule_E3(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(GNNModule_E3, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

        self.A = nn.Parameter(torch.empty(size=(num_hidden + num_in, 1)))

        from .proteinmpnn_module import PositionWiseFeedForward
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        """ Parallel computation of full transformer layer """
        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        e = F.sigmoid(F.leaky_relu(torch.matmul(h_EV, self.A))).squeeze(-1).exp()
        e = e / e.sum(-1).unsqueeze(-1)
        h_message = h_message * e.unsqueeze(-1)

        dh = scatter_sum(h_message, src_idx, dim=0) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E



class StructureEncoder(nn.Module):
    def __init__(self,  hidden_dim, num_encoder_layers=3, dropout=0, updating_edges=0, att_output_mlp=True, node_output_mlp=True, node_net = 'AttMLP', edge_net = 'EdgeMLP', node_context = False, edge_context = False):
        """ Graph labeling network """
        super(StructureEncoder, self).__init__()
        # self.encoder_layers = nn.ModuleList([])
        encoder_layers = []
        
        self.updating_edges = updating_edges
        if updating_edges == 0:
            module = GNNModule
        elif updating_edges == 1:
            module = GNNModule_E1
        elif updating_edges == 2:
            module = GNNModule_E2
        elif updating_edges == 3:
            module = GNNModule_E3
        elif updating_edges == 4:
            module = GeneralGNN

        if updating_edges == 4:
            for i in range(num_encoder_layers):
                encoder_layers.append(
                    module(hidden_dim, hidden_dim*2, dropout=dropout, node_net = node_net, edge_net = edge_net, node_context = node_context, edge_context = edge_context),
                )
        else:
            for i in range(num_encoder_layers):
                encoder_layers.append(
                    module(hidden_dim, hidden_dim*2, dropout=dropout, att_output_mlp=att_output_mlp, node_output_mlp=node_output_mlp),
                )
        
        self.encoder_layers = nn.Sequential(*encoder_layers)

    def forward(self, h_V, h_P, P_idx, batch_id):
        for layer in self.encoder_layers:
            if self.updating_edges == 0:
                h_V = layer(h_V, torch.cat([h_P, h_V[P_idx[1]]], dim=1), P_idx, batch_id)
                # h_V = h_V + layer(h_V, torch.cat([h_P, h_V[P_idx[1]]], dim=1), P_idx, batch_id)
            else:
                h_V, h_P = layer(h_V, h_P, P_idx, batch_id)
        return h_V, h_P

"""============================================================================================="""
""" Sequence Decoder """
"""============================================================================================="""

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class ConvBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size, padding, act_func, glu=0):
        super().__init__()

        self.glu = glu
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = act_func
        if glu == 0:
            self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        elif glu == 1:
            self.conv = nn.Conv1d(hidden_dim, 2*hidden_dim, kernel_size, padding=padding)
    
    def forward(self, x):
        if self.glu == 0:
            return self.conv(self.act(self.bn(x)))
        elif self.glu == 1:
            f_g = self.conv(self.act(self.bn(x)))
            split_dim = f_g.shape[1] // 2
            f_x, g_x = torch.split(f_g, split_dim, dim=1)
            return torch.sigmoid(g_x) * f_x

class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers=3, kernel_size=5, act_type='relu', glu=0, vocab=21):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # module_lst = [nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding)]
        # for _ in range(num_layers):
        #     module_lst.append(ConvBlock(hidden_dim, kernel_size, padding, activation_maps[act_type], glu))

        # self.CNN = nn.Sequential(*module_lst)
        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, batch_id=None, token_mask=None):
        # h_V = h_V.unsqueeze(0).permute(0,2,1)
        # hidden = self.CNN(h_V).permute(0,2,1).squeeze()
        logits = self.readout(h_V)
        # if token_mask is not None:
        #     token_mask = token_mask[None,:].to(h_V.device)
        #     logits = logits*token_mask -999999*(~token_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits

class CNNDecoder(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers=3, kernel_size=5, act_type='relu', glu=0, vocab=20):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        module_lst = [nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding)]
        for _ in range(num_layers):
            module_lst.append(ConvBlock(hidden_dim, kernel_size, padding, activation_maps[act_type], glu))

        self.CNN = nn.Sequential(*module_lst)

        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, batch_id):
        h_V = h_V.unsqueeze(0).permute(0,2,1)
        hidden = self.CNN(h_V).permute(0,2,1).squeeze()
        logits = self.readout(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits

class CNNDecoder2(nn.Module):
    def __init__(self,hidden_dim, input_dim, num_layers=3, kernel_size=5, act_type='relu', glu=0, vocab=20):
        super().__init__()
        self.ConfNN = nn.Embedding(50, hidden_dim)

        padding = (kernel_size - 1) // 2
        module_lst = [nn.Conv1d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=padding)]
        for _ in range(num_layers):
            module_lst.append(ConvBlock(hidden_dim, kernel_size, padding, activation_maps[act_type], glu))

        self.CNN = nn.Sequential(*module_lst)

        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, logits, batch_id):
        eps = 1e-5
        L = h_V.shape[0]
        idx = torch.argsort(-logits, dim=1)
        Conf = logits[range(L), idx[:,0]] / (logits[range(L), idx[:,1]] + eps)
        Conf = Conf.long()
        Conf = torch.clamp(Conf, 0, 49)
        h_C = self.ConfNN(Conf)
        
        h_V = torch.cat([h_V,h_C],dim=-1)
        h_V = h_V.unsqueeze(0).permute(0,2,1)
        hidden = self.CNN(h_V).permute(0,2,1).squeeze()
        logits = self.readout(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.relu(self.W_in(h_V))
        h = self.W_out(h)
        return h

class Local_Module(nn.Module):
    def __init__(self, num_hidden, num_in, is_attention, dropout=0.1, scale=30):
        super(Local_Module, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.is_attention = is_attention
        self.scale = scale
        self.dropout = nn.Dropout(0)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])
        self.W = nn.Sequential(*[
            nn.Linear(num_hidden + num_in, num_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_hidden, num_hidden)
        ])
        self.A = nn.Parameter(torch.empty(size=(num_hidden + num_in, 1)))
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, edge_idx):
        message = torch.cat( [h_V[edge_idx[0]], h_E], dim=1 )
        h_message = self.W(message) # [17790, 128]
        # Attention
        if self.is_attention == 1:
            att = F.sigmoid(F.leaky_relu(torch.matmul(message, self.A))).exp()
            att = att / scatter_sum(att, edge_idx[0], dim=0)[edge_idx[0]]
            h_message = h_message * att # [4, 312, 30, 128]

        # message aggragation
        dh = scatter_sum(h_message, edge_idx[0], dim=0) / self.scale
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        return h_V
    

class ATDecoder(nn.Module):
    def __init__(self, args, hidden_dim, dropout=0.1, vocab=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.UpdateE = nn.Linear(hidden_dim*2, hidden_dim)

        self.decoder = nn.ModuleList([])
        for _ in range(args.AT_layer_num):
            self.decoder.append(
                    Local_Module(hidden_dim, hidden_dim*3, is_attention=1, dropout=dropout)
                )
        self.readout = MLPDecoder(hidden_dim, hidden_dim, args.num_decoder_layers1, args.kernel_size1, args.act_type, args.glu)
    
    def forward(self, S, h_V, h_P, P_idx, batch_id, mask_bw=None, mask_fw=None):
        h_S = self.W_s(S)
        h_PS = torch.cat((h_P, h_S[P_idx[1]]), dim=1)
        h_PSV_enc = torch.cat((h_P, torch.zeros_like(h_S[P_idx[1]]), h_V[P_idx[1]]), dim=1)
        known_mask = (P_idx[0]>P_idx[1]).unsqueeze(-1)
        for dec_layer in self.decoder:
            h_PSV_dec = torch.cat((h_PS, h_V[P_idx[1]]), dim=1)
            h_PSV = h_PSV_dec*known_mask + h_PSV_enc*(~known_mask)

            # 仅当dst node的mask为1时使用h_PSV_dec, 否则使用h_PSV_enc
            # h_PSV = h_PSV_dec*mask_bw[P_idx[1]].view(-1,1) + h_PSV_enc*mask_fw[P_idx[1]].view(-1,1) 
            h_V = dec_layer(h_V, h_PSV , P_idx)
        log_probs, logits = self.readout(h_V)
        return log_probs

    def sampling(self, h_V, h_P, P_idx, batch_id, temperature=0.1, mask_bw=None, mask_fw=None, decoding_order=None):
        device = h_V.device
        L = h_V.shape[0]

        # cache
        S = torch.zeros( L, device=device, dtype=torch.int)
        energy = torch.zeros( L, device=device)
        h_S = torch.zeros( L, self.hidden_dim, device=device)
        h_PSV_enc = torch.cat((h_P, torch.zeros_like(h_S[P_idx[1]]), h_V[P_idx[1]]), dim=1)
        h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.decoder))]
        log_probs = torch.zeros( L, self.vocab, device=device)
        
        # shift = scatter_sum(torch.ones_like(batch_id),batch_id, dim=0)
        # shift = torch.cumsum(torch.cat([torch.zeros(1, device=device, dtype=torch.long),shift]), dim=0)[:-1]
        # (torch.sort(decoding_order)[0].diff(dim=0)==1).all()
        known = torch.zeros_like(P_idx[0])==1
        for t in decoding_order:
            edge_mask = P_idx[0] % L == t # 批量预测第t个氨基酸
            h_V_t = h_V[t:t+1,:]
            P_idx_t = P_idx[:, edge_mask]
            h_PS = torch.cat((h_P, h_S[P_idx[1]]), dim=1)
            h_PS_t = h_PS[edge_mask]
            h_PSV_enc_t = h_PSV_enc[edge_mask]
            known_mask = (P_idx_t[0]>P_idx_t[1]).unsqueeze(-1)
        
            for l, dec_layer in enumerate(self.decoder):
                h_PSV_t_dec = torch.cat((h_PS_t,
                                    h_V_stack[l][P_idx_t[1]]),
                                    dim=1)
                h_PSV_t = h_PSV_t_dec*known_mask + h_PSV_enc_t*(~known_mask)
                # h_PSV_t = h_PSV_t_dec*known[P_idx_t[1]].view(-1,1) + h_PSV_enc_t*(~known[P_idx_t[1]].view(-1,1))
                
                h_V_t = h_V_stack[l][t:t+1,:]
                edge_index_t_local = torch.zeros_like(P_idx_t)
                edge_index_t_local[1,:] = torch.arange(0, P_idx_t.shape[1], device=h_V.device)
                h_V_t = dec_layer(h_V_t, h_PSV_t , edge_index_t_local)
                
                h_V_stack[l+1][t] = h_V_t
            
            h_V_t = h_V_stack[-1][t]
            log_probs_t, logits_t = self.readout(h_V_t) 
            log_probs[t] = log_probs_t
            probs = F.softmax(logits_t/temperature, dim=-1)
            S_t = torch.multinomial(probs, 1).squeeze(-1)
            h_S[t::L] = self.W_s(S_t)
            S[t::L] = S_t
            known[t] = True
        
        return log_probs



    # def forward(self, S, h_V, h_E, edge_idx, batch_id):
    #     h_S = self.W_s(S)
    #     known_mask = (edge_idx[0]>edge_idx[1]).unsqueeze(-1)
    #     h_ES_known = self.UpdateE(torch.cat([h_E, h_S[edge_idx[1]]], dim=-1))

    #     for dec_layer in self.decoder:
    #         h_E_mix = h_ES_known*known_mask + h_E*(~known_mask)
    #         h_V, _ = dec_layer(h_V, h_E_mix, edge_idx, batch_id)
        
    #     log_probs, logits = self.readout(h_V)
    #     return log_probs
    
    # def sampling(self, h_V, h_E, edge_idx, batch_id, temperature=0.1):
    #     device = h_V.device
    #     L = h_V.shape[0]

    #     # cache
    #     S = torch.zeros( L, device=device, dtype=torch.int)
    #     h_S = torch.zeros( L, self.hidden_dim, device=device)
    #     h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.decoder))]
    #     log_probs = torch.zeros( L, self.vocab, device=device)
        
    #     for t in range(L):
    #         edge_mask = edge_idx[0] % L == t # 批量预测第t个氨基酸
    #         h_V_t = h_V[t:t+1,:]
    #         E_idx_t = edge_idx[:, edge_mask]
    #         h_ES_known = torch.cat((h_E, h_S[edge_idx[1]]), dim=1)
    #         h_ES_known_t = self.UpdateE(h_ES_known[edge_mask])
    #         h_E_t = h_E[edge_mask]
    #         known_mask = (E_idx_t[0]>E_idx_t[1]).unsqueeze(-1)
        
    #         for l, dec_layer in enumerate(self.decoder):
    #             h_ES_t = h_ES_known_t*known_mask + h_E_t*(~known_mask)
                
    #             h_V_t = h_V_stack[l][E_idx_t[1],:]
    #             edge_index_t_local = torch.zeros_like(E_idx_t)
    #             edge_index_t_local[1,:] = torch.arange(0, E_idx_t.shape[1], device=h_V.device)
    #             batch_id_t = torch.zeros(h_V_t.shape[0], dtype=torch.long, device=device)
    #             h_V_t, _ = dec_layer(h_V_t, h_ES_t , edge_index_t_local, batch_id_t)
                
    #             h_V_stack[l+1][t] = h_V_t[0]
            
    #         h_V_t = h_V_stack[-1][t]
    #         log_probs_t, logits_t = self.readout(h_V_t) 
    #         log_probs[t] = log_probs_t
    #         probs = F.softmax(logits_t/temperature, dim=-1)
    #         S_t = torch.multinomial(probs, 1).squeeze(-1)
    #         h_S[t::L] = self.W_s(S_t)
    #         S[t::L] = S_t
        
    #     return log_probs

if __name__ == '__main__':
    pass