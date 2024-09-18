import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

from .gvp_module import _norm_no_nan, _split, tuple_cat, _merge, tuple_sum, tuple_index


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, learned=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None

    def forward(self, x):
        bsz, seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.type_as(self._float_tensor)

        positions = self.make_positions(x)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def make_positions(self, x):
        mask = x.ne(self.padding_idx)
        range_buf = torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
        positions = range_buf.expand_as(x)
        return positions * mask.long() + self.padding_idx * (1 - mask.long())

    def get_embedding(self, num_embeddings):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if self.embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb

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

class DihedralFeatures(nn.Module):
    def __init__(self, node_embed_dim):
        """ Embed dihedral angle features. """
        super(DihedralFeatures, self).__init__()
        # 3 dihedral angles; sin and cos of each angle
        node_in = 6
        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in,  node_embed_dim, bias=True)
        self.norm_nodes = Normalize(node_embed_dim)

    def forward(self, X):
        """ Featurize coordinates as an attributed graph """
        V = self._dihedrals(X)
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        return V

    @staticmethod
    def _dihedrals(X, eps=1e-7, return_angles=False):
        # First 3 coordinates are N, CA, C
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        phi, psi, omega = torch.unbind(D,-1)

        if return_angles:
            return phi, psi, omega

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features


class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param tuple_io: whether to keep accepting tuple inputs and outputs when vi
    or vo = 0
    '''
    def __init__(self, in_dims, out_dims, h_dim=None, vector_gate=False,
                 activations=(F.relu, torch.sigmoid), tuple_io=True,
                 eps=1e-8):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.tuple_io = tuple_io
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if vector_gate:
                    self.wg = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.vector_gate = vector_gate
        self.scalar_act, self.vector_act = activations
        self.eps = eps
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2, eps=self.eps)
            s = self.ws(torch.cat([s, vn], -1))
            if self.scalar_act:
                s = self.scalar_act(s)
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    g = self.wg(s).unsqueeze(-1)
                else:
                    g = _norm_no_nan(v, axis=-1, keepdims=True, eps=self.eps)
                if self.vector_act:
                    g = self.vector_act(g)
                    v = v * g
        else:
            if self.tuple_io:
                assert x[1] is None
                x = x[0]
            s = self.ws(x)
            if self.scalar_act:
                s = self.scalar_act(s)
            if self.vo:
                v = torch.zeros(list(s.shape)[:-1] + [self.vo, 3],
                        device=s.device)
        
        if self.vo:
            return (s, v)
        elif self.tuple_io:
            return (s, None)
        else:
            return s

class GVPConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.
    
    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.
    
    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture
    '''
    def __init__(self, in_dims, out_dims, edge_dims, n_layers=3,
            vector_gate=False, module_list=None, aggr="mean", eps=1e-8,
            activations=(F.relu, torch.sigmoid)):
        super(GVPConv, self).__init__(aggr=aggr)
        self.eps = eps
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP((2*self.si + self.se, 2*self.vi + self.ve), out_dims,
                        vector_gate=vector_gate, activations=activations)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP(out_dims, out_dims,
                        vector_gate=vector_gate))
                module_list.append(GVP(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index, 
                    s=x_s, v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                    edge_attr=edge_attr)
        return _split(message, self.vo) 

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)

class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims, tuple_io=True, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.tuple_io = tuple_io
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        self.eps = eps
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            if self.tuple_io:
                return self.scalar_norm(x[0]), None
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False, eps=self.eps)
        nonzero_mask = (vn > 2 * self.eps)
        vn = torch.sum(vn * nonzero_mask, dim=-2, keepdim=True
            ) / (self.eps + torch.sum(nonzero_mask, dim=-2, keepdim=True))
        vn = torch.sqrt(vn + self.eps)
        v = nonzero_mask * (v / vn)
        return self.scalar_norm(s), v

class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        if x is None:
            return None
        device = x.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GVPConv`.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    '''
    def __init__(self, node_dims, edge_dims, vector_gate=False,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False, attention_heads=0,
                 conv_activations=(F.relu, torch.sigmoid),
                 n_edge_gvps=0, layernorm=True, eps=1e-8):
        
        super(GVPConvLayer, self).__init__()
        if attention_heads == 0:
            self.conv = GVPConv(
                    node_dims, node_dims, edge_dims, n_layers=n_message,
                    vector_gate=vector_gate,
                    aggr="add" if autoregressive else "mean",
                    activations=conv_activations, 
                    eps=eps,
            )
        else:
            raise NotImplementedError
        if layernorm:
            self.norm = nn.ModuleList([LayerNorm(node_dims, eps=eps) for _ in range(2)])
        else:
            self.norm = nn.ModuleList([nn.Identity() for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP(node_dims, hid_dims, vector_gate=vector_gate))
            for i in range(n_feedforward-2):
                ff_func.append(GVP(hid_dims, hid_dims, vector_gate=vector_gate))
            ff_func.append(GVP(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

        self.edge_message_func = None
        if n_edge_gvps > 0:
            si, vi = node_dims
            se, ve = edge_dims
            module_list = [
                GVP((2*si + se, 2*vi + ve), edge_dims, vector_gate=vector_gate)
            ]
            for i in range(n_edge_gvps - 2):
                module_list.append(GVP(edge_dims, edge_dims,
                    vector_gate=vector_gate))
            if n_edge_gvps > 1:
                module_list.append(GVP(edge_dims, edge_dims,
                    activations=(None, None)))
            self.edge_message_func = nn.Sequential(*module_list)
            if layernorm:
                self.edge_norm = LayerNorm(edge_dims, eps=eps)
            else:
                self.edge_norm = nn.Identity()
            self.edge_dropout = Dropout(drop_rate)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`. 
                If not `None`, will be used as srcqq node embeddings
                for forming messages where src >= dst. The corrent node 
                embeddings `x` will still be the base of the update and the 
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''
        if self.edge_message_func:
            src, dst = edge_index
            if autoregressive_x is None:
                x_src = x[0][src], x[1][src]
            else: 
                mask = (src < dst).unsqueeze(-1)
                x_src = (
                    torch.where(mask, x[0][src], autoregressive_x[0][src]),
                    torch.where(mask.unsqueeze(-1), x[1][src],
                        autoregressive_x[1][src])
                )
            x_dst = x[0][dst], x[1][dst]
            x_edge = (
                torch.cat([x_src[0], edge_attr[0], x_dst[0]], dim=-1),
                torch.cat([x_src[1], edge_attr[1], x_dst[1]], dim=-2)
            )
            edge_attr_dh = self.edge_message_func(x_edge)
            edge_attr = self.edge_norm(tuple_sum(edge_attr,
                self.edge_dropout(edge_attr_dh)))
        
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )
            
            count = scatter_add(torch.ones_like(dst), dst,
                        dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)
        
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
            
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_

        return x, edge_attr


def unflatten_graph(node_embeddings, batch_size):
    """
    Unflattens node embeddings.
    Args:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch total_nodes x node_embed_dim
                - vector: shape batch total_nodes x node_embed_dim x 3
        batch_size: int
    Returns:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch size x nodes x node_embed_dim
                - vector: shape batch size x nodes x node_embed_dim x 3
    """
    x_s, x_v = node_embeddings
    x_s = x_s.reshape(batch_size, -1, x_s.shape[1])
    x_v = x_v.reshape(batch_size, -1, x_v.shape[1], x_v.shape[2])
    return (x_s, x_v)

def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)


def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    v_expand = torch.unsqueeze(values, -1)
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
            torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)

def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )

def rotate(v, R):
    """
    Rotates a vector by a rotation matrix.
    
    Args:
        v: 3D vector, tensor of shape (length x batch_size x channels x 3)
        R: rotation matrix, tensor of shape (length x batch_size x 3 x 3)

    Returns:
        Rotated version of v by rotation matrix R.
    """
    R = R.unsqueeze(-3)
    v = v.unsqueeze(-1)
    return torch.sum(v * R, dim=-2)

def get_rotation_frames(coords):
    """
    Returns a local rotation frame defined by N, CA, C positions.

    Args:
        coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
        where the third dimension is in order of N, CA, C

    Returns:
        Local relative rotation frames in shape (batch_size x length x 3 x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)
    return R

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

class GVPInputFeaturizer(nn.Module):

    @staticmethod
    def get_node_features(coords, coord_mask, with_coord_mask=True):
        # scalar features
        node_scalar_features = GVPInputFeaturizer._dihedrals(coords)
        if with_coord_mask:
            node_scalar_features = torch.cat([
                node_scalar_features,
                coord_mask.float().unsqueeze(-1)
            ], dim=-1) 
        # vector features
        X_ca = coords[:, :, 1]
        orientations = GVPInputFeaturizer._orientations(X_ca)
        sidechains = GVPInputFeaturizer._sidechains(coords)
        node_vector_features = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        return node_scalar_features, node_vector_features

    @staticmethod
    def _orientations(X):
        forward = normalize(X[:, 1:] - X[:, :-1])
        backward = normalize(X[:, :-1] - X[:, 1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)
    
    @staticmethod
    def _sidechains(X):
        n, origin, c = X[:, :, 0], X[:, :, 1], X[:, :, 2]
        c, n = normalize(c - origin), normalize(n - origin)
        bisector = normalize(c + n)
        perp = normalize(torch.cross(c, n, dim=-1))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 

    @staticmethod
    def _dihedrals(X, eps=1e-7):
        X = torch.flatten(X[:, :, :3], 1, 2)
        bsz = X.shape[0]
        dX = X[:, 1:] - X[:, :-1]
        U = normalize(dX, dim=-1)
        u_2 = U[:, :-2]
        u_1 = U[:, 1:-1]
        u_0 = U[:, 2:]
    
        # Backbone normals
        n_2 = normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
    
        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    
        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [bsz, -1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], -1)
        return D_features

    @staticmethod
    def _positional_embeddings(edge_index, 
                               num_embeddings=None,
                               num_positional_embeddings=16,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32,
                device=edge_index.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    @staticmethod
    def _dist(X, coord_mask, padding_mask, top_k_neighbors, eps=1e-8):
        """ Pairwise euclidean distances """
        bsz, maxlen = X.size(0), X.size(1)
        coord_mask_2D = torch.unsqueeze(coord_mask,1) * torch.unsqueeze(coord_mask,2)
        residue_mask = ~padding_mask
        residue_mask_2D = torch.unsqueeze(residue_mask,1) * torch.unsqueeze(residue_mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = coord_mask_2D * norm(dX, dim=-1)
    
        # sorting preference: first those with coords, then among the residues that
        # exist but are masked use distance in sequence as tie breaker, and then the
        # residues that came from padding are last
        seqpos = torch.arange(maxlen, device=X.device)
        Dseq = torch.abs(seqpos.unsqueeze(1) - seqpos.unsqueeze(0)).repeat(bsz, 1, 1)
        D_adjust = nan_to_num(D) + (~coord_mask_2D) * (1e8 + Dseq*1e6) + (
            ~residue_mask_2D) * (1e10)
    
        if top_k_neighbors == -1:
            D_neighbors = D_adjust
            E_idx = seqpos.repeat(
                    *D_neighbors.shape[:-1], 1)
        else:
            # Identify k nearest neighbors (including self)
            k = min(top_k_neighbors, X.size(1))
            D_neighbors, E_idx = torch.topk(D_adjust, k, dim=-1, largest=False)
    
        coord_mask_neighbors = (D_neighbors < 5e7)
        residue_mask_neighbors = (D_neighbors < 5e9)
        return D_neighbors, E_idx, coord_mask_neighbors, residue_mask_neighbors

def flatten_graph(node_embeddings, edge_embeddings, edge_index):
    """
    Flattens the graph into a batch size one (with disconnected subgraphs for
    each example) to be compatible with pytorch-geometric package.
    Args:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch size x nodes x node_embed_dim
                - vector: shape batch size x nodes x node_embed_dim x 3
        edge_embeddings: edge embeddings of in tuple form (scalar, vector)
                - scalar: shape batch size x edges x edge_embed_dim
                - vector: shape batch size x edges x edge_embed_dim x 3
        edge_index: shape batch_size x 2 (source node and target node) x edges
    Returns:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch total_nodes x node_embed_dim
                - vector: shape batch total_nodes x node_embed_dim x 3
        edge_embeddings: edge embeddings of in tuple form (scalar, vector)
                - scalar: shape batch total_edges x edge_embed_dim
                - vector: shape batch total_edges x edge_embed_dim x 3
        edge_index: shape 2 x total_edges
    """
    x_s, x_v = node_embeddings
    e_s, e_v = edge_embeddings
    batch_size, N = x_s.shape[0], x_s.shape[1]
    node_embeddings = (torch.flatten(x_s, 0, 1), torch.flatten(x_v, 0, 1))
    edge_embeddings = (torch.flatten(e_s, 0, 1), torch.flatten(e_v, 0, 1))

    edge_mask = torch.any(edge_index != -1, dim=1)
    # Re-number the nodes by adding batch_idx * N to each batch
    edge_index = edge_index + (torch.arange(batch_size, device=edge_index.device) *
            N).unsqueeze(-1).unsqueeze(-1)
    edge_index = edge_index.permute(1, 0, 2).flatten(1, 2)
    edge_mask = edge_mask.flatten()
    edge_index = edge_index[:, edge_mask] 
    edge_embeddings = (
        edge_embeddings[0][edge_mask, :],
        edge_embeddings[1][edge_mask, :]
    )
    return node_embeddings, edge_embeddings, edge_index 


def unflatten_graph(node_embeddings, batch_size):
    """
    Unflattens node embeddings.
    Args:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch total_nodes x node_embed_dim
                - vector: shape batch total_nodes x node_embed_dim x 3
        batch_size: int
    Returns:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch size x nodes x node_embed_dim
                - vector: shape batch size x nodes x node_embed_dim x 3
    """
    x_s, x_v = node_embeddings
    x_s = x_s.reshape(batch_size, -1, x_s.shape[1])
    x_v = x_v.reshape(batch_size, -1, x_v.shape[1], x_v.shape[2])
    return (x_s, x_v)

class GVPGraphEmbedding(GVPInputFeaturizer):

    def __init__(self, args):
        super().__init__()
        self.top_k_neighbors = args.top_k_neighbors
        self.num_positional_embeddings = 16
        self.remove_edges_without_coords = True
        node_input_dim = (7, 3)
        edge_input_dim = (34, 1)
        node_hidden_dim = (args.node_hidden_dim_scalar,
                args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                args.edge_hidden_dim_vector)
        self.embed_node = nn.Sequential(
            GVP(node_input_dim, node_hidden_dim, activations=(None, None)),
            LayerNorm(node_hidden_dim, eps=1e-4)
        )
        self.embed_edge = nn.Sequential(
            GVP(edge_input_dim, edge_hidden_dim, activations=(None, None)),
            LayerNorm(edge_hidden_dim, eps=1e-4)
        )
        self.embed_confidence = nn.Linear(16, args.node_hidden_dim_scalar)

    def forward(self, coords, coord_mask, padding_mask, confidence):
        with torch.no_grad():
            node_features = self.get_node_features(coords, coord_mask)
            edge_features, edge_index = self.get_edge_features(
                coords, coord_mask, padding_mask)
        node_embeddings_scalar, node_embeddings_vector = self.embed_node(node_features)
        edge_embeddings = self.embed_edge(edge_features)

        rbf_rep = rbf(confidence, 0., 1.)
        node_embeddings = (
            node_embeddings_scalar + self.embed_confidence(rbf_rep),
            node_embeddings_vector
        )

        node_embeddings, edge_embeddings, edge_index = flatten_graph(
            node_embeddings, edge_embeddings, edge_index)
        return node_embeddings, edge_embeddings, edge_index

    def get_edge_features(self, coords, coord_mask, padding_mask):
        X_ca = coords[:, :, 1]
        # Get distances to the top k neighbors
        E_dist, E_idx, E_coord_mask, E_residue_mask = GVPInputFeaturizer._dist(
                X_ca, coord_mask, padding_mask, self.top_k_neighbors)
        # Flatten the graph to be batch size 1 for torch_geometric package 
        dest = E_idx
        B, L, k = E_idx.shape[:3]
        src = torch.arange(L, device=E_idx.device).view([1, L, 1]).expand(B, L, k)
        # After flattening, [2, B, E]
        edge_index = torch.stack([src, dest], dim=0).flatten(2, 3)
        # After flattening, [B, E]
        E_dist = E_dist.flatten(1, 2)
        E_coord_mask = E_coord_mask.flatten(1, 2).unsqueeze(-1)
        E_residue_mask = E_residue_mask.flatten(1, 2)
        # Calculate relative positional embeddings and distance RBF 
        pos_embeddings = GVPInputFeaturizer._positional_embeddings(
            edge_index,
            num_positional_embeddings=self.num_positional_embeddings,
        )
        D_rbf = rbf(E_dist, 0., 20.)
        # Calculate relative orientation 
        X_src = X_ca.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
        X_dest = torch.gather(
            X_ca,
            1,
            edge_index[1, :, :].unsqueeze(-1).expand([B, L*k, 3])
        )
        coord_mask_src = coord_mask.unsqueeze(2).expand(-1, -1, k).flatten(1, 2)
        coord_mask_dest = torch.gather(
            coord_mask,
            1,
            edge_index[1, :, :].expand([B, L*k])
        )
        E_vectors = X_src - X_dest
        # For the ones without coordinates, substitute in the average vector
        E_vector_mean = torch.sum(E_vectors * E_coord_mask, dim=1,
                keepdims=True) / torch.sum(E_coord_mask, dim=1, keepdims=True)
        E_vectors = E_vectors * E_coord_mask + E_vector_mean * ~(E_coord_mask)
        # Normalize and remove nans 
        edge_s = torch.cat([D_rbf, pos_embeddings], dim=-1)
        edge_v = normalize(E_vectors).unsqueeze(-2)
        edge_s, edge_v = map(nan_to_num, (edge_s, edge_v))
        # Also add indications of whether the coordinates are present 
        edge_s = torch.cat([
            edge_s,
            (~coord_mask_src).float().unsqueeze(-1),
            (~coord_mask_dest).float().unsqueeze(-1),
        ], dim=-1)
        edge_index[:, ~E_residue_mask] = -1
        if self.remove_edges_without_coords:
            edge_index[:, ~E_coord_mask.squeeze(-1)] = -1
        return (edge_s, edge_v), edge_index.transpose(0, 1) 

class GVPEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_graph = GVPGraphEmbedding(args)

        node_hidden_dim = (args.node_hidden_dim_scalar,
                args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                args.edge_hidden_dim_vector)
        
        conv_activations = (F.relu, torch.sigmoid)
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(
                    node_hidden_dim,
                    edge_hidden_dim,
                    drop_rate=args.dropout,
                    vector_gate=True,
                    attention_heads=0,
                    n_message=3,
                    conv_activations=conv_activations,
                    n_edge_gvps=0,
                    eps=1e-4,
                    layernorm=True,
                ) 
            for i in range(args.num_encoder_layers)
        )

    def forward(self, coords, coord_mask, padding_mask, confidence):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
                coords, coord_mask, padding_mask, confidence)
        
        for i, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings,
                    edge_index, edge_embeddings)

        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
        return node_embeddings


from collections import OrderedDict
from torch._C import _disabled_torch_function_impl
class Parameter(torch.Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`locally-disable-grad-doc` for more details. Default: `True`
    """
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.tensor([])
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.requires_grad)
            memo[id(self)] = result
            return result

    def __repr__(self):
        return 'Parameter containing:\n' + super(Parameter, self).__repr__()

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (
            torch._utils._rebuild_parameter,
            (self.data, self.requires_grad, OrderedDict())
        )

    __torch_function__ = _disabled_torch_function_impl

from typing import Tuple

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, : x.shape[-2], :]
    sin = sin[:, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )

def utils_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)

from typing import Dict, Optional, Tuple, List, Sequence
from torch import Tensor, nn
import uuid
class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state

def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if (
            not self.rot_emb
            and self.enable_torch_version
            and not self.onnx_trace
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
            and not need_head_weights
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )

        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).type_as(attn).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(
                        0
                    ):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim : 2 * dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value



class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    `layernorm -> dropout -> add residual`

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.dropout_module = nn.Dropout(args.dropout)
        self.activation_fn = F.relu
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
            )

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.
    `layernorm -> dropout -> add residual`

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = nn.Dropout(args.dropout)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim

        self.activation_fn = F.relu

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.ffn_layernorm = (
            LayerNorm(args.decoder_ffn_embed_dim)
            if getattr(args, "scale_fc", False)
            else None
        )
        self.w_resid = (
            nn.Parameter(
                torch.ones(
                    self.embed_dim,
                ),
                requires_grad=True,
            )
            if getattr(args, "scale_resids", False)
            else None
        )

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.need_attn = True

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)

        residual = x
        x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        return x, attn, None


class GVPTransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__()
        self.args = args
        self.dictionary = dictionary

        self.dropout_module = nn.Dropout(args.dropout)

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim,
            self.padding_idx,
        )
        self.embed_gvp_input_features = nn.Linear(15, embed_dim)
        self.embed_confidence = nn.Linear(16, embed_dim)
        self.embed_dihedrals = DihedralFeatures(embed_dim)

        self.gvp_encoder = GVPEncoder(args)
        gvp_out_dim = args.node_hidden_dim_scalar + (3 *
                args.node_hidden_dim_vector)
        self.embed_gvp_output = nn.Linear(gvp_out_dim, embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(self, coords, padding_mask, confidence):
        """
        Args:
            coords: N, CA, C backbone coordinates in shape length x 3 (atoms) x 3 
            padding_mask: boolean Tensor (true for padding) of shape length
            confidence: confidence scores between 0 and 1 of shape length
        """
        components = dict()
        coord_mask = torch.all(torch.all(torch.isfinite(coords), dim=-1), dim=-1)
        coords = nan_to_num(coords)
        mask_tokens = (
            padding_mask * self.dictionary.pad_token_id + 
            ~padding_mask * self.dictionary.mask_token_id
        )
        components["tokens"] = self.embed_tokens(mask_tokens) * self.embed_scale
        components["diherals"] = self.embed_dihedrals(coords)

        # GVP encoder
        gvp_out_scalars, gvp_out_vectors = self.gvp_encoder(coords,
                coord_mask, padding_mask, confidence)
        R = get_rotation_frames(coords)
        # Rotate to local rotation frame for rotation-invariance
        gvp_out_features = torch.cat([
            gvp_out_scalars,
            rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        components["gvp_out"] = self.embed_gvp_output(gvp_out_features)

        components["confidence"] = self.embed_confidence(
             rbf(confidence, 0., 1.))

        # In addition to GVP encoder outputs, also directly embed GVP input node
        # features to the Transformer
        scalar_features, vector_features = GVPInputFeaturizer.get_node_features(
            coords, coord_mask, with_coord_mask=False)
        features = torch.cat([
            scalar_features,
            rotate(vector_features, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        components["gvp_input_features"] = self.embed_gvp_input_features(features)

        embed = sum(components.values())
        # for k, v in components.items():
        #     print(k, torch.mean(v, dim=(0,1)), torch.std(v, dim=(0,1)))

        x = embed
        x = x + self.embed_positions(mask_tokens)
        x = self.dropout_module(x)
        return x, components 

    def forward(
        self,
        coords,
        encoder_padding_mask,
        confidence,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            coords (Tensor): backbone coordinates
                shape batch_size x num_residues x num_atoms (3 for N, CA, C) x 3
            encoder_padding_mask (ByteTensor): the positions of
                  padding elements of shape `(batch_size x num_residues)`
            confidence (Tensor): the confidence score of shape (batch_size x
                num_residues). The value is between 0. and 1. for each residue
                coordinate, or -1. if no coordinate is given
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(num_residues, batch_size, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch_size, num_residues)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch_size, num_residues, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(num_residues, batch_size, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(coords,
                encoder_padding_mask, confidence)
        # account for padding while computing the representation
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # dictionary
            "encoder_states": encoder_states,  # List[T x B x C]
        }

class TransformerDecoder(nn.Module):
    """
    Transformer decoder consisting of *args.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
    ):
        super().__init__()
        self.args = args
        self.dictionary = dictionary
        self._future_mask = torch.empty(0)

        self.dropout_module = nn.Dropout(args.dropout)

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim

        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)

        self.project_in_dim = (
            nn.Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim,
            self.padding_idx,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.build_output_projection(args, dictionary)

    def build_output_projection(self, args, dictionary):
        self.output_projection = nn.Linear(
            args.decoder_embed_dim, len(dictionary), bias=False
        )
        nn.init.normal_(
            self.output_projection.weight, mean=0, std=args.decoder_embed_dim ** -0.5
        )

    def build_decoder_layer(self, args):
        return TransformerDecoderLayer(args)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )

        if not features_only:
            x = self.output_layer(x)
        x = x.transpose(1, 2) # B x T x C -> B x C x T
        x = torch.nan_to_num(x, 0)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        x += positions

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=False,
                need_head_weights=False,
            )
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x C x T
        x = x.transpose(0, 1)

        return x, {"inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        return self.output_projection(features)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens

class CoordBatchConverter(BatchConverter):
    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x 3 x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>") 
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))

        coords_and_confidence, strs, tokens = super().__call__(batch)

        # pad beginning and end of each protein due to legacy reasons
        coords = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.)
            for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
        padding_mask = torch.isnan(coords[:,:,0,0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.) * padding_mask
        return coords, confidence, strs, tokens, padding_mask

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result

proteinseq_toks = {
    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
}

import itertools
class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self):
        return BatchConverter(self)

    @classmethod
    def from_architecture(cls) -> "Alphabet":
        standard_toks = proteinseq_toks["toks"]
        prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
        append_toks = ("<mask>", "<cath>", "<af2>")
        prepend_bos = True
        append_eos = False
        use_msa = False
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]
