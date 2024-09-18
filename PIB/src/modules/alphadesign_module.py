import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_softmax

from ..tools import gather_nodes


def get_attend_mask(idx, mask):
    mask_attend = gather_nodes(mask.unsqueeze(-1), idx).squeeze(-1) # 一阶邻居节点的mask: 1代表节点存在, 0代表节点不存在
    mask_attend = mask.unsqueeze(-1) * mask_attend # 自身的mask*邻居节点的mask
    return mask_attend

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

    def forward(self, h_V, h_E, center_id, batch_id):
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


class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, edge_drop=0.0):
        super(NeighborAttention, self).__init__()
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

    def forward(self, h_V, h_E, center_id, batch_id):
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        
        w = self.Bias(torch.cat([h_V[center_id], h_E],dim=-1)).view(E, n_heads, 1) 
        attend_logits = w/np.sqrt(d) 

        V = self.W_V(h_E).view(-1, n_heads, d) 
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        h_V = scatter_sum(attend*V, center_id, dim=0).view([N, self.num_hidden])

        h_V_update = self.W_O(h_V)
        return h_V_update

class GNNModule(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0, use_SGT=False):
        super(GNNModule, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(2)])
        if use_SGT:
            self.attention = NeighborAttention(num_hidden, num_in, num_heads, edge_drop=0.0) # TODO: edge_drop
        else:
            self.attention = QKV(num_hidden, num_in, num_heads, edge_drop=0.0)
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
    
class StructureEncoder(nn.Module):
    def __init__(self,  hidden_dim, num_encoder_layers=3, dropout=0, use_SGT=False):
        """ Graph labeling network """
        super(StructureEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_encoder_layers):
            self.encoder_layers.append(nn.ModuleList([
                # Local_Module(hidden_dim, hidden_dim*2, is_attention=is_attention, dropout=dropout),
                GNNModule(hidden_dim, hidden_dim*2, dropout=dropout, use_SGT=use_SGT),
                GNNModule(hidden_dim, hidden_dim*2, dropout=dropout, use_SGT=use_SGT)
            ]))

    def forward(self, h_V, h_P, P_idx, batch_id):
        h_V = h_V
        # graph encoder
        for (layer1, layer2) in self.encoder_layers:
            h_EV_local = torch.cat([h_P, h_V[P_idx[1]]], dim=1)
            h_V = layer1(h_V, h_EV_local, P_idx, batch_id)
            
            h_EV_global = torch.cat([h_P, h_V[P_idx[1]]], dim=1)
            h_V = h_V + layer2(h_V, h_EV_global, P_idx, batch_id)
        return h_V

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

class CNNDecoder(nn.Module):
    def __init__(self,hidden_dim, input_dim, vocab=33):
        super().__init__()
        self.CNN = nn.Sequential(nn.Conv1d(input_dim, hidden_dim,5, padding=2),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Conv1d(hidden_dim, hidden_dim,5, padding=2),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Conv1d(hidden_dim, hidden_dim,5, padding=2))

        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, batch_id):
        h_V = h_V.unsqueeze(0).permute(0,2,1)
        hidden = self.CNN(h_V).permute(0,2,1).squeeze()
        logits = self.readout( hidden )
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits

class CNNDecoder2(nn.Module):
    def __init__(self,hidden_dim, input_dim, vocab=33):
        super().__init__()
        self.ConfNN = nn.Embedding(50, hidden_dim)
        
        self.CNN = nn.Sequential(nn.Conv1d(hidden_dim+input_dim, hidden_dim,5, padding=2),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Conv1d(hidden_dim, hidden_dim,5, padding=2),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Conv1d(hidden_dim, hidden_dim,5, padding=2))

        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, logits, batch_id):
        eps = 1e-5
        L = h_V.shape[0]
        idx = torch.argsort(-logits, dim=1)
        Conf = logits[range(L), idx[:,0]] / (logits[range(L), idx[:,1]] + eps)
        Conf = Conf.long()
        Conf = torch.clamp(Conf, 0, 49)
        h_C = self.ConfNN(Conf)
        
        # pos = self.PosEnc(pos)
        h_V = torch.cat([h_V,h_C],dim=-1)
        h_V = h_V.unsqueeze(0).permute(0,2,1)
        hidden = self.CNN(h_V).permute(0,2,1).squeeze()
        logits = self.readout( hidden )
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

from .graphtrans_module import Normalize
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

class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers=3, kernel_size=5, act_type='relu', glu=0, vocab=33):
        super().__init__()
        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, batch_id=None):
        logits = self.readout(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits

class ATDecoder(nn.Module):
    def __init__(self, args, hidden_dim, dropout=0.1, vocab=33):
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
        
        known = torch.zeros_like(P_idx[0])==1
        for t in range(L):
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