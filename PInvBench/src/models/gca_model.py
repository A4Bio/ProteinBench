import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_min

from ..tools.design_utils import gather_nodes, _dihedrals, _rbf, _orientations_coarse_gl
from ..modules.graphtrans_module import *
from ..modules.gca_module import Local_Module, Global_Module


class GCA_Model(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(GCA_Model, self).__init__()
        self.node_features = args.hidden
        self.edge_features = args.hidden
        self.hidden = args.hidden
        self.top_k = args.k_neighbors
        self.num_rbf = 16
        self.num_positional_embeddings = 16

        vocab = args.vocab_size
        num_encoder_layers = args.num_encoder_layers
        num_decoder_layers = args.num_decoder_layers
        is_attention = args.is_attention
        dropout = args.dropout

        # node_in, edge_in = 6, 39 - 16
        node_in, edge_in = 12, 39 - 16
        self.embeddings = PositionalEncodings(self.num_positional_embeddings)
        self.node_embedding = nn.Linear(node_in, self.node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, self.edge_features, bias=True)
        self.norm_nodes = Normalize(self.node_features)
        self.norm_edges = Normalize(self.edge_features)

        self.W_v = nn.Linear(self.node_features, self.hidden, bias=True)
        self.W_e = nn.Linear(self.edge_features, self.hidden, bias=True)
        self.W_f = nn.Linear(self.edge_features, self.hidden, bias=True)
        self.W_s = nn.Embedding(vocab, self.hidden)

        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_encoder_layers):
            self.encoder_layers.append(nn.ModuleList([
                Local_Module(self.hidden, self.hidden*2, is_attention=is_attention, dropout=dropout),
                Global_Module(self.hidden, self.hidden*2, dropout=dropout)
            ]))

        self.decoder_layers = nn.ModuleList([])
        for _ in range(num_decoder_layers):
            self.decoder_layers.append(
                Local_Module(self.hidden, self.hidden*3, is_attention=is_attention, dropout=dropout)
            )

        self.W_out = nn.Linear(self.hidden, vocab, bias=True)
        self._init_params()

        self.encode_t = 0
        self.decode_t = 0

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _autoregressive_mask(self, E_idx):
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes)
        ii = ii.view((1, -1, 1)).to(E_idx.device)
        mask = E_idx - ii < 0
        mask = mask.type(torch.float32)
        return mask

    def _get_encoder_mask(self, idx, mask):
        mask_attend = gather_nodes(mask.unsqueeze(-1), idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend 
        return mask_attend

    def _get_decoder_mask(self, idx, mask):
        mask_attend = self._autoregressive_mask(idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        return mask_bw, mask_fw

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _encoder_network(self, h_V, h_P, h_F, P_idx, F_idx, mask):
        '''
        h_V: [batch, num_nodes, 128]
        h_P: [batch, num_nodes, K, 128]
        h_F: [batch, num_nodes, num_nodes, 128]
        P_idx: [batch, num_nodes, K]
        F_idx: [batch, num_nodes, num_nodes]
        mask: [batch, num_nodes]
        '''
        P_idx_mask_attend = self._get_encoder_mask(P_idx, mask) # part
        F_idx_mask_attend = self._get_encoder_mask(F_idx, mask) # full
        for (local_layer, global_layer) in self.encoder_layers:
            # local_layer
            h_EV_local = cat_neighbors_nodes(h_V, h_P, P_idx) # [4, 312, 30, 256]
            h_V = local_layer(h_V, h_EV_local, mask_V=mask, mask_attend=P_idx_mask_attend)
            # global layer
            h_EV_global = cat_neighbors_nodes(h_V, h_F, F_idx)
            h_V = h_V + global_layer(h_V, h_EV_global, mask_V=mask, mask_attend=F_idx_mask_attend)
        return h_V

    def _get_sv_encoder(self, S, h_V, h_P, P_idx):
        h_S = self.W_s(S)
        h_PS = cat_neighbors_nodes(h_S, h_P, P_idx)
        h_PS_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_P, P_idx) 
        h_PSV_encoder = cat_neighbors_nodes(h_V, h_PS_encoder, P_idx)
        return h_PS, h_PSV_encoder

    def _get_features(self, batch):
        S, X, mask, chain_mask = batch['S'], batch['X'], batch['mask'], batch['chain_mask']
        X_ca = X[:,:,1,:]
        D_neighbors, F_idx = self._full_dist(X_ca, mask, 500)
        P_idx = F_idx[:, :, :self.top_k].clone()

        _V = _dihedrals(X) # node feature
        _V = self.norm_nodes(self.node_embedding(_V))
        _F = torch.cat((_rbf(D_neighbors, self.num_rbf), _orientations_coarse_gl(X, F_idx)), -1)
        _F = self.norm_edges(self.edge_embedding(_F))
        _P = _F[..., :self.top_k, :]
    
        h_V = self.W_v(_V)
        h_P, h_F = self.W_e(_P), self.W_f(_F)

        batch.update({'S':S,
                'h_V': h_V, 
                'h_P': h_P, 
                'h_F': h_F,
                'P_idx': P_idx,
                'F_idx': F_idx,
                'mask': mask})

        return batch
    
    def sparse_to_dense(self, S, h_V, h_P, edge_idx_P, h_F, edge_idx_F, batch_id):
        device = h_V.device
        num_nodes = scatter_sum(torch.ones_like(batch_id), batch_id)
        batch = num_nodes.shape[0]
        N = num_nodes.max()
        
        S_ = torch.zeros([batch, N], device=device).long()
        row = batch_id
        col = torch.cat([torch.arange(0,n) for n in num_nodes]).to(device)
        S_[row, col] = S
        S = S_
        
        # node feature
        dim_V = h_V.shape[-1]
        h_V_ = torch.zeros([batch, N, dim_V], device=device)
        row = batch_id
        col = torch.cat([torch.arange(0,n) for n in num_nodes]).to(device)
        h_V_[row, col] = h_V
        h_V = h_V_
        
        mask = torch.zeros([batch, N], device=device)
        mask[row, col] = 1
        
        # edge feature
        K = 30
        dim_P = h_P.shape[-1]
        h_P_ = torch.zeros([batch, N, K, dim_P], device=device)
        row2 = batch_id[edge_idx_P[0]]
        batch_shift, _ = scatter_min(edge_idx_P[0], batch_id[edge_idx_P[0]])
        local_dst_idx = edge_idx_P[0] - batch_shift[batch_id[edge_idx_P[0]]]
        local_src_idx = edge_idx_P[1] - batch_shift[batch_id[edge_idx_P[1]]]
        
        nn_num = scatter_sum(torch.ones_like(edge_idx_P[0]), edge_idx_P[0])
        nn_idx = torch.cat([torch.arange(0,n) for n in nn_num]).to(device)
        h_P_[row2, local_dst_idx, nn_idx] = h_P
        h_P = h_P_
        
        nn_num = scatter_sum(torch.ones_like(edge_idx_P[0]), edge_idx_P[0])
        nn_idx = torch.cat([torch.arange(0,n) for n in nn_num]).to(device)
        
        P_idx = torch.arange(0, K, device=device).reshape(1,1,K).repeat(batch, N, 1)
        P_idx[row2, local_dst_idx, nn_idx] = local_src_idx
        
        
        # edge feature
        K = N
        dim_F = h_F.shape[-1]
        h_F_ = torch.zeros([batch, N, K, dim_F], device=device)
        row2 = batch_id[edge_idx_F[0]]
        batch_shift, _ = scatter_min(edge_idx_F[0], batch_id[edge_idx_F[0]])
        local_dst_idx = edge_idx_F[0] - batch_shift[batch_id[edge_idx_F[0]]]
        local_src_idx = edge_idx_F[1] - batch_shift[batch_id[edge_idx_F[1]]]
        
        nn_num = scatter_sum(torch.ones_like(edge_idx_F[0]), edge_idx_F[0])
        nn_idx = torch.cat([torch.arange(0,n) for n in nn_num]).to(device)
        h_F_[row2, local_dst_idx, nn_idx] = h_F
        h_F = h_F_
        
        nn_num = scatter_sum(torch.ones_like(edge_idx_F[0]), edge_idx_F[0])
        nn_idx = torch.cat([torch.arange(0,n) for n in nn_num]).to(device)
        
        F_idx = torch.arange(0, K, device=device).reshape(1,1,K).repeat(batch, N, 1)
        F_idx[row2, local_dst_idx, nn_idx] = local_src_idx
        
        return S, h_V, h_P, h_F, P_idx,F_idx, mask
    
    def forward(self, batch):
        h_V, h_P, h_F, P_idx, F_idx, S, mask = batch['h_V'], batch['h_P'], batch['h_F'], batch['P_idx'], batch['F_idx'], batch['S'], batch['mask']
        
        t1 = time.time()
        h_V = self._encoder_network(h_V, h_P, h_F, P_idx, F_idx, mask)
        h_PS, h_PSV_encoder = self._get_sv_encoder(S, h_V, h_P, P_idx)
        t2 = time.time()
        
        # Decoder
        P_idx_mask_bw, P_idx_mask_fw = self._get_decoder_mask(P_idx, mask)
        for local_layer in self.decoder_layers:
            # local_layer
            h_PSV_local = cat_neighbors_nodes(h_V, h_PS, P_idx)
            h_PSV_local = P_idx_mask_bw * h_PSV_local + P_idx_mask_fw * h_PSV_encoder
            h_V = local_layer(h_V, h_PSV_local, mask_V=mask)
        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        t3 = time.time()

        self.encode_t += t2-t1
        self.decode_t += t3-t2
        return {'log_probs':log_probs}

    def sample(self, h_V, h_P, h_F, P_idx, F_idx, mask=None, temperature=0.1, **kwargs):
        t1 = time.time()
        h_V = self._encoder_network(h_V, h_P, h_F, P_idx, F_idx, mask)
        t2 = time.time()

        # Decoder
        P_idx_mask_bw, P_idx_mask_fw = self._get_decoder_mask(P_idx, mask)
        N_batch, N_nodes = h_V.size(0), h_V.size(1)
        h_S = torch.zeros_like(h_V)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=h_V.device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.decoder_layers))]
        all_probs = []
        for t in range(N_nodes):
            # Hidden layers
            P_idx_t = P_idx[:,t:t+1,:]
            h_P_t = h_P[:,t:t+1,:,:]
            h_PS_t = cat_neighbors_nodes(h_S, h_P_t, P_idx_t)
            h_PSV_encoder_t = P_idx_mask_fw[:,t:t+1,:,:] * cat_neighbors_nodes(h_V, h_PS_t, P_idx_t)
            for l, local_layer in enumerate(self.decoder_layers):
                # local layer
                h_PSV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_PS_t, P_idx_t)
                h_V_t = h_V_stack[l][:,t:t+1,:]
                h_PSV_t = P_idx_mask_bw[:,t:t+1,:,:] * h_PSV_decoder_t + h_PSV_encoder_t
                
                h_V_stack[l+1][:,t,:] = local_layer(
                    h_V_t, h_PSV_t, mask_V=mask[:, t:t+1]
                ).squeeze(1)
            # Sampling step
            h_V_t = h_V_stack[-1][:,t,:]
            logits = self.W_out(h_V_t) / temperature
            probs = F.softmax(logits, dim=-1)
            S_t = torch.multinomial(probs, 1).squeeze(-1)
            # Update
            h_S[:,t,:] = self.W_s(S_t)
            S[:,t] = S_t
            
            all_probs.append(probs)
        
        self.probs = torch.cat(all_probs, dim=0)
        
        t3 = time.time()
        self.encode_t += t2-t1
        self.decode_t += t3-t2
        return S