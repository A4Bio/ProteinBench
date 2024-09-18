import torch
import torch.nn as nn

from ..modules.alphadesign_module import ATDecoder, CNNDecoder, CNNDecoder2, StructureEncoder
from ..tools.design_utils import gather_nodes, _dihedrals, _rbf, _orientations_coarse_gl


class AlphaDesign_Model(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(AlphaDesign_Model, self).__init__()
        self.args = args
        node_features = args.node_features
        edge_features = args.edge_features
        hidden_dim = args.hidden_dim
        dropout = args.dropout
        num_encoder_layers = args.num_encoder_layers
        self.top_k = args.k_neighbors
        self.num_rbf = 16
        self.num_positional_embeddings = 16

        if args.use_new_feat:
            node_in, edge_in = 12, 16+7
        else:
            node_in, edge_in = 6, 16+7
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = nn.BatchNorm1d(node_features)
        self.norm_edges = nn.BatchNorm1d(edge_features)

        self.W_v = nn.Sequential(
            nn.Linear(node_features, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )
        
        
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True) 
        self.W_f = nn.Linear(edge_features, hidden_dim, bias=True)

        self.encoder = StructureEncoder(hidden_dim, num_encoder_layers, dropout, use_SGT=self.args.use_SGT)

        if args.autoregressive:
            self.decoder = ATDecoder(args, hidden_dim, dropout)
        else:
            self.decoder = CNNDecoder(hidden_dim, hidden_dim)
            self.decoder2 = CNNDecoder2(hidden_dim, hidden_dim)
            
        # self.chain_embed = nn.Embedding(2,16)
        self._init_params()
    
    def forward(self, batch,  AT_test = False, return_logit=False):
        h_V, h_P, P_idx, batch_id = batch['_V'], batch['_E'], batch['E_idx'], batch['batch_id']
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))
        
        h_V = self.encoder(h_V, h_P, P_idx, batch_id)
        log_probs0 = None
        if AT_test:
            log_probs = self.decoder.sampling(h_V, h_P, P_idx,  batch_id)
        else:
            log_probs0, logits = self.decoder(h_V, batch_id)
            log_probs, logits = self.decoder2(h_V, logits, batch_id)
        if return_logit:
            return {'log_probs': log_probs, 'logits': logits}
        else:
            return {'log_probs': log_probs, 'log_probs0': log_probs0}
        
    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx  

    def _get_features(self, batch):
        S, score, X, mask = batch['S'], batch['score'], batch['X'], batch['mask']
        mask_bool = (mask==1)
        
        B, N, _,_ = X.shape
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx = self._full_dist(X_ca, mask, self.top_k)

        # sequence
        S = torch.masked_select(S, mask_bool)
        if score is not None:
            score = torch.masked_select(score, mask_bool)

        # node feature
        _V = _dihedrals(X) 
        if not self.args.use_new_feat:
            _V = _V[...,:6]
        _V = torch.masked_select(_V, mask_bool.unsqueeze(-1)).reshape(-1,_V.shape[-1])

        # edge feature
        _E = torch.cat((_rbf(D_neighbors, self.num_rbf), _orientations_coarse_gl(X, E_idx)), -1) # [4,387,387,23]
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1) # 一阶邻居节点的mask: 1代表节点存在, 0代表节点不存在
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1 # 自身的mask*邻居节点的mask
        _E = torch.masked_select(_E, mask_attend.unsqueeze(-1)).reshape(-1,_E.shape[-1])
        
        # edge index
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B,1,1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1,-1)
        dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1,-1)
        E_idx = torch.cat((dst, src), dim=0).long()
        
        
        # 3D point
        sparse_idx = mask.nonzero()
        X = X[sparse_idx[:,0],sparse_idx[:,1],:,:]
        batch_id = sparse_idx[:,0]

        mask = torch.masked_select(mask, mask_bool)

        batch.update({'X':X,
                'S':S,
                'score':score,
                '_V':_V,
                '_E':_E,
                'E_idx':E_idx,
                'batch_id': batch_id,
                'mask':mask})

        return batch