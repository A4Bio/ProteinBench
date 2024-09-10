import time
import torch

import torch.nn as nn
import numpy as np
from src.modules.uniif_module import *
# from transformers import AutoTokenizer
from src.datasets.protein_rna_dataset import MyTokenizer
from src.tools.affine_utils import Rigid, Rotation, get_interact_feats
from torch_geometric.nn.pool import knn_graph

class UniIF_Model(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(UniIF_Model, self).__init__()
        self.__dict__.update(locals())
        geo_layer, attn_layer, node_layer, edge_layer, encoder_layer, hidden_dim, dropout, mask_rate = args.geo_layer, args.attn_layer, args.node_layer, args.edge_layer, args.encoder_layer, args.hidden_dim, args.dropout, args.mask_rate
        self.tokenizer = MyTokenizer()
    

        if args['dataname']=='RNA':
            self.node_embedding = build_MLP(2, 114, hidden_dim, hidden_dim)
            self.edge_embedding = build_MLP(2, 272, hidden_dim, hidden_dim)
        if args['dataname']=='Protein':
            self.node_embedding = build_MLP(2, 76, hidden_dim, hidden_dim)
            self.edge_embedding = build_MLP(2, 196, hidden_dim, hidden_dim)
        if args['dataname']=='pdb':
            self.node_embedding = build_MLP(2, 76, hidden_dim, hidden_dim)
            self.edge_embedding = build_MLP(2, 196+16, hidden_dim, hidden_dim)
        self.virtual_embedding = nn.Embedding(30, hidden_dim) 
        self.encoder = StructureEncoder(geo_layer, attn_layer, node_layer, edge_layer, encoder_layer, hidden_dim, dropout, mask_rate)
        self.decoder = MLPDecoder(hidden_dim)
        self.chain_embeddings = nn.Embedding(2, 16)

        self._init_params()

    def _init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, batch, num_global = 3):
        X, h_V, h_E, edge_idx, batch_id, chain_features = batch['X'], batch['_V'], batch['_E'], batch['edge_idx'], batch['batch_id'], batch['chain_features']
        edge_idx_g, batch_id_g = batch['edge_idx_g'], batch['batch_id_g']
        T = Rigid(Rotation(batch['T_rot']), batch['T_trans'])
        T_g = Rigid(Rotation(batch['T_g_rot']), batch['T_g_trans'])
        T_ts = Rigid(Rotation(batch['T_ts_rot']), batch['T_ts_trans'])
        T_gs = Rigid(Rotation(batch['T_gs_rot']), batch['T_gs_trans'])
        rbf_ts, rbf_gs = batch['rbf_ts'], batch['rbf_gs']
        T_gs.rbf = rbf_gs
        T_ts.rbf = rbf_ts
        h_E = torch.cat([h_E, self.chain_embeddings(chain_features)], dim=-1)

        h_E_0 = h_E

        h_V = self.node_embedding(h_V)
        h_E = self.edge_embedding(h_E)
        h_V_g = self.virtual_embedding(batch['_V_g'])
        h_E_g = torch.zeros((edge_idx_g.shape[1], h_E.shape[1]), device=h_V.device, dtype=h_V.dtype)
        h_S = None
        h_E_0 = torch.cat([h_E_0, torch.zeros((edge_idx_g.shape[1], h_E_0.shape[1]), device=h_V.device, dtype=h_V.dtype)])


        h_V = self.encoder(h_S,
                                T, T_g, 
                                h_V, h_V_g,
                                h_E, h_E_g,
                                T_ts, T_gs, 
                                edge_idx, edge_idx_g,
                                batch_id, batch_id_g, h_E_0)
        log_probs, logits = self.decoder(h_V)

        return {'log_probs': log_probs, 'logits':logits}
        

    def _get_features(self, batch):
        return batch

        