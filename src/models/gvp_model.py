import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_scatter import scatter_mean

from src.modules.gvp_module import GVP, GVPConvLayer, LayerNorm, tuple_index
from src.tools import gather_nodes, _dihedrals, _get_rbf, _get_dist, _rbf, _orientations_coarse_gl_tuple
import numpy as np
class GVP_Model(nn.Module):
    '''
    GVP-GNN for structure-conditioned autoregressive 
    protein design as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 20 amino acids at each position in a `torch.Tensor` of 
    shape [n_nodes, 20].
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers in each of the encoder
                       and decoder modules
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, args, drop_rate=0.1):
    
        super(GVP_Model, self).__init__()
        self.args = args
        self.node_in_dim = (6, 3)
        self.node_h_dim = (100, 16)
        self.edge_in_dim = (32, 1)
        self.edge_h_dim = (32, 1)
        self.num_layers = 3
        self.W_v = nn.Sequential(
            GVP(self.node_in_dim, self.node_h_dim, activations=(None, None)),
            LayerNorm(self.node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(self.edge_in_dim, self.edge_h_dim, activations=(None, None)),
            LayerNorm(self.edge_h_dim)
        )
        
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(self.node_h_dim, self.edge_h_dim, drop_rate=drop_rate) 
            for _ in range(self.num_layers))
        
        self.W_s = nn.Embedding(33, 20)
        self.edge_h_dim = (self.edge_h_dim[0] + 20, self.edge_h_dim[1])
      
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(self.node_h_dim, self.edge_h_dim, 
                             drop_rate=drop_rate, autoregressive=True) 
            for _ in range(self.num_layers))
        
        self.W_out = GVP(self.node_h_dim, (33, 0), activations=(None, None))

        self.encode_t = 0
        self.decode_t = 0

    def _get_features(self, batch):
        return batch

    def forward(self, batch):
        '''
        Forward pass to be used at train-time, or evaluating likelihood.
        
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: int `torch.Tensor` of shape [num_nodes]
        '''
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        seq = batch.seq
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        
        encoder_embeddings = h_V
        
        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x = encoder_embeddings)
        
        logits = self.W_out(h_V)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return {'log_probs': log_probs, 'logits': logits}

    # def forward(self, batch):
    #     '''
    #     Forward pass to be used at train-time, or evaluating likelihood.
        
    #     :param batch: a dictionary containing the following keys:
    #         - h_V: tuple (s, V) of node embeddings
    #         - h_E: tuple (s, V) of edge embeddings
    #         - edge_index: `torch.Tensor` of shape [2, num_edges]
    #         - seq: int `torch.Tensor` of shape [num_nodes]
    #     '''

    #     h_V = batch['h_V']
    #     h_E = batch['h_E']
    #     edge_index = batch['E_idx']
    #     seq = batch['S']
        

    #     h_V = self.W_v(h_V)
    #     h_E = self.W_e(h_E)
        

    #     for layer in self.encoder_layers:
    #         h_V = layer(h_V, edge_index, h_E)
        
    #     encoder_embeddings = h_V
        

    #     h_S = self.W_s(seq)
    #     h_S = h_S[edge_index[0]]
    #     h_S[edge_index[0] >= edge_index[1]] = 0
    #     h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        

    #     for layer in self.decoder_layers:
    #         h_V = layer(h_V, edge_index, h_E, autoregressive_x = encoder_embeddings)
        

    #     logits = self.W_out(h_V)
    #     log_probs = nn.functional.log_softmax(logits, dim=-1)
        
    #     return {'log_probs': log_probs, 'logits': logits}

    def sample(self, h_V, edge_index, h_E, n_samples, temperature=0.1):
        '''
        Samples sequences auto-regressively from the distribution
        learned by the model.
        
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param n_samples: number of samples
        :param temperature: temperature to use in softmax 
                            over the categorical distribution
        
        :return: int `torch.Tensor` of shape [n_samples, n_nodes] based on the
                 residue-to-int mapping of the original training data
        '''
        
        with torch.no_grad():
            device = edge_index.device
            L = h_V[0].shape[0]

            h_V = self.W_v(h_V)
            h_E = self.W_e(h_E)
            
            for layer in self.encoder_layers:
                h_V = layer(h_V, edge_index, h_E)   

            h_V = (h_V[0].repeat(n_samples, 1),
                   h_V[1].repeat(n_samples, 1, 1))
            
            h_E = (h_E[0].repeat(n_samples, 1),
                   h_E[1].repeat(n_samples, 1, 1))
            
            edge_index = edge_index.expand(n_samples, -1, -1)
            offset = L * torch.arange(n_samples, device=device).view(-1, 1, 1)
            edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
            
            seq = torch.zeros(n_samples * L, device=device, dtype=torch.int)
            h_S = torch.zeros(n_samples * L, 20, device=device)
    
            h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]
            
            all_probs = []
            for i in range(L):
                h_S_ = h_S[edge_index[0]]
                h_S_[edge_index[0] >= edge_index[1]] = 0
                h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
                        
                edge_mask = edge_index[1] % L == i
                edge_index_ = edge_index[:, edge_mask]
                h_E_ = tuple_index(h_E_, edge_mask)
                node_mask = torch.zeros(n_samples * L, device=device, dtype=torch.bool)
                node_mask[i::L] = True
                
                for j, layer in enumerate(self.decoder_layers):
                    out = layer(h_V_cache[j], edge_index_, h_E_,
                               autoregressive_x=h_V_cache[0], node_mask=node_mask)
                    
                    out = tuple_index(out, node_mask)
                    
                    if j < len(self.decoder_layers)-1:
                        h_V_cache[j+1][0][i::L] = out[0]
                        h_V_cache[j+1][1][i::L] = out[1]
                    
                logits = self.W_out(out)
                seq[i::L] = Categorical(logits=logits / temperature).sample()
                h_S[i::L] = self.W_s(seq[i::L])
                all_probs.append(torch.softmax(logits, dim=-1))

            self.probs = torch.cat(all_probs, dim=0)
            return seq.view(n_samples, L)
    
    def test_recovery(self, protein):
        h_V = (protein.node_s, protein.node_v)
        h_E = (protein.edge_s, protein.edge_v) 
        sample = self.sample(h_V, protein.edge_index, h_E, n_samples=1)
        return sample.squeeze(0)
    
 