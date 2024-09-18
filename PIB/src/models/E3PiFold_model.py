import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from ..modules.E3PiFold import GaussianEncoder, TransformerEncoderWithPair
from ..tools import gather_nodes, _dihedrals, _get_rbf, _get_dist, _rbf, _orientations_coarse_gl_tuple

class E3PiFold(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.node_embed = nn.Linear(21, config.embed_dim)
        self.protein_embedder = GaussianEncoder(config.kernel_num, config.embed_dim, config.attention_heads, config.use_dist, config.use_product)

        self.encoder = TransformerEncoderWithPair(
            config.encoder_layers,
            config.embed_dim,
            config.ffn_embed_dim,
            config.attention_heads,
            config.emb_dropout,
            config.dropout,
            config.attention_dropout,
            config.activation_dropout,
            config.max_seq_len,
        )
        self.predictor = nn.Linear(config.embed_dim, 33)
    
    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx 
    
    def _get_features(self, batch):
        X = batch['X']
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx = self._full_dist(X_ca, batch['mask'], 30)
        V_angles = _dihedrals(X.float())
        V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X.float(), E_idx)
        h_V = torch.cat([V_angles, V_direct], dim=-1).to(X.dtype)
        batch['h_V'] = h_V 
        return batch
    
    def forward(self, batch):
        '''
        X, H, seq_mask
        '''
        X = batch['X'][:,:,1]
        H = self.node_embed(batch['h_V'])
        seq_mask = batch['mask']
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        padding_mask = 1 - seq_mask
        x, graph_attn_bias = self.protein_embedder(X, H, pair_mask)
        (
            encoder_rep, 
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias, pair_mask=pair_mask)
        logits = self.predictor(x)
        log_probs = F.log_softmax(logits, dim=-1)

        return {'log_probs': log_probs}


if __name__ == '__main__':
    B, N, dim = 16, 512, 768
    X = torch.randn(B, N, 3)
    H = torch.randn(B, N, dim)
    seq_mask = (torch.ones(B, N)>0.5).float()

    config = {'encoder_layers': 12,
              'kernel_num':16,
              'embed_dim': 768,
              'ffn_embed_dim': 3072,
              'attention_heads': 8,
              'emb_dropout': 0.1,
              'dropout': 0.1,
              'attention_dropout': 0.1,
              'activation_dropout': 0.0,
              'max_seq_len': 256}
    config = OmegaConf.create(config)
    model = E3PiFold(config)
    feat = model(X, H, seq_mask)