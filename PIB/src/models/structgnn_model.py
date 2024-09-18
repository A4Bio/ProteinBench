import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.graphtrans_module import Struct2Seq, cat_neighbors_nodes, gather_nodes, ProteinFeatures


class StructGNN_Model(nn.Module):
    def __init__(self, args):
        super(StructGNN_Model, self).__init__()
        self.args = args
        self.device = 'cuda:0'
        self.smoothing = args.smoothing
        self.model = Struct2Seq(
            vocab=args.vocab_size,
            num_letters=args.vocab_size,
            node_features=args.hidden,
            edge_features=args.hidden, 
            hidden_dim=args.hidden,
            k_neighbors=args.k_neighbors,
            protein_features=args.features,
            dropout=args.dropout,
            use_mpnn=True)
        
        self.featurizer =  ProteinFeatures(
            args.hidden, args.hidden, top_k=args.k_neighbors,
            features_type=args.features,
            dropout=args.dropout
        )
    
    def _get_features(self, batch):
        X, lengths, mask = batch['X'], batch['lengths'], batch['mask']
        V, E, E_idx = self.featurizer(X, lengths, mask)
        batch.update({'V':V, 
                "E":E,
                "E_idx":E_idx, 
                "mask": mask})
        return batch
        
        
    def forward(self, batch):
        """ Graph-conditioned sequence model """
        S, V, E, E_idx, mask = batch['S'], batch['V'], batch['E'], batch['E_idx'], batch['mask']

        # Prepare node and edge embeddings
        h_V = self.model.W_v(V)
        h_E = self.model.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.model.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.model.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_ES_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, E_idx)

        # Decoder uses masked self-attention
        mask_attend = self.model._autoregressive_mask(E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        
        if self.model.forward_attention_decoder:
            mask_fw = mask_1D * (1. - mask_attend)
            h_ESV_encoder_fw = h_ESV_encoder
        else:
            h_ESV_encoder_fw = 0
        for layer in self.model.decoder_layers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV_dec = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV_dec + mask_fw*h_ESV_encoder_fw 
            h_V = layer(h_V, h_ESV, mask_V=mask) 

        logits = self.model.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return {'log_probs': log_probs}

    def sample(self, V, E, E_idx, mask, chain_mask=None, temperature=1.0):
        """ Autoregressive decoding of a model """
         # Prepare node and edge embeddings
        h_V = self.model.W_v(V)
        h_E = self.model.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.model.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        # Decoder alternates masked self-attention
        mask_attend = self.model._autoregressive_mask(E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        N_batch, N_nodes = V.size(0), V.size(1)
        h_S = torch.zeros_like(h_V)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device = self.device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.model.decoder_layers))]
        all_probs = []
        for t in range(N_nodes):
            # Hidden layers
            E_idx_t = E_idx[:,t:t+1,:]
            h_E_t = h_E[:,t:t+1,:,:]
            # use cache
            h_ES_enc_t = cat_neighbors_nodes(torch.zeros_like(h_S), h_E_t, E_idx_t)
            h_ESV_encoder_t = mask_fw[:,t:t+1,:,:] * cat_neighbors_nodes(h_V, h_ES_enc_t, E_idx_t)
            
            for l, layer in enumerate(self.model.decoder_layers):
                # Updated relational features for future states
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t) # [batch, 1, K, 384]
                h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_decoder_t + h_ESV_encoder_t # [batch, 1, K, 384]
                
                h_V_t = h_V_stack[l][:,t:t+1,:] # [batch, 1 128]
                h_V_stack[l+1][:,t,:] = layer(
                    h_V_t, h_ESV_t, mask_V=mask[:,t:t+1]
                ).squeeze(1) # [1, 128]

            # Sampling step
            h_V_t = h_V_stack[-1][:,t,:]
            logits = self.model.W_out(h_V_t) / temperature
            probs = F.softmax(logits, dim=-1)
            S_t = torch.multinomial(probs, 1).squeeze(-1)
            
            all_probs.append(probs)
            
            # Update
            h_S[:,t,:] = self.model.W_s(S_t)
            S[:,t] = S_t
        
        self.probs = torch.cat(all_probs, dim=0)
        return S