# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch
from torch import nn
import torch.nn.functional as F

from ..modules.esmif_module import GVPTransformerEncoder, TransformerDecoder
from transformers import AutoTokenizer

class GVPTransformerModel(nn.Module):
    """
    GVP-Transformer inverse folding model.

    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """

    def __init__(self, args):
        super().__init__()
        # alphabet = Alphabet.from_architecture()
        alphabet = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers") 
        encoder_embed_tokens = self.build_embedding(
            args, alphabet, args.encoder_embed_dim,
        )
        decoder_embed_tokens = self.build_embedding(
            args, alphabet, args.decoder_embed_dim, 
        )
        encoder = self.build_encoder(args, alphabet, encoder_embed_tokens)
        decoder = self.build_decoder(args, alphabet, decoder_embed_tokens)
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = GVPTransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
        )
        return decoder

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad_token_id
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward(
        self,
        coords,
        padding_mask,
        confidence,
        prev_output_tokens,
        return_all_hiddens: bool = False,
        features_only: bool = False,
    ):
        encoder_out = self.encoder(coords, padding_mask, confidence,
            return_all_hiddens=return_all_hiddens)
            
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )

        return logits, extra

    def sample(self, batch_coords, padding_mask, partial_seq=None, temperature=1.0, confidence=None):
        """
        Samples sequences based on multinomial sampling (no beam search).

        Args:
            coords: L x 3 x 3 list representing one backbone
            partial_seq: Optional, partial sequence with mask tokens if part of
                the sequence is known
            temperature: sampling temperature, use low temperature for higher
                sequence recovery and high temperature for higher diversity
            confidence: optional length L list of confidence scores for coordinates
        """
        L = batch_coords.shape[1]
        
        sampled_tokens = torch.zeros((1, 1+L), device=batch_coords.device).long()
            
        # Save incremental states for faster sampling
        incremental_state = dict()
        
        # Run encoder only once
        t1 = time.time()
        encoder_out = self.encoder(batch_coords, padding_mask, confidence)
        t2 = time.time()

        # Decode one token at a time
        for i in range(1, L+1):
            logits, _ = self.decoder(
                sampled_tokens[:, :i], 
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            # sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
            sampled_tokens[:, i] = probs.argmax(dim=-1)
        sampled_seq = sampled_tokens[0, 1:]
        t3 = time.time()

        # Convert back to string via lookup
        return sampled_seq