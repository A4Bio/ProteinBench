import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from omegaconf import OmegaConf

class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = F.gelu

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

class GaussianLayer(nn.Module):
    def __init__(self, num_distance=25, K=16, edge_dim=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, num_distance*self.K)  # 16 * 25 = 400, it's the total number of kernels
        self.stds = nn.Embedding(1, num_distance*self.K)
        self.mul = nn.Linear(edge_dim, num_distance)
        self.bias = nn.Linear(edge_dim, num_distance)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_feat):
        mul = self.mul(edge_feat).type_as(x)
        bias = self.bias(edge_feat).type_as(x)

        # x = mul * x.unsqueeze(-1) + bias # [B, N, N, 25, 1]
        x = mul * x + bias # [B, N, N, 25]
        x = x.unsqueeze(-1) # [B, N, N, 25, 1]
        x = x.expand(-1, -1, -1, -1, self.K) # [B, N, N, 25, K]
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1) # [B, N, N, 25*K]
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class GaussianEncoder(nn.Module):
    def __init__(self, kernel_num, feat_dim, num_head, use_dist=1, use_product=1):
        super().__init__()
        self.num_distance = 0
        self.use_dist = use_dist
        self.use_product = use_product
        if use_dist:
            self.num_distance += 1

        if use_product:
            self.num_distance += 1

        self.gbf = GaussianLayer(self.num_distance, kernel_num, feat_dim)
        self.node_gate = nn.Linear(feat_dim, 1)

        self.gbf_proj = NonLinearHead(
            input_dim=kernel_num*self.num_distance,
            out_dim=num_head,
            hidden=128,
        )
        self.centrality_proj = NonLinearHead(
            input_dim=kernel_num*self.num_distance,
            out_dim=feat_dim,
            hidden=1024,
        )
    
    def get_encoding_features(self, dist, et, pair_mask=None, get_bias=True):
        n_node = dist.size(-2)
        gbf_feature = self.gbf(dist, et)

        if pair_mask is not None:
            centrality_encoding = gbf_feature * pair_mask.unsqueeze(-1)
        else:
            centrality_encoding = gbf_feature # [B, N, N, 25*K]
        centrality_encoding = self.centrality_proj(centrality_encoding.sum(dim=-2)) # [B, N, encoder_embed_dim]

        graph_attn_bias = self.gbf_proj(gbf_feature) # [B, N, N, num_head]
        if get_bias:
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous() # [B, num_head, N, N]
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node) # [B*num_head, N, N]
        return graph_attn_bias, centrality_encoding

    
    def build_pairwise_product_dist(self, coords, node_feat):
        dist = _get_dist(coords,coords)
        coords = coords * self.node_gate(node_feat)
        pretext = coords[:,:,None]+coords[:,None,:]
        A = torch.einsum('bijd,bjd->bij', pretext, coords)
        B = torch.einsum('bid,bjd->bij', coords, coords)
        product = A*B
        product, dist = product[...,None], dist[...,None]
        geo_feat = torch.empty_like(product)[...,0:0]
        if self.use_dist:
            geo_feat = torch.cat([geo_feat, dist], dim=-1)
        
        if self.use_product:
            geo_feat = torch.cat([geo_feat, product], dim=-1)
        return geo_feat
    
    def forward(self, coords, node_feat, pair_mask=None, get_bias=True):
        geo_feat = self.build_pairwise_product_dist(coords, node_feat)
        edge_feat = node_feat[:,:,None,:]-node_feat[:,None,:,:]
        graph_attn_bias, centrality_encoding = self.get_encoding_features(geo_feat, edge_feat, pair_mask=pair_mask, get_bias=get_bias)
        x = centrality_encoding
        return x, graph_attn_bias

class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tensor:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            attn = F.dropout(F.softmax(attn_weights, dim=-1), p=self.dropout, training=self.training)
        else:
            attn_weights += attn_bias
            attn = F.dropout(F.softmax(attn_weights, dim=-1), p=self.dropout, training=self.training)
        # pdb.set_trace()
        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn

class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
        # edge_attn_hidden_dim = 8,
        # edge_attn_heads = 4,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = F.gelu
        # self.edge_attn_hidden_dim = edge_attn_hidden_dim
        # self.edge_attn_heads = edge_attn_heads

        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            dropout=attention_dropout,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.post_ln = post_ln


    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        return_attn: bool=False,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        # new added
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
            # edge_repr = attn_weights
            # edge_repr[edge_repr == float("-inf")] = 0
            # edge_repr = edge_repr.view(x.shape[0], -1, x.shape[1], x.shape[1]).permute(0, 2, 3, 1).contiguous()
            # edge_repr_update = self.edge_attn(edge_repr, pair_mask)
            # edge_repr_update = edge_repr_update.permute(0, 3, 1, 2).contiguous()
            # edge_repr_update = edge_repr_update.view(-1, x.shape[1], x.shape[1]) # [bsz*num_heads, tgt_len, src_len]
            # attn_weights = attn_weights + edge_repr_update # residual connection and keep padding mask

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs

class TransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = nn.LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        if not no_final_head_layer_norm:
            self.final_head_layer_norm = nn.LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        bsz = emb.size(0)
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert attn_mask is not None
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)

        for i in range(len(self.layers)):
            x, attn_mask, _ = self.layers[i](
                x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
            )

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()

        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(
            pair_mask, delta_pair_repr_norm, dim=(-1, -2)
        )

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm

def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

def _get_dist(A, B):
    D_A_B = torch.sqrt(torch.sum((A[..., None,:] - B[...,None,:,:])**2,-1) + 1e-6) #[B, L, L]
    return D_A_B



