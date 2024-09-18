import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphtrans_module import Normalize, PositionWiseFeedForward, NeighborAttention


class Local_Module(nn.Module):
    def __init__(self, num_hidden, num_in, is_attention, dropout=0.1, scale=30):
        super(Local_Module, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.is_attention = is_attention
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
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

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        '''
        h_V: [batch, num_nodes, 128]
        h_E: [batch, num_nodes, K, 128]
        mask_V: [batch, num_nodes]
        mask_attend: [batch, num_nodes, K]
        '''
        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1) 
        # get message
        h_message = self.W(h_EV) # [4, 312, 30, 384]-->[4, 312, 30, 128]
        # Attention
        if self.is_attention == 0:
            e = F.sigmoid(F.leaky_relu(torch.matmul(h_EV, self.A))).squeeze(-1).exp() # [4, 312, 30, 384]-->[4, 312, 30]
            e = e / e.sum(-1).unsqueeze(-1) # [4, 312, 30]
            h_message = h_message * e.unsqueeze(-1) # [4, 312, 30, 128]

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        
        # message aggragation
        dh = torch.sum(h_message, -2) / self.scale # [4, 312, 128]

        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class Global_Module(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0.1):
        super(Global_Module, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])
        self.attention = NeighborAttention(num_hidden, num_in, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        dh = self.attention(h_V, h_E, mask_attend)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V