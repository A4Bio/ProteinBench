import torch.nn as nn
from ..modules.pifold_module import *
from torch_scatter import scatter_softmax, scatter_log_softmax

def positional_encoding(x):
    batch_size, seq_len, hidden_size = x.size()
    pos = torch.arange(0, seq_len).float().unsqueeze(1).repeat(1, hidden_size // 2)
    div = torch.exp(torch.arange(0, hidden_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_size))
    sin = torch.sin(pos * div)
    cos = torch.cos(pos * div)
    pos_encoding = torch.cat([sin, cos], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1)
    return pos_encoding


class MSAAttention(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.MSA_Q = nn.Linear(hidden_dim, hidden_dim)
        self.MSA_K = nn.Linear(hidden_dim, hidden_dim)
        self.MSA_V = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, inputs_embeds):
        pos_enc = positional_encoding(inputs_embeds)
        inputs_embeds = inputs_embeds + pos_enc
        
        query = self.MSA_Q(inputs_embeds)  # shape: [batch, N, 128]
        key = self.MSA_K(inputs_embeds)  # shape: [batch, N, 128]
        value = self.MSA_V(inputs_embeds)  # shape: [batch, N, 128]
        attn_scores = torch.bmm(query, key.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_scores, dim=2)
        
        attn_output = torch.bmm(attn_weights, value)
        return attn_output


class GNNTuning_Model(nn.Module):
    def __init__(self, args, num_encoder_layers, hidden_dim, input_design_dim, input_esm_dim, input_struct_dim=3072, input_esmif_dim=512, dropout=0.1):
        super(GNNTuning_Model, self).__init__()
        self.args = args
        encoder_layers = []
        for i in range(num_encoder_layers):
            encoder_layers.append(
                GeneralGNN(hidden_dim, hidden_dim*2, dropout=dropout, node_net = "AttMLP", edge_net = "EdgeMLP", node_context = 1, edge_context = 0),
            )
        self.encoder_layers = nn.Sequential(*encoder_layers)
        
        from transformers import AutoTokenizer
        from transformers.models.esm.modeling_esm import EsmModel, EsmEmbeddings
        from transformers.models.esm.configuration_esm import EsmConfig
        
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers")
        config = EsmConfig(attention_probs_dropout_prob=0,
                           hidden_size=hidden_dim,
                           intermediate_size=1280,
                           mask_token_id=32,
                           num_attention_heads=12,
                           num_hidden_layers=3,
                           pad_token_id=1,
                           position_embedding_type="rotary",
                           token_dropout=False,
                           vocab_size=33
                           )
        
        self.DesignEmbed = EsmEmbeddings(config)
        self.ESMEmbed = EsmEmbeddings(config)
        self.EdgeEmbed = nn.Sequential(nn.Linear(416+16+16, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim,hidden_dim))
        
        self.DesignConf = nn.Sequential(nn.Linear(1, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128,1),
                                     nn.Sigmoid())
        
        self.ESMConf = nn.Sequential(nn.Linear(1, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128,1))
        
        self.DesignProj = nn.Sequential(nn.Linear(input_design_dim, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim,hidden_dim))
        
        self.ESMProj = nn.Sequential(nn.Linear(input_esm_dim, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim,hidden_dim))
        
        self.StructProj = nn.Sequential(nn.Linear(input_struct_dim, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim,hidden_dim))
        
        self.ESMIFProj = nn.Sequential(nn.Linear(input_esmif_dim, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim,hidden_dim))
        
        self.ReadOut = nn.Linear(hidden_dim,33)
        # self.TimeEmbed = nn.Embedding(20, hidden_dim)
        # self.ProbEmbed = nn.Sequential(nn.Linear(33, 512),
        #                              nn.ReLU(),
        #                              nn.Linear(512, hidden_dim),
        #                              nn.ReLU(),
        #                              nn.Linear(hidden_dim,hidden_dim))
        
        self.MLP1 = nn.Sequential(nn.Linear(1, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim,1),
                                     nn.Sigmoid())
        
        self.MLP2 = nn.Sequential(nn.Linear(1, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim,1),
                                     nn.Sigmoid())

        
        
    
    # def embed_gnn(self, pretrain_gnn, mask_select_id, mask_select_feat):
    #     gnn_embed = self.DesignEmbed(mask_select_id(pretrain_gnn['pred_ids'])).squeeze()
    #     gnn_conf = self.DesignConf(mask_select_id(pretrain_gnn['confs']))
    #     gnn_proj = self.DesignProj(mask_select_feat(pretrain_gnn['embeds']))
        
    #     if self.args.use_confembed:
    #         return gnn_embed*F.sigmoid(gnn_conf)  + gnn_proj 
    #     else:
    #         return gnn_embed  + gnn_proj 
    
    # def embed_esm(self, pretrain_esm, mask_select_id, mask_select_feat):
    #     esm_embed = self.ESMEmbed(mask_select_id(pretrain_esm['pred_ids'])).squeeze()
    #     esm_conf = self.ESMConf(mask_select_id(pretrain_esm['confs']))
    #     esm_proj = self.ESMProj(mask_select_feat(pretrain_esm['embeds']))
    #     if self.args.use_confembed:
    #         return esm_embed*F.sigmoid(esm_conf) + esm_proj 
    #     else:
    #         return esm_embed + esm_proj
    
    # def embed_struct(self, pretrain_struct, mask_select_feat):
    #     struct_proj = self.StructProj(mask_select_feat(pretrain_struct['embeds']))
    #     return struct_proj
    
    # def embed_esmif(self, pretrain_esmif, mask_select_feat):
    #     struct_proj = self.ESMIFProj(mask_select_feat(pretrain_esmif['embeds']))
    #     return struct_proj
    
    def fuse(self, mask_select_feat, mask_select_id, gnn_embed=None, esm_embed=None, gearnet_embed=None, esmif_embed=None, gnn_pred_id=None, esm_pred_id=None, confidence=None, confidence_esm=None):
        gnn, esm, gearnet, esmif, conf = 0, 0, 0, 0, 1.0
        if gnn_embed is not None:
            gnn = self.DesignProj(mask_select_feat(gnn_embed))
            gnn += self.DesignEmbed(mask_select_id(gnn_pred_id)).squeeze()
        
        if esm_embed is not None:
            esm = self.ESMProj(mask_select_feat(esm_embed))
            esm += self.ESMEmbed(mask_select_id(esm_pred_id)).squeeze()
        
        if gearnet_embed is not None:
            gearnet = self.StructProj(mask_select_feat(gearnet_embed))
        
        if esmif_embed is not None:
            esmif = self.ESMIFProj(mask_select_feat(esmif_embed))
        
        if conf is not None:
            conf = self.DesignConf(mask_select_id(confidence))
            esm_conf = self.ESMConf(mask_select_id(confidence_esm))
        
        return (gnn*conf+esm*esm_conf+gearnet+esmif)
        

    def forward(self, batch):
        pretrain_design,  h_E_raw, E_idx, mask_attend, batch_id = batch['pretrain_design'], batch['h_E'], batch['E_idx'], batch['attention_mask'], batch['batch_id']
        
        if self.args.use_LM:
            pretrain_esm_msa = batch['pretrain_esm_msa']
        
        if self.args.use_gearnet:
            pretrain_struct = batch['pretrain_struct']
        
        if self.args.use_esmif:
            pretrain_esmif = batch['pretrain_esmif']
            
        mask_select_id = lambda x:  torch.masked_select(x, mask_attend.bool()).reshape(-1,1)
        mask_select_feat = lambda x:  torch.masked_select(x, mask_attend.bool().unsqueeze(-1)).reshape(-1,x.shape[-1])
        
        inputs_embeds = 0
        for i in range(self.args.msa_n):
            gnn_embed = pretrain_design['embeds']
            esm_embed = pretrain_esm_msa['embeds'][i] if self.args.use_LM else None
            gearnet_embed = pretrain_struct['embeds'][i] if self.args.use_gearnet else None
            esmif_embed = pretrain_esmif['embeds'] if self.args.use_esmif else None
            confidence = pretrain_design['confs']
            confidence_esm = pretrain_esm_msa['confs'][i]
            inputs_embeds += self.fuse(mask_select_feat, mask_select_id, gnn_embed, esm_embed, gearnet_embed, esmif_embed, pretrain_design['pred_ids'], pretrain_esm_msa['pred_ids'][i], confidence, confidence_esm)
        
        h_V = inputs_embeds
        h_E = self.EdgeEmbed(h_E_raw)
        
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, batch_id)
            
        logits = self.ReadOut(h_V)
        
        # confidence update
        old_confs = mask_select_id(pretrain_design['confs'])
        confs = torch.softmax(logits, dim=-1).max(dim=-1)[0][:,None]
        h_V = h_V*self.MLP1(confs-old_confs) + inputs_embeds*self.MLP2(old_confs-confs)
        logits = self.ReadOut(h_V)
        
        B, N = pretrain_design['confs'].shape
        vocab_size = logits.shape[-1]
        
        new_logits = torch.zeros(B,N,vocab_size, device=logits.device).reshape(B*N, vocab_size)
        new_logits = new_logits.masked_scatter_(mask_attend.bool().view(-1,1), logits)
        new_logits = new_logits.reshape(B,N,vocab_size)
        log_probs = torch.log_softmax(new_logits, dim=-1)
        
        device = logits.device
        seqs, confs, embeds, probs2 = self.to_matrix(h_V, logits, batch_id)
        
        
        ret = {"pred_ids":seqs['input_ids'].to(device),
               "confs":confs,
               "embeds":embeds,
               "probs":probs2,
               "attention_mask":seqs['attention_mask'].to(device),
               "h_E":h_E_raw,
               "E_idx":E_idx,
               "batch_id":batch_id,
               "log_probs":log_probs}
        return ret

    def to_matrix(self, h_V, logits, batch_id):
        
        probs = F.softmax(logits, dim=-1)
        conf, pred_id = probs.max(dim=-1)
        
        maxL = 0
        for b in batch_id.unique():
            mask = batch_id==b
            L = mask.sum()
            if L>maxL:
                maxL=L

        confs = []
        seqs = []
        embeds = []
        probs2 = []
        for b in batch_id.unique():
            mask = batch_id==b
            # elements = [alphabet[int(id)] for id in pred_id[mask]]
            elements = self.tokenizer.decode(pred_id[mask]).split(" ")
            seqs.append(elements)
            confs.append(conf[mask])
            embeds.append(h_V[mask])
            probs2.append(probs[mask])
        
        seqs = self.tokenizer(["".join(one) for one in seqs], padding=True, truncation=True, return_tensors='pt', add_special_tokens=False)
        confs = torch.stack([F.pad(one, (0, maxL-len(one))) for one in confs])
        embeds = torch.stack([F.pad(one, (0,0, 0, maxL-len(one))) for one in embeds])
        probs2 = torch.stack([F.pad(one, (0,0, 0, maxL-len(one)), value=1/33) for one in probs2])
        return seqs, confs, embeds, probs2
