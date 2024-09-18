import torch
import os.path as osp
from ..models.pifold_model import PiFold_Model
import torch.nn.functional as F


class PretrainPiFold_Model(PiFold_Model):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        PiFold_Model.__init__(self, args)
        if args.augment_eps>0:
            pretrain_pifold_path = osp.join(self.args.res_dir, self.args.data_name, f"PiFold_{args.augment_eps}", "checkpoint.pth")
        else:
            pretrain_pifold_path = osp.join(self.args.res_dir, self.args.data_name, "PiFold", "checkpoint.pth")
        self.load_state_dict(torch.load(pretrain_pifold_path))
    
    @torch.no_grad()
    def forward(self, batch):
        h_V, h_P, P_idx, batch_id = batch['h_V'], batch['h_E'], batch['E_idx'], batch['batch_id']
        device = h_V.device
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))
        
        h_V, h_P = self.encoder(h_V, h_P, P_idx, batch_id)
        log_probs, logits = self.decoder(h_V, batch_id)
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
        
        ret = {"pred_ids":seqs['input_ids'].to(device),
               "confs":confs,
               "embeds":embeds,
               "probs":probs2,
               "attention_mask":seqs['attention_mask'].to(device),
               "E_idx":P_idx,
               "batch_id":batch_id,
               "h_E":h_P}
        return ret

