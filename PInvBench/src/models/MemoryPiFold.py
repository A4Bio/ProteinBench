import torch
import torch.nn as nn
from .PretrainPiFold_model import PretrainPiFold_Model
from torch_scatter import scatter_sum
import torch.nn.functional as F

class MemoPiFold_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.PretrainPiFold = PretrainPiFold_Model(args)
        self.memory = {}
    
    def save_memory(self, path):
        params = {key:val for key,val in self.state_dict().items() if "GNNTuning" in key}
        torch.save({"params":params,"memory": self.memory}, path)

    def load_memory(self, path):
        data = torch.load(path)
        self.load_state_dict(data['params'], strict=False)
        self.memory = data['memory']
        
    def initoutput(self, B, max_L, nums, device):
        self.confs = torch.ones(B, max_L, device=device)
        self.embeds = torch.ones(B, max_L, 128, device=device)
        self.probs = torch.ones(B, max_L, 33, device=device)
        self.attention_mask = torch.ones_like(self.confs)==0
        self.titles = [None for i in range(B)]
        for id, num in enumerate(nums):
            self.attention_mask[id, :num] = True
        self.edge_feats = []
        
    def retrivel(self, batch, nums, batch_uid, device, use_memory):
        # retrieval
        unseen = []
        
        for idx, name in enumerate(batch['title']):
            if (name in self.memory) and use_memory:
                try:
                    self.confs[batch_uid[idx],:nums[idx]] = self.memory[name]['conf'].to(device)
                except:
                    self.confs[batch_uid[idx],:nums[idx]] = self.memory[name]['conf'].to(device)
                self.embeds[batch_uid[idx],:nums[idx]] = self.memory[name]['embed'].to(device)
                self.probs[batch_uid[idx],:nums[idx]] = self.memory[name]['prob'].to(device)
                self.edge_feats.append((batch_uid[idx], self.memory[name]['h_E'].to(device)))
                self.titles[batch_uid[idx]] = name
            else:
                unseen.append(idx)
        return unseen

    def rebatch(self, unseen, batch_uid, batch_id, batch, shift, nums, device):
        h_V2, h_E2, E_idx2, batch_id2 = [], [], [], []
        shift2 = [0]
        idx=0
        for id in batch_uid:
            if id not in unseen:
                continue
            node_mask = batch_id == id
            edge_mask = batch_id[batch['E_idx'][0]] == id
            h_V2.append(batch['h_V'][node_mask])
            h_E2.append(batch['h_E'][edge_mask])
            new_E_idx = batch['E_idx'][:,edge_mask] 
            new_E_idx = new_E_idx- shift[batch_id[new_E_idx[0]]]+shift2[-1]
            E_idx2.append(new_E_idx)
            new_batch_id = torch.ones(node_mask.sum().long(), device=device)*idx
            batch_id2.append(new_batch_id)
            shift2.append(shift2[-1]+nums[id])
            idx+=1
        
        h_V2 = torch.cat(h_V2)
        h_E2 = torch.cat(h_E2)
        E_idx2 = torch.cat(E_idx2, dim=-1)
        batch_id2 = torch.cat(batch_id2).long()
        return {"h_V":h_V2, 'h_E':h_E2, 'E_idx':E_idx2, 'batch_id':batch_id2}
    
    def update_save2memory(self, unseen, batch_id2, E_idx2, batch, pretrain_gnn, max_L):
        for id in batch_id2.unique():
            node_mask = batch_id2 == id
            edge_mask = batch_id2[E_idx2[0]] == id
            title = batch['title'][unseen[int(id)]]
            conf = pretrain_gnn['confs'][id]
            conf = F.pad(conf, (0, max_L-len(conf)))
            embed = pretrain_gnn['embeds'][id]
            embed = F.pad(embed, (0,0,0,max_L-len(embed)))
            prob = pretrain_gnn['probs'][id]
            prob = F.pad(prob, (0,0,0,max_L-len(prob)))
            self.edge_feats.append((unseen[int(id)], pretrain_gnn['h_E'][edge_mask]))
            
            self.confs[unseen[int(id)]] = conf
            self.embeds[unseen[int(id)]] = embed
            self.probs[unseen[int(id)]] = prob
            self.titles[unseen[int(id)]] = title
            
            attn_mask = self.attention_mask[unseen[int(id)]]
            
            # save to memory
            self.memory[title] = {'conf': conf[attn_mask].detach().to('cpu'), 
                                'embed': embed[attn_mask].detach().to('cpu'), 
                                'prob': prob[attn_mask].detach().to('cpu'),
                                'h_E':pretrain_gnn['h_E'][edge_mask].detach().to('cpu')}
    
    @torch.no_grad()
    def forward(self, batch, use_memory=False):
        batch_id = batch['batch_id']
        batch_uid = batch_id.unique()
        device = batch_id.device
        
        nums = scatter_sum(torch.ones_like(batch_id), batch_id)
        shift = torch.cat([torch.zeros(1, device=device), torch.cumsum(nums, dim=0)]).long()
        max_L, B = nums.max(), batch_uid.shape[0]
        
        self.initoutput(B, max_L, nums, device)
        unseen = self.retrivel(batch, nums, batch_uid, device, use_memory)
        
        # organize data
        if len(unseen)>0:
            # rebatch
            new_batch = self.rebatch(unseen, batch_uid, batch_id, batch, shift, nums, device)
            
            # forward pass 
            pretrain_gnn = self.PretrainPiFold(new_batch)
            
            self.update_save2memory(unseen, pretrain_gnn['batch_id'], pretrain_gnn['E_idx'], batch, pretrain_gnn, max_L)
            
        
        self.edge_feats = sorted(self.edge_feats, key=lambda x: x[0])
        self.edge_feats = torch.cat([one[1] for one in self.edge_feats])
        
        pred_ids = self.probs.argmax(dim=-1)*self.attention_mask + (~self.attention_mask)*1
        
        return {'title': self.titles,
                'pred_ids': pred_ids,
                'confs': self.confs, 
                'embeds': self.embeds, 
                'probs': self.probs, 
                'attention_mask': self.attention_mask,
                'h_E':self.edge_feats,
                'E_idx': batch['E_idx'],
                'batch_id': batch['batch_id']}
            
            
    
    def _get_features(self, S, score, X, mask, chain_mask, chain_encoding):
        return self.PretrainPiFold._get_features(S, score, X, mask, chain_mask, chain_encoding)