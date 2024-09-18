import torch
import torch.nn as nn
from .Tuning import GNNTuning_Model


class MemoTuning(nn.Module):
    def __init__(self, args, tunning_layers_n, tunning_layers_dim, input_design_dim, input_esm_dim, tunning_dropout, tokenizer, fix_memory=False):
        super().__init__()
        self.args = args
        self.tunning_layers_dim = tunning_layers_dim
        self.GNNTuning = GNNTuning_Model(args, num_encoder_layers=tunning_layers_n, hidden_dim=tunning_layers_dim, input_design_dim=input_design_dim, input_esm_dim=input_esm_dim, dropout = tunning_dropout)
        self.tokenizer = tokenizer
        self.memory = {}
        
    def save_param_memory(self, path):
        torch.save({"params":self.state_dict(),"memory": self.memory}, path)

    def load_param_memory(self, path):
        data = torch.load(path)
        self.load_state_dict(data['params'])
        self.memory = data['memory']
    
    def get_seqs(self, pred_ids_raw, attention_mask):
        query_seqs = []
        for pred_ids, mask in zip(pred_ids_raw, attention_mask):
            seq = self.tokenizer.decode(pred_ids[mask], clean_up_tokenization_spaces=False)
            seq = "".join(seq.split(" "))
            query_seqs.append(seq)
        return query_seqs
    
    def initoutput(self, pretrain_design, B, max_L, device):
        # initialize output
        self.out_pred_ids = torch.zeros_like(pretrain_design['pred_ids'])
        self.out_confs = torch.zeros_like(pretrain_design['confs'])
        self.out_embeds = torch.zeros(B, max_L, self.tunning_layers_dim, device = device)
        self.out_attention_mask = torch.zeros_like(pretrain_design['attention_mask'])
        self.out_probs = torch.zeros_like(pretrain_design['probs'])
        self.out_log_probs = torch.zeros_like(pretrain_design['probs'])
        self.titles = [None for i in range(B)]
        
    
    
    def retrivel(self, keys, num_nodes,device, use_memory):
        unseen = []
        for idx in range(len(keys)):
            key = keys[idx]
            if (key in self.memory) and use_memory:
                self.out_pred_ids[idx, :num_nodes[idx]] = self.memory[key]['pred_ids'].to(device)
                self.out_confs[idx, :num_nodes[idx]] = self.memory[key]['confs'].to(device)
                self.out_embeds[idx, :num_nodes[idx]] = self.memory[key]['embeds'].to(device)
                self.out_attention_mask[idx, :num_nodes[idx]] = self.memory[key]['attention_mask'].to(device)
                self.out_probs[idx, :num_nodes[idx]] = self.memory[key]['probs'].to(device)
                self.out_log_probs[idx, :num_nodes[idx]] = self.memory[key]['log_probs'].to(device)
                self.titles[idx] = key
            else:
                unseen.append(idx)
        return unseen
    
    def rebatch(self,unseen, batch_id_raw, E_idx_raw, h_E_raw, shift, num_nodes, pretrain_design, pretrain_esm_msa, pretrain_struct, pretrain_esmif, device):
        unseen_design_pred_ids = []
        unseen_design_confs = []
        unseen_design_embeds = []
        unseen_design_attention_mask = []
        
        unseen_esm_pred_ids = []
        unseen_esm_confs = []
        unseen_esm_embeds = []
        unseen_esm_attention_mask = []
        unseen_struct_embeds = []
        unseen_esmif_embeds = []
        h_E = []
        E_idx = []
        batch_id = []
        
        new_shift = 0
        for bid, i in enumerate(unseen):
            edge_mask = batch_id_raw[E_idx_raw[0]] == i
            h_E.append(h_E_raw[edge_mask])
            E_idx.append(E_idx_raw[:,edge_mask]-shift[i]+new_shift)
            batch_id.append(torch.ones(num_nodes[i], device=device).long()*bid)
            new_shift += num_nodes[i]
            
            unseen_design_pred_ids.append(pretrain_design['pred_ids'][i])
            unseen_design_confs.append(pretrain_design['confs'][i])
            unseen_design_embeds.append(pretrain_design['embeds'][i])
            unseen_design_attention_mask.append(pretrain_design['attention_mask'][i])
            
            if self.args.use_LM:
                unseen_esm_pred_ids.append(pretrain_esm_msa['pred_ids'][:,i])
                unseen_esm_confs.append(pretrain_esm_msa['confs'][:,i])
                unseen_esm_embeds.append(pretrain_esm_msa['embeds'][:,i])
                unseen_esm_attention_mask.append(pretrain_esm_msa['attention_mask'][:,i])
            
            if self.args.use_gearnet:
                unseen_struct_embeds.append(pretrain_struct['embeds'][:,i])
            
            if self.args.use_esmif:
                unseen_esmif_embeds.append(pretrain_esmif['embeds'][i])
            
            
        unseen_design_pred_ids = torch.stack(unseen_design_pred_ids)
        unseen_design_confs = torch.stack(unseen_design_confs)
        unseen_design_embeds = torch.stack(unseen_design_embeds)
        unseen_design_attention_mask = torch.stack(unseen_design_attention_mask)
        
        if self.args.use_LM:
            unseen_esm_pred_ids = torch.stack(unseen_esm_pred_ids, dim=1)
            unseen_esm_confs = torch.stack(unseen_esm_confs, dim=1)
            unseen_esm_embeds = torch.stack(unseen_esm_embeds, dim=1)
            unseen_esm_attention_mask = torch.stack(unseen_esm_attention_mask, dim=1)
        
        if self.args.use_gearnet:
            unseen_struct_embeds = torch.stack(unseen_struct_embeds, dim=1)
        
        if self.args.use_esmif:
            unseen_esmif_embeds = torch.stack(unseen_esmif_embeds, dim=0)
            
        
        unseen_batch = {"pretrain_design":
                            {"pred_ids": unseen_design_pred_ids, 
                            "confs":unseen_design_confs, 
                            "embeds": unseen_design_embeds, 
                            "attention_mask":unseen_design_attention_mask},
                        "h_E": torch.cat(h_E),
                        "E_idx": torch.cat(E_idx, dim=1),
                        "batch_id": torch.cat(batch_id),
                        "attention_mask":unseen_design_attention_mask
                        }

        if self.args.use_LM:
            unseen_batch["pretrain_esm_msa"]={"pred_ids": unseen_esm_pred_ids, 
                            "confs":unseen_esm_confs, 
                            "embeds": unseen_esm_embeds, 
                            "attention_mask":unseen_esm_attention_mask}
        
        if self.args.use_gearnet:
            unseen_batch["pretrain_struct"] = {
                            "embeds":unseen_struct_embeds}
        
        if self.args.use_esmif:
            unseen_batch["pretrain_esmif"] = {"embeds":unseen_esmif_embeds}
        return unseen_batch
    
    def save2memory(self,keys,unseen,num_nodes, unseen_results):
        # save to memory
        for i in range(len(unseen)):
            key = keys[unseen[i]]
            num = num_nodes[unseen[i]]
            self.memory[key] = {"pred_ids":unseen_results['pred_ids'][i][:num].detach().to('cpu'), 
                                "confs":unseen_results['confs'][i][:num].detach().to('cpu'), 
                                "embeds":unseen_results['embeds'][i][:num].detach().to('cpu'),
                                "probs":unseen_results['probs'][i][:num].detach().to('cpu'),
                                "log_probs":unseen_results['log_probs'][i][:num].detach().to('cpu'),
                                "attention_mask":unseen_results['attention_mask'][i][:num].detach().to('cpu')}
    
    def update(self, unseen, num_nodes, unseen_results, keys):
        # update
        for i in range(len(unseen)):
            num = num_nodes[unseen[i]]
            self.out_pred_ids[unseen[i], :num] = unseen_results['pred_ids'][i][:num]
            self.out_confs[unseen[i], :num] = unseen_results['confs'][i][:num]
            self.out_embeds[unseen[i], :num] = unseen_results['embeds'][i][:num]
            self.out_probs[unseen[i], :num] = unseen_results['probs'][i][:num]
            self.out_log_probs[unseen[i], :num] = unseen_results['log_probs'][i][:num]
            self.titles[unseen[i]] = keys[unseen[i]]

    def forward(self, batch, use_memory=False):
        self.use_memory = use_memory
        pretrain_design,  h_E_raw, E_idx_raw, mask_attend, batch_id_raw = batch['pretrain_design'] ,batch['h_E'], batch['E_idx'], batch['attention_mask'], batch['batch_id']
        device = h_E_raw.device
        
        pretrain_esm_msa = None
        if self.args.use_LM:
            pretrain_esm_msa = batch['pretrain_esm_msa']
        
        pretrain_struct = None
        if self.args.use_gearnet:
            pretrain_struct = batch['pretrain_struct']
        
        pretrain_esmif = None
        if self.args.use_esmif:
            pretrain_esmif = batch['esm_feat']
        
        
        num_nodes = batch['attention_mask'].sum(dim=-1)
        shift = torch.cat([torch.zeros(1, device=device), torch.cumsum(num_nodes, dim=0)]).long()
        
        B, max_L = num_nodes.shape[0], num_nodes.max()
        
        self.initoutput(pretrain_design, B, max_L, device)
        
        
        # keys = list(zip(design_seqs, *lm_seqs))
        keys = batch['title']
        unseen = self.retrivel(keys, num_nodes,device, use_memory)
        
                
        if len(unseen)>0:
            unseen_batch = self.rebatch(unseen, batch_id_raw, E_idx_raw, h_E_raw, shift, num_nodes, pretrain_design, pretrain_esm_msa, pretrain_struct, pretrain_esmif, device)
            unseen_results = self.GNNTuning(unseen_batch)
            
            self.save2memory(keys,unseen,num_nodes, unseen_results)
            self.update(unseen, num_nodes, unseen_results, keys)
            
        return {'title':self.titles,'pred_ids':self.out_pred_ids, 'confs':self.out_confs, 'embeds':self.out_embeds, 'probs':self.out_probs, "log_probs":self.out_log_probs, 'attention_mask':pretrain_design['attention_mask']}
        
        
        
        



    