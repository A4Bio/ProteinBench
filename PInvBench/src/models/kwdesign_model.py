import time
import torch
import torch.nn as nn
from .MemoryTuning import MemoTuning
import copy
from .MemoryESM import MemoESM
from .MemoryPiFold import MemoPiFold_model
from .MemoryESMIF import MemoESMIF
import torch
from torch_scatter import scatter_sum
from ..tools import gather_nodes, _dihedrals, _get_rbf, _get_dist, _rbf, _orientations_coarse_gl_tuple
import numpy as np
def beam_search(post, k):
    """Beam Search Decoder

    Parameters:

        post(Tensor) – the posterior of network.
        k(int) – beam size of decoder.

    Outputs:

        indices(Tensor) – a beam of index sequence.
        log_prob(Tensor) – a beam of log likelihood of sequence.

    Shape:

        post: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).

    Examples:

        >>> post = torch.softmax(torch.randn([32, 20, 1000]), -1)
        >>> indices, log_prob = beam_search_decoder(post, 3)

    """

    batch_size, seq_length, token_size = post.shape
    log_post = post.log()
    log_prob, indices = log_post[:, 0, :].topk(k, sorted=True)
    indices = indices.unsqueeze(-1)
    for i in range(1, seq_length):
        log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, k, 1) # [batch, k, 33]
        log_prob, index = log_prob.view(batch_size, -1).topk(k, sorted=True)
        index = index%token_size
        indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)
    return indices, log_prob



class Design_Model(nn.Module):
    def __init__(self, args, temporature, msa_n, tunning_layers_n, tunning_layers_dim, input_design_dim, input_esm_dim, tunning_dropout, design_model, LM_model, ESMIF_model, param_path=None):
        super(Design_Model, self).__init__()
        self.args = args
        self.temporature = temporature
        self.msa_n = msa_n
        self.design_model = design_model
        self.LM_model = LM_model
        self.ESMIF_model = ESMIF_model
        # self.GNNTuning = GNNTuning_Model(num_encoder_layers=tunning_layers_n, hidden_dim=tunning_layers_dim, input_design_dim=input_design_dim, input_esm_dim=input_esm_dim, dropout = tunning_dropout)
        self.GNNTuning = MemoTuning(args, tunning_layers_n, tunning_layers_dim, input_design_dim, input_esm_dim, tunning_dropout, tokenizer=self.LM_model.tokenizer)
        # self.Predictor = nn.Linear(tunning_layers_dim, 21)
        self.conf_max = 0
        self.patience = 0
        self.best_params = None
        self.confidence = []
        if param_path is not None:
            self.GNNTuning.load_state_dict(torch.load(param_path))
        
    def get_MSA(self, pretrain_design):
        pretrain_gnn_msa = pretrain_design
        B, N = pretrain_gnn_msa['confs'].shape
        probs, pred_ids, confs, attention_mask, titles = [], [], [], [], []
        
        for m in range(self.msa_n):
            titles.append(pretrain_design['title'])
            probs.append(torch.softmax(pretrain_gnn_msa['probs']/self.temporature, dim=-1))
            # pred_ids.append( msa_pred_ids[:,m,:])
            pred_ids.append(torch.multinomial(probs[-1].reshape(-1,33), 1).reshape(B,N))
        
            confs.append(probs[-1].reshape(-1,33)[torch.arange(pred_ids[-1].reshape(-1).shape[0]),pred_ids[-1].reshape(-1)].reshape(B,N))
            # confs.append(probs[-1].reshape(-1,33)[torch.arange(pred_ids[-1].reshape(-1).shape[0]).cuda,pred_ids[-1].reshape(-1)].reshape(B,N))
            attention_mask.append(pretrain_gnn_msa['attention_mask'])
        
        pretrain_esm_msa = {}
        pretrain_esm_msa['title'] = sum(titles,[])
        pretrain_esm_msa['probs'] = torch.cat(probs, dim=0)
        pretrain_esm_msa['pred_ids'] = torch.cat(pred_ids, dim=0)
        pretrain_esm_msa['confs'] = torch.cat(confs, dim=0)
        pretrain_esm_msa['attention_mask'] = torch.cat(attention_mask, dim=0)
        return pretrain_esm_msa

    def forward(self,  batch, design_memory=True, LM_memory=True, Struct_memory=True, Tuning_memory=True, ESMIF_memory=True):
        '''
        MemoPiFold: batch_id,titile, E_idx, h_V, h_E
        MemoESM: pred_ids, attention_mask, confs
        Tunning: pretrain_design
                    - pred_ids, confs, embeds
                 pretrain_esm_msa
                    - pred_ids, confs, embeds
                 h_E, E_idx, batch_id
        '''
        with torch.no_grad():
            pretrain_design = self.design_model(batch, design_memory)
            
            
            if self.args.use_LM:
                # language model forward
                pretrain_msa = self.get_MSA(pretrain_design)
                pretrain_esm_msa = self.LM_model(pretrain_msa, LM_memory)
                
                B, N = pretrain_design['confs'].shape
                pretrain_esm_msa['embeds'] = pretrain_esm_msa['embeds'].reshape(self.msa_n, B, N, -1)
                pretrain_esm_msa['pred_ids'] = pretrain_esm_msa['pred_ids'].reshape(self.msa_n, B, N)
                pretrain_esm_msa['confs'] = pretrain_esm_msa['confs'].reshape(self.msa_n, B, N)
                pretrain_esm_msa['attention_mask'] = pretrain_esm_msa['attention_mask'].reshape(self.msa_n, B, N)

            if self.args.use_gearnet:
                # structure model forward
                pretrain_msa = self.get_MSA(pretrain_design)
                protein_seqs_msa = self.LM_model.tokenizer.decode(pretrain_msa['pred_ids'][pretrain_msa['attention_mask']], clean_up_tokenization_spaces=False).split(" ")
                protein_coords_msa = batch['position'][:,1,:].repeat((self.msa_n,1))
                num_nodes = pretrain_msa['attention_mask'].sum(dim=1)
                msa_id = torch.arange(self.msa_n, device=num_nodes.device).repeat_interleave(pretrain_design['attention_mask'].shape[0])
                pretrain_struct_msa = self.Struct_model(protein_seqs_msa, protein_coords_msa, num_nodes, msa_id, pretrain_msa['title'], Struct_memory)
            
            if self.args.use_esmif:
                # esmif model forward
                esm_feat = self.ESMIF_model(batch, ESMIF_memory)
            
        
        new_batch = {}
        new_batch['title'] = pretrain_design['title']
        new_batch['pretrain_design'] = pretrain_design
        new_batch['h_E'] = batch['h_E']
        new_batch['E_idx'] = batch['E_idx']
        new_batch['batch_id'] = batch['batch_id']
        new_batch['attention_mask'] = pretrain_design['attention_mask']
        
        if self.args.use_LM:
            new_batch['pretrain_esm_msa'] = pretrain_esm_msa
        if self.args.use_gearnet:
            new_batch['pretrain_struct'] = pretrain_struct_msa
        if self.args.use_esmif:
            new_batch['esm_feat'] = esm_feat
        
        results = self.GNNTuning(new_batch, Tuning_memory)
        
        avg_confs = (results['attention_mask']*results['confs']).sum(dim=1)/results['attention_mask'].sum(dim=1)
        self.confidence.append(avg_confs)
        return results
    
    
class KWDesign_model(nn.Module):
    def __init__(self, args):
        super(KWDesign_model, self).__init__()
        self.args = args
        input_design_dim, input_esm_dim = args.input_design_dim, args.input_esm_dim
        tunning_layers_dim = args.tunning_layers_dim
        
        self.memo_pifold = MemoPiFold_model(args)
        self.memo_esmif = MemoESMIF(args)
        if args.load_memory:
            memory = torch.load(args.memory_path)
            self.memo_pifold = memory['memo_pifold']
            self.memo_esmif = memory['memo_esmif']

        for i in range(1, self.args.recycle_n+1):
            if i==1:
                self.register_module(f"Design{i}", 
                                     Design_Model(args, args.temporature, args.msa_n, args.tunning_layers_n, args.tunning_layers_dim, input_design_dim, input_esm_dim, args.tunning_dropout, self.memo_pifold, MemoESM(args), self.memo_esmif))
            else:
                self.register_module(f"Design{i}", 
                                     Design_Model(args, args.temporature, args.msa_n, args.tunning_layers_n, args.tunning_layers_dim, tunning_layers_dim, input_esm_dim, args.tunning_dropout, self.get_submodule(f"Design{i-1}"), MemoESM(args),  self.memo_esmif))
    
    def update(self, batch, node_nums, conf, results, log_probs_mat, threshold, current_batch_id):
        fix_mask = conf>threshold
        log_probs_mat[current_batch_id[fix_mask]] = results['log_probs'][fix_mask]
        current_batch_id = current_batch_id[conf<=threshold]
        
        batch_id_old = batch['batch_id']
        batch_id_old2new = torch.zeros_like(batch_id_old)-1
        batch_id_old2new[current_batch_id] = torch.arange(current_batch_id.shape[0], device=conf.device)
        
        
        node_mask = (batch_id_old.view(-1,1) == current_batch_id).any(dim=1)
        edge_mask = node_mask[batch['E_idx'][0]]
        shift_old = torch.cat([torch.zeros(1, device=node_nums.device),node_nums.cumsum(dim=0)]).long()
        shift_new = torch.cat([torch.zeros(1, device=node_nums.device),node_nums[current_batch_id].cumsum(dim=0)]).long()
        
        
        edge_batch_id = batch_id_old[batch['E_idx'][0]]
        E_idx = (batch['E_idx'] - shift_old[edge_batch_id] + shift_new[batch_id_old2new[edge_batch_id]])[:,edge_mask]
        
        new_batch = {"title": [batch['title'][int(idx)] for idx in current_batch_id],
                     "h_V": batch['h_V'][node_mask],
                     "h_E": batch['h_E'][edge_mask],
                     "E_idx": E_idx,
                     "batch_id": batch_id_old2new[batch_id_old[node_mask]],
                     "alphabet": batch["alphabet"],
                     "S": batch["S"],
                     "position": batch["position"]}
        return new_batch, log_probs_mat, current_batch_id
    
    
    def forward(self, batch):
        mask_select_feat = lambda x, mask_attend:  torch.masked_select(x, mask_attend.bool().unsqueeze(-1)).reshape(-1,x.shape[-1])
        
        log_probs_list, confs_list = [], []
        for i in range(1, self.args.recycle_n+1):
            module = self.get_submodule(f"Design{i}")
            if i< self.args.recycle_n:
                results = module(batch, Tuning_memory=True)
            else:
                results = module(batch, Tuning_memory=False)
            
            log_probs = mask_select_feat(results['log_probs'], results['attention_mask'])
            log_probs_list.append(log_probs)
            
            confs = mask_select_feat(results['confs'][:,:,None], results['attention_mask'])
            confs_list.append(confs)
            
        max_conf_idx = torch.cat(confs_list, dim=1).argmax(dim=1)
        
        log_probs_mat = torch.stack(log_probs_list)
        log_probs = log_probs_mat[max_conf_idx, torch.arange(max_conf_idx.shape[0], device=max_conf_idx.device)]

        outputs = {f"log_probs{i+1}": log_probs_list[i] for i in range(len(log_probs_list))}
        outputs["log_probs"]=log_probs
        
        return outputs
    

