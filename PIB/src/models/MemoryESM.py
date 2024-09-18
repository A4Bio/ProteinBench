import subprocess
import os
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import shutil
from .PretrainESM_model import PretrainESM_Model


class MemoESM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.PretrainESM = PretrainESM_Model(args)
        self.tokenizer = self.PretrainESM.tokenizer
        self.memory = {}
        # self.fix_memory = False
    
    # def save_memory(self, path):
    #     params = {key:val for key,val in self.state_dict().items() if "GNNTuning" in key}
    #     torch.save({"params":params,"memory": self.memory}, path)

    # def load_memory(self, path):
    #     data = torch.load(path)
    #     self.load_state_dict(data['params'], strict=False)
    #     self.memory = data['memory']
        
    def clean_input(self, batch, score_cut=0.99):
        '''
        require: batch['pred_ids'], batch['attention_mask'], batch['confs']
        '''
        symbol = "<mask>"
        replace_dict = {"-":symbol,
                        ".":symbol,
                        "<eos>":symbol,
                        "<unk>":symbol,
                        "<cls>":symbol,
                        "<pad>":symbol,
                        "<null_1>":symbol,
                        "<mask>":symbol,
                        "U":symbol,
                        "O":symbol}

        device = batch['pred_ids'].device
        query_seqs = []
        for pred_ids, mask, score in zip(batch['pred_ids'], batch['attention_mask'], batch['confs']):
            seq = self.tokenizer.decode(pred_ids[mask], clean_up_tokenization_spaces=False)
            elements = []
            for idx, x in enumerate(seq.split(" ")):
                symbol = replace_dict.get(x, x)
                if score[idx] < score_cut:
                    symbol = "<mask>"
                elements.append(symbol)
            seq = "".join(elements)
            query_seqs.append(seq)
        
        results = self.tokenizer.batch_encode_plus(query_seqs, return_tensors="pt", padding=True)
        return query_seqs
    
    def initoutput(self, B, maxL, device):
        self.out_pred_ids = torch.zeros(B, maxL, dtype=torch.long, device=device)
        self.out_confs = torch.zeros(B, maxL, dtype=torch.float, device=device)
        self.out_embeds = torch.zeros(B, maxL, 1280, dtype=torch.float, device=device)
        self.titles = [None for i in range(B)]
        
    def retrivel(self, titles, num_nodes, device, use_memory):
        # retrieval
        unseen = []
        for idx in range(len(titles)):
            name = titles[idx]
            if (name in self.memory) and use_memory:
                memo_pred_ids = self.memory[name]['pred_ids'].to(device)
                memo_confs = self.memory[name]['confs'].to(device)
                memo_embeds = self.memory[name]['embeds'].to(device)
                
                self.out_pred_ids[idx, :num_nodes[idx]] = memo_pred_ids
                self.out_confs[idx, :num_nodes[idx]] = memo_confs
                self.out_embeds[idx, :num_nodes[idx]] = memo_embeds
                self.titles[idx] = name
            else:
                unseen.append(idx)
        return unseen
    
    def rebatch(self, unseen, batch):
        unseen_pred_ids = []
        unseen_attention_mask = []
        for i in unseen:
            unseen_pred_ids.append(batch['pred_ids'][i])
            unseen_attention_mask.append(batch['attention_mask'][i])
        unseen_pred_ids = torch.stack(unseen_pred_ids)
        unseen_attention_mask = torch.stack(unseen_attention_mask)
        return {"pred_ids":unseen_pred_ids, "attention_mask":unseen_attention_mask}
    
    def save2memory(self, unseen,outputs, titles, unseen_attention_mask):
         # save to memory
        for i in range(len(unseen)):
            name = titles[unseen[i]]
            self.titles[unseen[i]] = name
            mask = unseen_attention_mask[i]
            self.memory[name] = {"pred_ids":outputs['pred_ids'][i][mask].detach().to('cpu'), 
                                "confs":outputs['confs'][i][mask].detach().to('cpu'), 
                                "embeds":outputs['embeds'][i][mask].detach().to('cpu')}
    
    def update(self, unseen, unseen_attention_mask, num_nodes, outputs):
        # update
        for idx in range(len(unseen)):
            mask = unseen_attention_mask[idx]==1
            self.out_pred_ids[unseen[idx], :num_nodes[unseen[idx]]] = outputs['pred_ids'][idx][mask]
            self.out_confs[unseen[idx], :num_nodes[unseen[idx]]] = outputs['confs'][idx][mask]
            self.out_embeds[unseen[idx], :num_nodes[unseen[idx]]] = outputs['embeds'][idx][mask]
    
    @torch.no_grad()
    def forward(self, batch, use_memory=False):
        # debatch
        # clean_seqs = self.clean_input(batch)
        device = batch['probs'].device
        B, maxL, _ = batch['probs'].shape
        num_nodes = batch['attention_mask'].sum(dim=-1).tolist()
        self.initoutput(B, maxL, device)
        unseen = self.retrivel(batch['title'], num_nodes, device, use_memory)
        
        
        if len(unseen)>0:
            # batch forward
            new_batch = self.rebatch(unseen, batch)
            outputs = self.PretrainESM(new_batch)
            
            self.save2memory(unseen,outputs, batch['title'], new_batch['attention_mask'])
            self.update(unseen, new_batch['attention_mask'], num_nodes, outputs)
        
        return {'title':self.titles,'pred_ids':self.out_pred_ids, 'confs':self.out_confs, 'embeds':self.out_embeds, 'attention_mask':batch['attention_mask']}



if __name__ == '__main__': 
    
    # work_space = '/gaozhangyang/experiments/PiFoldV2/data/mmseq_workspace2'
    # target_seqs = ["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPQTKTYFPHFDLSHGSAQVKGHG", "MVHLTPEEKSAVTALWGKVNVDEVGVEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKV",
    #  "MVLSPADKTNVKAAWGKVGAGGAEALERMFLSFPQKTYYTYFPHFDLSHGSAQVKGHG"]

    # query_seqs = ["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKFPHFDLSHGSAQV", "MVHLTPEEKSAVTALWGKVNVDEVGGGRLLVVYPWTQRFFESFGDLSTPDAV",]
    
    # results = search_seqs(query_seqs, target_seqs, work_space)
    # print(results)
        

    import biotite.sequence as seq
    import biotite.sequence.align as align

    # Create example query and target protein sequences
    query_seq1 = seq.ProteinSequence("MSKXXKAFLNKXXL")
    target_seq1 = seq.ProteinSequence("MSKVKAALNKVLL")
    target_seq2 = seq.ProteinSequence("MSKVKKALNKVLL")
    target_seq3 = seq.ProteinSequence("MSTVAAALKMLLL")

    results = search_seqs_biotite(["MSKXXKAFLNKXXL"], ["MSKVKAALNKVLL", "MSKVKKALNKVLL", "MSTVAAALKMLLL"])

    # Print the alignments
    print("Query alignments:")



    