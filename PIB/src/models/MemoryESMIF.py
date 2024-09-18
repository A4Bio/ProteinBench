import subprocess
import os
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from .PretrainESMIF_model import PretrainESMIF_Model
from torch_scatter import scatter_sum

class MemoESMIF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.PretrainESMIF = PretrainESMIF_Model(args)
        self.memory = {}
        # self.fix_memory = False
    
    # def save_memory(self, path):
    #     params = {key:val for key,val in self.state_dict().items() if "GNNTuning" in key}
    #     torch.save({"params":params,"memory": self.memory}, path)

    # def load_memory(self, path):
    #     data = torch.load(path)
    #     self.load_state_dict(data['params'], strict=False)
    #     self.memory = data['memory']
        
    def initoutput(self, B, maxL, device):
        self.out_embeds = torch.zeros(B, maxL, 512, dtype=torch.float, device=device)
        self.titles = [None for i in range(B)]
        
    def retrivel(self, titles, num_nodes, device, use_memory):
        # retrieval
        unseen = []
        for idx in range(len(titles)):
            name = titles[idx]
            if (name in self.memory) and use_memory:
                memo_embeds = self.memory[name]['embeds'].to(device)
                self.out_embeds[idx, :num_nodes[idx]] = memo_embeds
                self.titles[idx] = name
            else:
                unseen.append(idx)
        return unseen
    
    def rebatch(self, unseen, batch):
        unseen_position = []
        for i in unseen:
            mask = batch['batch_id']==i
            unseen_position.append(batch['position'][mask][:,:3,:])
        return {"position":unseen_position}
    
    def save2memory(self, unseen,outputs, titles, num_nodes):
         # save to memory
        for i in range(len(unseen)):
            name = titles[unseen[i]]
            self.titles[unseen[i]] = name
            num = num_nodes[unseen[i]]
            self.memory[name] = {"embeds":outputs['feat'][i,:num].detach().to('cpu')}
    
    def update(self, unseen, num_nodes, outputs):
        # update
        for idx in range(len(unseen)):
            num = num_nodes[unseen[idx]]
            self.out_embeds[unseen[idx], :num_nodes[unseen[idx]]] = outputs['feat'][idx, :num]
    
    @torch.no_grad()
    def forward(self, batch, use_memory=False):
        # debatch
        # clean_seqs = self.clean_input(batch)
        device = batch['position'].device
        num_nodes = scatter_sum(torch.ones_like(batch['batch_id']), batch['batch_id'], dim=0)
        B, maxL = num_nodes.shape[0], num_nodes.max()
        self.initoutput(B, maxL, device)
        unseen = self.retrivel(batch['title'], num_nodes, device, use_memory)
        
        
        if len(unseen)>0:
            # batch forward
            new_batch = self.rebatch(unseen, batch)
            outputs = self.PretrainESMIF(new_batch['position'])
            
            self.save2memory(unseen,outputs, batch['title'], num_nodes)
            self.update(unseen, num_nodes, outputs)
        
        return {'title':self.titles, 'embeds':self.out_embeds}



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



    