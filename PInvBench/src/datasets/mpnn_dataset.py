import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.utils.data as data
from Bio.PDB import PDBParser
import torch
import random
import csv
from dateutil import parser
from .fast_dataloader import DataLoaderX
from torch.utils.data import DataLoader
import time

from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm


def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
    """

    Parallel map using joblib.

    Parameters
    ----------
    pickleable_fn : callable
        Function to map over data.
    data : iterable
        Data over which we want to parallelize the function call.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. By default, it is one less than
        the number of CPUs.
    verbose: int, optional
        The verbosity level. If nonzero, the function prints the progress messages.
        The frequency of the messages increases with the verbosity level. If above 10,
        it reports all iterations. If above 50, it sends the output to stdout.
    kwargs
        Additional arguments for :attr:`pickleable_fn`.

    Returns
    -------
    list
        The i-th element of the list corresponds to the output of applying
        :attr:`pickleable_fn` to :attr:`data[i]`.
    """
    if n_jobs is None:
        n_jobs = cpu_count() - 1

    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
    )



def build_training_clusters(params, debug):
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])
   
    if debug:
        val_ids = []
        test_ids = []
 
    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[4])] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]
    
    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    if debug:
        rows = rows[:20]
    for r in rows:
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
    if debug:
        valid=train       
    return train, valid, test


def loader_pdb(item,params):

    pdbid,chid = item[0].split('_')
    PREFIX = "%s/pdb/%s/%s"%(params['DIR'],pdbid[1:3],pdbid)
    
    # load metadata
    if not os.path.isfile(PREFIX+".pt"):
        return {'seq': np.zeros(5)}
    meta = torch.load(PREFIX+".pt")
    asmb_ids = meta['asmb_ids']
    asmb_chains = meta['asmb_chains']
    chids = np.array(meta['chains'])

    # find candidate assemblies which contain chid chain
    asmb_candidates = set([a for a,b in zip(asmb_ids,asmb_chains)
                           if chid in b.split(',')])

    # if the chains is missing is missing from all the assemblies
    # then return this chain alone
    if len(asmb_candidates)<1:
        chain = torch.load("%s_%s.pt"%(PREFIX,chid))
        L = len(chain['seq'])
        return {'seq'    : chain['seq'],
                'xyz'    : chain['xyz'],
                'idx'    : torch.zeros(L).int(),
                'masked' : torch.Tensor([0]).int(),
                'label'  : item[0]}

    # randomly pick one assembly from candidates
    asmb_i = random.sample(list(asmb_candidates), 1)

    # indices of selected transforms
    idx = np.where(np.array(asmb_ids)==asmb_i)[0]

    # load relevant chains
    chains = {c:torch.load("%s_%s.pt"%(PREFIX,c))
              for i in idx for c in asmb_chains[i]
              if c in meta['chains']}

    # generate assembly
    asmb = {}
    for k in idx:

        # pick k-th xform
        xform = meta['asmb_xform%d'%k]
        u = xform[:,:3,:3]
        r = xform[:,:3,3]

        # select chains which k-th xform should be applied to
        s1 = set(meta['chains'])
        s2 = set(asmb_chains[k].split(','))
        chains_k = s1&s2

        # transform selected chains 
        for c in chains_k:
            try:
                xyz = chains[c]['xyz']
                xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:,None,None,:]
                asmb.update({(c,k,i):xyz_i for i,xyz_i in enumerate(xyz_ru)})
            except KeyError:
                return {'seq': np.zeros(5)}

    # select chains which share considerable similarity to chid
    seqid = meta['tm'][chids==chid][0,:,1]
    homo = set([ch_j for seqid_j,ch_j in zip(seqid,chids)
                if seqid_j>params['HOMO']])
    # stack all chains in the assembly together
    seq,xyz,idx,masked = "",[],[],[]
    seq_list = []
    for counter,(k,v) in enumerate(asmb.items()):
        seq += chains[k[0]]['seq']
        seq_list.append(chains[k[0]]['seq'])
        xyz.append(v)
        idx.append(torch.full((v.shape[0],),counter))
        if k[0] in homo:
            masked.append(counter)

    return {'seq'    : seq,
            'xyz'    : torch.cat(xyz,dim=0),
            'idx'    : torch.cat(idx,dim=0),
            'masked' : torch.Tensor(masked).int(),
            'label'  : item[0]}

def get_pdbs(data, max_length=10000, num_units=1000000):
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0

 
    data = {k:v for k,v in data.items()}
    c1 += 1
    if 'label' in list(data):
        my_dict = {}
        s = 0
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        mask_list = []
        visible_list = []
        if len(list(np.unique(data['idx']))) < 352:
            for idx in list(np.unique(data['idx'])):
                letter = chain_alphabet[idx]
                res = np.argwhere(data['idx']==idx)
                initial_sequence= "".join(list(np.array(list(data['seq']))[res][0,]))
                if initial_sequence[-6:] == "HHHHHH":
                    res = res[:,:-6]
                if initial_sequence[0:6] == "HHHHHH":
                    res = res[:,6:]
                if initial_sequence[-7:-1] == "HHHHHH":
                    res = res[:,:-7]
                if initial_sequence[-8:-2] == "HHHHHH":
                    res = res[:,:-8]
                if initial_sequence[-9:-3] == "HHHHHH":
                    res = res[:,:-9]
                if initial_sequence[-10:-4] == "HHHHHH":
                    res = res[:,:-10]
                if initial_sequence[1:7] == "HHHHHH":
                    res = res[:,7:]
                if initial_sequence[2:8] == "HHHHHH":
                    res = res[:,8:]
                if initial_sequence[3:9] == "HHHHHH":
                    res = res[:,9:]
                if initial_sequence[4:10] == "HHHHHH":
                    res = res[:,10:]
                if res.shape[1] < 4:
                    pass
                else:
                    my_dict['seq_chain_'+letter]= "".join(list(np.array(list(data['seq']))[res][0,]))
                    concat_seq += my_dict['seq_chain_'+letter]
                    if idx in data['masked']:
                        mask_list.append(letter)
                    else:
                        visible_list.append(letter)
                    coords_dict_chain = {}
                    all_atoms = np.array(data['xyz'][res,])[0,] #[L, 14, 3]
                    coords_dict_chain['N_chain_'+letter]=all_atoms[:,0,:].tolist()
                    coords_dict_chain['CA_chain_'+letter]=all_atoms[:,1,:].tolist()
                    coords_dict_chain['C_chain_'+letter]=all_atoms[:,2,:].tolist()
                    coords_dict_chain['O_chain_'+letter]=all_atoms[:,3,:].tolist()
                    my_dict['coords_chain_'+letter]=coords_dict_chain
            my_dict['name']= data['label']
            my_dict['masked_list']= mask_list
            my_dict['visible_list']= visible_list
            my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
            my_dict['seq'] = concat_seq
            if len(concat_seq) <= max_length:
                return my_dict
    return None

def safe_iter(ID, split_dict, params, alphabet_set, max_length=1000):
    sel_idx = np.random.randint(0, len(split_dict[ID]))
    out = loader_pdb(split_dict[ID][sel_idx], params)
    entry = get_pdbs(out)
    if entry is None:
        return None
    
    seq = entry['seq']
    bad_chars = set([s for s in seq]).difference(alphabet_set)
    if len(bad_chars) != 0:
        return None
    
    if len(entry['seq']) > max_length:
        return None
    
    masked_chains = entry['masked_list']
    visible_chains = entry['visible_list']
    
    all_chains = masked_chains + visible_chains
    visible_temp_dict = {}
    masked_temp_dict = {}
    
    for step, letter in enumerate(all_chains):
        chain_seq = entry[f'seq_chain_{letter}']
        if letter in visible_chains:
            visible_temp_dict[letter] = chain_seq
        elif letter in masked_chains:
            masked_temp_dict[letter] = chain_seq
            
    for km, vm in masked_temp_dict.items():
        for kv, vv in visible_temp_dict.items():
            if vm == vv:
                if kv not in masked_chains:
                    masked_chains.append(kv)
                if kv in visible_chains:
                    visible_chains.remove(kv)
    
    all_chains = masked_chains + visible_chains
    random.shuffle(all_chains)
    
    
    x_chain_list = []
    chain_mask_list = []
    chain_seq_list = []
    chain_encoding_list = []
    c = 1
    
    for step, letter in enumerate(all_chains):
        if letter in visible_chains:
            chain_seq = entry[f'seq_chain_{letter}']
            chain_length = len(chain_seq)
            chain_coords = entry[f'coords_chain_{letter}'] #this is a dictionary
            chain_mask = np.zeros(chain_length) #0.0 for visible chains
            x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_length,4,3]
            x_chain_list.append(x_chain)
            chain_mask_list.append(chain_mask)
            chain_seq_list.append(chain_seq)
            chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
            c+=1
        elif letter in masked_chains: 
            chain_seq = entry[f'seq_chain_{letter}']
            chain_length = len(chain_seq)
            chain_coords = entry[f'coords_chain_{letter}'] #this is a dictionary
            chain_mask = np.ones(chain_length) #0.0 for visible chains
            x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
            x_chain_list.append(x_chain)
            chain_mask_list.append(chain_mask)
            chain_seq_list.append(chain_seq)
            chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
            c+=1
    
    chain_mask_all = torch.from_numpy(np.concatenate(chain_mask_list))
    chain_encoding_all = torch.from_numpy(np.concatenate(chain_encoding_list))
    x_chain_all = torch.from_numpy(np.concatenate(x_chain_list))
    
    data = {
        "title":entry['name'],
        "seq":''.join(chain_seq_list), #len(seq)=n
        "chain_mask":chain_mask_all,
        "chain_encoding":chain_encoding_all,
        "CA":x_chain_all[:,1], # [n,3]
        "C":x_chain_all[:,2],
        "O":x_chain_all[:,3],
        "N":x_chain_all[:,0]} # [n,]
    return data

class MPNNDataset(data.Dataset):
    def __init__(self, data_path='/gaozhangyang/drug_dataset/proteinmpnn_data/pdb_2021aug02', rescut=3.5, split='train'):
        self.data_path = data_path
        self.rescut = rescut
        self.params = {
            "LIST"    : f"{self.data_path}/list.csv", 
            "VAL"     : f"{self.data_path}/valid_clusters.txt",
            "TEST"    : f"{self.data_path}/test_clusters.txt",
            "DIR"     : f"{self.data_path}",
            "DATCUT"  : "2030-Jan-01",
            "RESCUT"  : self.rescut, #resolution cutoff for PDBs
            "HOMO"    : 0.70 #min seq.id. to detect homo chains
        }
        
        if not os.path.exists("/gaozhangyang/experiments/OpenCPD/data/mpnn_data/split.pt"):
            train, valid, test = build_training_clusters(self.params, False)
            split = {"train": train, "valid":valid, "test":test}
            torch.save(split, "/gaozhangyang/experiments/OpenCPD/data/mpnn_data/split.pt")
        else:
            split = torch.load("/gaozhangyang/experiments/OpenCPD/data/mpnn_data/split.pt")
        
        self.split_dict = split[mode]
        alphabet='ACDEFGHIKLMNPQRSTVWYX'
        self.alphabet_set = set([a for a in alphabet])
        self.IDs = list(self.split_dict.keys())
        # self.data = self.preprocess()
    
    def cache_split(self,):
        train, valid, test = build_training_clusters(self.params, False)
        
        return {"train": train, "valid":valid, "test":test}
    
    @classmethod
    def safe_iter(self, ID, split_dict, params, alphabet_set, max_length=1000):
        # sel_idx = np.random.randint(0, len(split_dict[ID]))
        sel_idx = 0
        out = loader_pdb(split_dict[ID][sel_idx], params)
        entry = get_pdbs(out)
        if entry is None:
            return None
        
        seq = entry['seq']
        bad_chars = set([s for s in seq]).difference(alphabet_set)
        if len(bad_chars) != 0:
            return None
        
        if len(entry['seq']) > max_length:
            return None
        
        masked_chains = entry['masked_list']
        visible_chains = entry['visible_list']
        
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        
        for step, letter in enumerate(all_chains):
            chain_seq = entry[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
                
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)
        
        
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = entry[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = entry[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                c+=1
            elif letter in masked_chains: 
                chain_seq = entry[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = entry[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                c+=1
        
        chain_mask_all = np.concatenate(chain_mask_list)
        chain_encoding_all = np.concatenate(chain_encoding_list)
        x_chain_all = np.concatenate(x_chain_list)
        
        data = {
            "title":entry['name']+str(int(chain_mask_all.sum())),
            "seq":''.join(chain_seq_list), #len(seq)=n
            "chain_mask":chain_mask_all,
            "chain_encoding":chain_encoding_all,
            "CA":x_chain_all[:,1], # [n,3]
            "C":x_chain_all[:,2],
            "O":x_chain_all[:,3],
            "N":x_chain_all[:,0]} # [n,]
        return data

        
    
    def preprocess(self):
        data = pmap_multi(self.safe_iter, [(ID,) for ID in self.IDs], split_dict=self.split_dict, params=self.params, alphabet_set=self.alphabet_set)
        return data
        
    def __len__(self):
        # return len(self.data)
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        out = self.safe_iter(ID, split_dict=self.split_dict, params=self.params, alphabet_set=self.alphabet_set)
        return out


def collate_fn(batch):
    return batch


if __name__ == "__main__":
    MPNNDataset = MPNNDataset()
    loader = DataLoaderX(local_rank=0, dataset = MPNNDataset, collate_fn=collate_fn, batch_size=4)
    # loader = DataLoader(dataset = MPNNDataset, collate_fn=collate_fn, batch_size=4, prefetch_factor=4, num_workers=4)
    for batch in tqdm(loader):
        for one in batch:
            if one is not None:
                for key, val in one.items():
                    if type(val) == torch.Tensor:
                        result = val.to('cuda:0')
        time.sleep(2)
    print()