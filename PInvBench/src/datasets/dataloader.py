import copy
import random
import os.path as osp

import torch
import torch.utils.data as data

from .cath_dataset import CATHDataset
from .alphafold_dataset import AlphaFoldDataset
from .ts_dataset import TSDataset
from .casp_dataset import CASPDataset
from .mpnn_dataset import MPNNDataset
from .featurizer import (featurize_AF, featurize_GTrans, featurize_GVP,
                         featurize_ProteinMPNN, featurize_Inversefolding)
from .fast_dataloader import DataLoaderX

class GTransDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=None, **kwargs):
        super(GTransDataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,**kwargs)
        self.featurizer = collate_fn


class BatchSampler(data.Sampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.
    
    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.
    
    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts))  
                        if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes
        self._form_batches()
    
    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)
    
    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: 
            self._form_batches()
        for batch in self.batches: 
            yield batch


class GVPDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, num_workers=0,
                 featurizer=None, max_nodes=3000, **kwargs):
        super(GVPDataLoader, self).__init__(dataset, 
                                            batch_sampler = BatchSampler(node_counts = [ len(data['seq']) for data in dataset], max_nodes=max_nodes), 
                                            num_workers = num_workers, 
                                            collate_fn = featurizer.collate,
                                            **kwargs)
        self.featurizer = featurizer


def load_data(data_name, method, batch_size, data_root, pdb_path, split_csv, max_nodes=3000, num_workers=8, removeTS=0, test_casp=False, **kwargs):
    if data_name == 'CATH4.2' or data_name == 'TS':
        cath_set = CATHDataset(osp.join(data_root, 'cath4.2'), mode='train', test_name='All', removeTS=removeTS)
        train_set, valid_set, test_set = map(lambda x: copy.copy(x), [cath_set] * 3)
        valid_set.change_mode('valid')
        test_set.change_mode('test')
        if data_name == 'TS':
            test_set = TSDataset(osp.join(data_root, 'ts'))
            
        collate_fn = featurize_GTrans
    elif data_name == 'CATH4.3':
        cath_set = CATHDataset(osp.join(data_root, 'cath4.3'), mode='train', test_name='All', removeTS=removeTS, version=4.3)
        train_set, valid_set, test_set = map(lambda x: copy.copy(x), [cath_set] * 3)
        valid_set.change_mode('valid')
        test_set.change_mode('test')
        
        collate_fn = featurize_GTrans
    elif data_name == 'AlphaFold':
        af_set = AlphaFoldDataset(osp.join(data_root, 'af2db'), upid=upid, mode='train', limit_length=limit_length, joint_data=joint_data)
        train_set, valid_set, test_set = map(lambda x: copy.copy(x), [af_set] * 3)
        valid_set.change_mode('valid')
        test_set.change_mode('test')
        collate_fn = featurize_AF
    elif data_name=='MPNN':
        train_set = MPNNDataset(mode='train')
        valid_set = MPNNDataset(mode='valid')
        test_set = MPNNDataset(mode='test')
        collate_fn = featurize_GTrans
    
    elif data_name == 'S350':
        cath_set = CATHDataset(osp.join(data_root, 's350'), mode='train', test_name='All', removeTS=removeTS, version=4.3)
        train_set, valid_set, test_set = map(lambda x: copy.copy(x), [cath_set] * 3)
        valid_set.change_mode('train')
        test_set.change_mode('train')
        
        collate_fn = featurize_GTrans
        
    elif data_name == 'Protherm':
        cath_set = CATHDataset(osp.join(data_root, 'protherm'), mode='train', test_name='All', removeTS=removeTS, version=4.3)
        train_set, valid_set, test_set = map(lambda x: copy.copy(x), [cath_set] * 3)
        valid_set.change_mode('valid')
        test_set.change_mode('test')
        
        collate_fn = featurize_GTrans
    if test_casp:
        test_set = CASPDataset(osp.join(data_root, 'casp15'))

    if method in ['AlphaDesign', 'PiFold', 'KWDesign', 'GraphTrans', 'StructGNN']:
        pass
    elif method == 'GVP':
        featurizer = featurize_GVP()
        collate_fn = featurizer.collate
    elif method == 'ProteinMPNN':
        collate_fn = featurize_ProteinMPNN
    elif method == 'ESMIF':
        collate_fn = featurize_Inversefolding
        
    # train_set.data = train_set.data[:100]
    # valid_set.data = valid_set.data[:100]
    # test_set.data = test_set.data[:100]

    train_loader = DataLoaderX(local_rank=0, dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, prefetch_factor=8)
    valid_loader = DataLoaderX(local_rank=0,dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, prefetch_factor=8)
    test_loader = DataLoaderX(local_rank=0,dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, prefetch_factor=8)
        
    return train_loader, valid_loader, test_loader


def make_cath_loader(test_set, method, batch_size, max_nodes=3000, num_workers=8):
    if method in ['pifold','adesign', 'graphtrans', 'structgnn', 'gca']:
        collate_fn = featurize_GTrans
        test_loader = GTransDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    elif method == 'gvp':
        featurizer = featurize_GVP()
        test_loader = GVPDataLoader(test_set, num_workers=num_workers, featurizer=featurizer, max_nodes=max_nodes)
    elif method == 'proteinmpnn':
        collate_fn = featurize_ProteinMPNN
        test_loader = GTransDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    elif method == 'esmif':
        collate_fn = featurize_Inversefolding
        test_loader = GTransDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return test_loader
