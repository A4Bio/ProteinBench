import os
import os.path as osp
import json
import numpy as np
import pickle as cPickle

import torch.utils.data as data
from ..datasets.utils import cached_property


class AlphaFoldDataset(data.Dataset):
    def __init__(self, path='./', upid='', mode='train', max_length=500, limit_length=1, joint_data=0):
        
        self.path = path
        self.upid = upid
        self.max_length = max_length
        self.limit_length = limit_length
        self.joint_data = joint_data
        
        if mode in ['train', 'valid', 'test']:
            self.data = self.cache_data[mode]
        
        if mode == 'all':
            self.data = self.cache_data['train'] + self.cache_data['valid'] + self.cache_data['test']
        
        self.lengths = np.array([ len(sample['seq']) for sample in self.data])
        self.max_len = np.max(self.lengths)
        self.min_len = np.min(self.lengths)

    def _raw_data(self, path, upid):
        if not os.path.exists(path):
            raise "no such file:{} !!!".format(path)
        else:
            path = osp.join(path, upid)
            data_ = cPickle.load(open(path+'/data_{}.pkl'.format(upid),'rb'))
            score_ = cPickle.load(open(path+'/data_{}_score.pkl'.format(upid),'rb'))
            for i in range(len(data_)):
                data_[i]['score'] = score_[i]['res_score']
        return data_

    def _data_info(self, data):
        len_inds = []
        seq2ind = {}
        for ind, temp in enumerate(data):
            if self.limit_length:
                if 30 < len(temp['seq']) and len(temp['seq']) < self.max_length:
                    # 'title', 'seq', 'CA', 'C', 'O', 'N'
                    len_inds.append(ind)
                    seq2ind[temp['seq']] = ind
            else:
                len_inds.append(ind)
                seq2ind[temp['seq']] = ind
        return len_inds, seq2ind
        
    def get_data(self, path, upid, **kwargs):
        data_ = self._raw_data(path, upid)
        path = osp.join(path, upid)

        file_name = 'split_clu_l.json' if self.limit_length else 'split_clu.json'

        assert os.path.exists(osp.join(path, file_name))
        split = json.load(open(osp.join(path, file_name),'r'))
        data_dict = {'train':[data_[i] for i in split['train']],
                     'valid':[data_[i] for i in split['valid']],
                     'test':[data_[i] for i in split['test']]}
        return data_dict

    def get_full_data(self, path, **kwargs):
        datanames = [dataname for dataname in os.listdir(path) if ('_v2' in dataname)]
        file_name = 'split_clu_l.json' if self.limit_length else 'split_clu.json'
        assert os.path.exists(osp.join(path, 'full', file_name))
        split = json.load(open(osp.join(path, 'full', file_name),'r'))
        return split 
    
    @cached_property
    def cache_data(self): # TODO: joint_data
        path = self.path
        upid = self.upid
        if self.joint_data:
            datanames = [dataname for dataname in os.listdir(path) if ('_v2' in dataname)]
            data_dict = {'train':[], 'valid':[], 'test':[]}
            full_inds = self.get_full_data(path)

            for dataname in datanames:
                temp = self._raw_data(path, dataname)
                train_idx, valid_idx, test_idx = map(lambda fold: full_inds[dataname][fold], ['train', 'valid', 'test'])
                data_dict['train'] += [temp[i] for i in train_idx]
                data_dict['valid'] += [temp[i] for i in valid_idx]

                data_test = []
                for i in test_idx:
                    item = temp[i]
                    item['category'] = dataname
                    data_test.append(temp[i])
                
                data_dict['test'] += data_test

        else:
            data_dict = self.get_data(path, upid)
            for item in data_dict['test']:
                item['category'] = upid

        return data_dict
    
    def change_mode(self, mode):
        self.data = self.cache_data[mode]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]