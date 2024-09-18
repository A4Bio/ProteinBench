import os
import json
import numpy as np
from tqdm import tqdm
import random
import torch.utils.data as data
from .utils import cached_property
from transformers import AutoTokenizer
import torch
from ..tools.affine_utils import Rigid, Rotation, get_interact_feats
from torch_geometric.nn.pool import knn_graph
from torch_scatter import scatter_sum
import _pickle as cPickle
from sklearn.cluster import KMeans

def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device, dtype=values.dtype)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)

def cross(u, v, dim=-1):
    dtype = u.dtype
    if dtype!=torch.float32:
        u = u.to(dtype=torch.float32)
        v = v.to(dtype=torch.float32)
    return torch.cross(u,v, dim).to(dtype=dtype)

class MyTokenizer:
    def __init__(self):
        self.alphabet_protein = 'ACDEFGHIKLMNPQRSTVWY' # [X] for unknown token
        self.alphabet_RNA = 'AUGC'
    
    def encode(self, seq, RNA=False):
        if RNA:
            return [self.alphabet_RNA.index(s) for s in seq]
        else:
            return [self.alphabet_protein.index(s) for s in seq]
        
    def decode(self, indices, RNA=False):
        if RNA:
            return ' '.join([self.alphabet_RNA[i] for i in indices])
        else:
            return ' '.join([self.alphabet_protein[i] for i in indices])
class Protein_RNA_Dataset(data.Dataset):
    def __init__(self, path='./data/',  split='train', max_length=500, test_name='All', data = None, version=4.3, k_neighbors=30, virtual_frame_num=3, dataname='RNA'):
        self.__dict__.update(locals())
        
        if data is None:
            self.all_data = self.cache_data()
            self.data = self.all_data[split]
        else:
            self.data = data
        
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers")
        self.tokenizer = MyTokenizer()
    
    def cache_data(self):
        data_dict = {'train': [], 'val': [], 'test': []}
        # # type 0: RNA, type 1: Protein
        if self.dataname=='RNA':
            #========================RNA data===============================#
            alphabet_set = set(['A', 'U', 'C', 'G'])
            self.path = 'data/RNAdata'
            if os.path.exists(self.path):
                data_dict = {'train': [], 'val': [], 'test': []}
                # val and test data
                for split in ['train','val','test']:
                    data = cPickle.load(open(os.path.join(self.path, split + '_data.pt'), 'rb'))
                    for entry in tqdm(data):
                        for key, val in entry['coords'].items():
                            entry[key] = np.asarray(val)
                        entry['title'] = ''
                        entry['type'] = 0
                        chain_length = len(entry['seq'])
                        entry['chain_mask'] = np.ones(chain_length)
                        entry['chain_encoding'] = 1 * entry['chain_mask']
                        bad_chars = set([s for s in entry['seq']]).difference(alphabet_set)
                        if len(bad_chars) == 0:
                            data_dict[split].append(entry)
        
        if self.dataname == 'Protein' or self.dataname == 'pdb':
            #=======================Protein data=============================
            alphabet='ACDEFGHIKLMNPQRSTVWY'
            alphabet_set = set([a for a in alphabet])
            self.path = 'data/cath4.3'
            if not os.path.exists(self.path):
                raise "no such file:{} !!!".format(self.path)
            else:
                with open(self.path+'/chain_set.jsonl') as f:
                    lines = f.readlines()
                data_list = []
                for line in tqdm(lines):
                    entry = json.loads(line)
                    seq = entry['seq']

                    for key, val in entry['coords'].items():
                        entry['coords'][key] = np.asarray(val)
                    
                    bad_chars = set([s for s in seq]).difference(alphabet_set)

                    if len(bad_chars) == 0:
                        if len(entry['seq']) <= self.max_length: 
                            chain_length = len(entry['seq'])
                            chain_mask = np.ones(chain_length)
                            data_list.append({
                                'title':entry['name'],
                                'type':1,
                                'seq':entry['seq'],
                                'CA':entry['coords']['CA'],
                                'C':entry['coords']['C'],
                                'O':entry['coords']['O'],
                                'N':entry['coords']['N'],
                                'chain_mask': chain_mask,
                                'chain_encoding': 1*chain_mask,
                                'chain_index': 1*chain_mask
                            })
                            
                with open(self.path+'/chain_set_splits.json') as f:
                    dataset_splits = json.load(f)
                
                if self.test_name == 'L100':
                    with open(self.path+'/test_split_L100.json') as f:
                        test_splits = json.load(f)
                    dataset_splits['test'] = test_splits['test']

                if self.test_name == 'sc':
                    with open(self.path+'/test_split_sc.json') as f:
                        test_splits = json.load(f)
                    dataset_splits['test'] = test_splits['test']
                
                name2set = {}
                name2set.update({name:'train' for name in dataset_splits['train']})
                name2set.update({name:'val' for name in dataset_splits['validation']})
                name2set.update({name:'test' for name in dataset_splits['test']})

                for data in data_list:
                    if name2set.get(data['title']):
                        if name2set[data['title']] == 'train':
                            data_dict['train'].append(data)
                        
                        if name2set[data['title']] == 'val':
                            data_dict['val'].append(data)
                        
                        if name2set[data['title']] == 'test':
                            data['category'] = 'Unkown'
                            data['score'] = 100.0
                            data_dict['test'].append(data)
        if self.dataname == 'pdb':
            self.path = 'data/pdb'
            if not os.path.exists(self.path):
                raise "no such file:{} !!!".format(self.path)
            else:
                with open(self.path+'/pdb.jsonl') as f:
                    lines = f.readlines()
                data_dict['train'] = []
                for line in tqdm(lines):
                    entry = json.loads(line)
                    seq = entry['seq']

                    for key, val in entry['coords'].items():
                        entry['coords'][key] = np.asarray(val)
                    
                    bad_chars = set([s for s in seq]).difference(alphabet_set)

                    if len(bad_chars) == 0:
                        if len(entry['seq']) <= self.max_length: 
                            chain_length = len(entry['seq'])
                            chain_mask = np.ones(chain_length)
                            data_dict['train'].append({
                                'title':entry['name'],
                                'type':1,
                                'seq':entry['seq'],
                                'CA':entry['coords']['CA'],
                                'C':entry['coords']['C'],
                                'O':entry['coords']['O'],
                                'N':entry['coords']['N'],
                                'chain_mask': chain_mask,
                                'chain_encoding': 1*chain_mask,
                                'chain_index': np.repeat(np.arange(len(entry['chain_length'])), list(entry['chain_length'].values()))
                            })

        return data_dict

    def change_mode(self, mode):
        self.data = self.cache_data[mode]
    
    def __len__(self):
        return len(self.data)
      
    def _get_features(self, batch):
        S, score, X, chain_mask, chain_encoding, chain_index = batch['seq'], batch['score'], batch['X'], batch['chain_mask'], batch['chain_encoding'], batch['chain_index']

        X, S = X.unsqueeze(0), S.unsqueeze(0)
        mask = torch.isfinite(torch.sum(X,(2,3))).float() # atom mask
        numbers = torch.sum(mask, axis=1).int()
        S_new = torch.zeros_like(S)
        X_new = torch.zeros_like(X)+torch.nan
        for i, n in enumerate(numbers):
            X_new[i,:n,::] = X[i][mask[i]==1]
            S_new[i,:n] = S[i][mask[i]==1]

        X = X_new
        S = S_new
        isnan = torch.isnan(X)
        mask = torch.isfinite(torch.sum(X,(2,3))).float()
        X[isnan] = 0.

        mask_bool = (mask==1)
        def node_mask_select(x):
            shape = x.shape
            x = x.reshape(shape[0], shape[1],-1)
            out = torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])
            out = out.reshape(-1,*shape[2:])
            return out

        batch_id = torch.arange(mask_bool.shape[0], device=mask_bool.device)[:,None].expand_as(mask_bool)

        seq = node_mask_select(S)
        X = node_mask_select(X)
        batch_id = node_mask_select(batch_id)
        C_a = X[:,1,:]
        
        edge_idx = knn_graph(C_a, k=self.k_neighbors, batch=batch_id, loop=True, flow='target_to_source')

        
        # if not (scatter_sum(torch.ones_like(edge_idx[0]), edge_idx[0])==30).all():
        #     return None
        
        N, CA, C = X[:,0], X[:,1], X[:,2]

        T = Rigid.make_transform_from_reference(N.float(), CA.float(), C.float())
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        T_ts = T[dst_idx,None].invert().compose(T[src_idx,None])

        # global virtual frames
        num_global = self.virtual_frame_num
        # # 创建KMeans模型并进行拟合
        # kmeans = KMeans(n_clusters=num_global, random_state=0).fit(C_a.numpy())

        # # 获取每个数据点所属的簇标签
        # labels = kmeans.labels_

        # rot_g, trans_g = [], []
        # for c in range(num_global):
        #     X_c = T[labels==c]._trans
        #     X_m = X_c.mean(dim=0, keepdim=True)
        #     X_c = X_c-X_m
        #     U,S,V = torch.svd(X_c.T@X_c)
        #     rot_g.append(U)
        #     trans_g.append(X_m)
        
        '''
        U的每一列，为原始空间中的坐标基向量
        R = U
        U2, S2, V2 = torch.svd((R@X_c.T)@(X_c@R.T))
        R@U == U2
        '''

        X_c = T._trans
        X_m = X_c.mean(dim=0, keepdim=True)
        X_c = X_c-X_m
        U,S,V = torch.svd(X_c.T@X_c)
        d = (torch.det(U) * torch.det(V)) < 0.0
        D = torch.zeros_like(V)
        D[ [0,1], [0,1]] = 1
        D[2,2] = -1*d+1*(~d)
        V = D@V
        R = torch.matmul(U, V.permute(0,1))

        # R = torch.zeros_like(R)
        # X_m = torch.zeros_like(X_m)

        rot_g = [R]*num_global
        trans_g = [X_m]*num_global
        
        feat = get_interact_feats(T, T_ts, X.float(), edge_idx, batch_id)
        _V, _E = feat['_V'], feat['_E']

        '''
        global_src: N+1,N+1,N+2,N+2,..N+B, N+B+1,N+B+1,N+B+2,N+B+2,..N+B+B
        global_dst: 0,  1,  2,  3,  ..N,   0,    1,    2,    3,    ..N
        batch_id_g: 1,  1,  2,  2,  ..B,   1,    1,    2,    2,    ..B
        '''
        T_g = Rigid(Rotation(torch.stack(rot_g)), torch.cat(trans_g,dim=0))
        num_nodes = scatter_sum(torch.ones_like(batch_id), batch_id)
        global_src = torch.cat([batch_id  +k*num_nodes.shape[0] for k in range(num_global)]) + num_nodes
        global_dst = torch.arange(batch_id.shape[0], device=batch_id.device).repeat(num_global)
        global_idx = torch.arange(num_global, device=batch_id.device) + num_nodes
        edge_idx_g = torch.stack([global_dst, global_src])
        edge_idx_g_inv = torch.stack([global_src, global_dst])
        edge_idx_g_inter = torch.stack([global_idx.repeat(num_nodes), global_idx.repeat_interleave(num_nodes)])
        # edge_idx_g = torch.cat([edge_idx_g, edge_idx_g_inv, edge_idx_g_inter], dim=1)
        edge_idx_g = torch.cat([edge_idx_g, edge_idx_g_inv], dim=1)

        batch_id_g = torch.zeros(num_global,dtype=batch_id.dtype)
        T_all = Rigid.cat([T, T_g], dim=0)
        # T_gs = T_all[edge_idx_g[1],None].invert().compose(T_all[edge_idx_g[0],None])
        idx, _ = edge_idx_g.min(dim=0)
        T_gs = T_all[idx,None].invert().compose(T_all[idx,None])

        rbf_ts = rbf(T_ts._trans.norm(dim=-1), 0, 50, 16)[:,0].view(_E.shape[0],-1)
        rbf_gs = rbf(T_gs._trans.norm(dim=-1), 0, 50, 16)[:,0].view(edge_idx_g.shape[1],-1)
        # rbf_sg = rbf(T_sg._trans.norm(dim=-1), 0, 50, 16)[:,0].view(_E_g.shape[0],-1)
        _V_g = torch.arange(num_global)
        _E_g = torch.zeros([edge_idx_g.shape[1], 128])

        mask = torch.masked_select(mask, mask_bool)
        chain_features = torch.from_numpy(chain_index[edge_idx[0]] == chain_index[edge_idx[1]]).int()

        batch={
                'T':T,
                'T_g': T_g,
                'T_ts': T_ts,
                'T_gs': T_gs,
                # 'T_sg': T_sg,
                'rbf_ts': rbf_ts,
                'rbf_gs': rbf_gs,
                # 'rbf_sg': rbf_sg,
                'X':X,
                'chain_features': chain_features,
                '_V': _V,
                '_E': _E,
                '_V_g': _V_g,
                '_E_g': _E_g,
                'seq':seq,
                # 'score':score,
                'edge_idx':edge_idx,
                'edge_idx_g': edge_idx_g,
                'batch_id': batch_id,
                'batch_id_g': batch_id_g,
                'num_nodes': num_nodes,
                'mask': mask,
                'chain_mask': chain_mask,
                'chain_encoding': chain_encoding,
                'K_g': num_global}
        return batch


    
    def merge_coords(self, item):
        X = []
        for name in ['P', "O5'", "C5'", "C4'", "C3'", "O3'", 'N', 'CA', 'C', 'O']:
            if name in item:
                X.append(torch.from_numpy(item[name]).unsqueeze(1))

        return torch.cat(X, dim=1)

    def __getitem__(self, index):
        item = self.data[index]
        L = len(item['seq'])
        if item['type']==0:
            item['seq'] = torch.LongTensor(self.tokenizer.encode(item['seq'], RNA=True)) + 20
        else:
            item['seq'] = torch.LongTensor(self.tokenizer.encode(item['seq'], RNA=False))
        item['score'] = 100.0
        item['X'] = self.merge_coords(item)
        item['chain_mask'] = torch.from_numpy(item['chain_mask'])
        item['chain_encoding'] = torch.from_numpy(item['chain_encoding'])
        
        if L>self.max_length:
            # 计算截断的最大索引
            max_index = L - self.max_length
            # 生成随机的截断索引
            truncate_index = random.randint(0, max_index)
            # 进行截断
            item['seq'] = item['seq'][truncate_index:truncate_index+self.max_length]
            item['X'] = item['X'][truncate_index:truncate_index+self.max_length]
            item['chain_mask'] = item['chain_mask'][truncate_index:truncate_index+self.max_length]
            item['chain_encoding'] = item['chain_encoding'][truncate_index:truncate_index+self.max_length]
        try:
            return self._get_features(item)
        except:
            return