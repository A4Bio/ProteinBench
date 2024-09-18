import torch
import numpy as np
import itertools
import torch.nn.functional as F
import math
import torch_geometric
import torch_cluster
from collections.abc import Mapping, Sequence
from torch_geometric.data import Data, Batch
from torch.utils.data.dataloader import default_collate
from ..tools.affine_utils import Rigid
from ..datasets.protein_rna_dataset import MyTokenizer
from ..tools.affine_utils import Rigid, Rotation, get_interact_feats
from torch_geometric.nn.pool import knn_graph
from torch_scatter import scatter_sum
from ..modules.uniif_module import *
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers") # mask token: 32

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def shuffle_subset(n, p):
    n_shuffle = np.random.binomial(n, p)
    ix = np.arange(n)
    ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
    ix_subset_shuffled = np.copy(ix_subset)
    np.random.shuffle(ix_subset_shuffled)
    ix[ix_subset] = ix_subset_shuffled
    return ix


def featurize_AF(batch, shuffle_fraction=0.):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.zeros([B, L_max])

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1) # [#atom, 4, 3]
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, )) # [#atom, 4, 3]
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        if shuffle_fraction > 0.:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices[idx_shuffle]
            score[i,:l] = b['score'][idx_shuffle]
        else:
            S[i, :l] = indices
            score[i,:l] = b['score']

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int)
    S_new = np.zeros_like(S)
    score_new = np.zeros_like(score)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]
        score_new[i,:n] = score[i][mask[i]==1]

    X = X_new
    S = S_new
    score = score_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, S, score, mask, lengths


def featurize_GTrans(batch):
    """ Pack and pad batch into torch tensors """
    # alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    batch = [one for one in batch if one is not None]
    B = len(batch)
    if B==0:
        return None
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.ones([B, L_max]) * 100.0
    chain_mask = np.zeros([B, L_max])-1 # 1:需要被预测的掩码部分 0:可见部分
    chain_encoding = np.zeros([B, L_max])-1
    

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1) # [#atom, 4, 3]
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, )) # [#atom, 4, 3]
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.array(tokenizer.encode(b['seq'], add_special_tokens=False))
        # indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)


        S[i, :l] = indices
        chain_mask[i,:l] = b['chain_mask']
        chain_encoding[i,:l] = b['chain_encoding']

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int32)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    lengths = torch.from_numpy(lengths)
    chain_mask = torch.from_numpy(chain_mask)
    chain_encoding = torch.from_numpy(chain_encoding)
    
    return {"title": [b['title'] for b in batch],
            "X":X,
            "S":S,
            "score": score,
            "mask":mask,
            "lengths":lengths,
            "chain_mask":chain_mask,
            "chain_encoding":chain_encoding}


class featurize_GVP:
    def __init__(self, num_positional_embeddings=16, top_k=30, num_rbf=16):
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        # self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
        #                'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
        #                'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
        #                'N': 2, 'Y': 18, 'M': 12}
        # self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
    
    def featurize(self, batch):
        data_all = []
        for b in batch:
            if b is None:
                continue
            coords = torch.tensor(np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1))
            seq = torch.tensor(np.array(tokenizer.encode(b['seq'], add_special_tokens=False)))
        
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1].float()
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            
            pos_embeddings = self._positional_embeddings(edge_index) # [E, 16]
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]] # [E, 3]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf) # [E, 16]
            
            dihedrals = self._dihedrals(coords)  # [n,6]
            orientations = self._orientations(X_ca) # [n,2,3]
            sidechains = self._sidechains(coords) # [n,3]
            
            node_s = dihedrals.float() # [n,6]

            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2).float() # [n, 3, 3]

            edge_s = torch.cat([rbf, pos_embeddings], dim=-1).float() # [E, 32]
            edge_v = _normalize(E_vectors).unsqueeze(-2).float() # [E, 1, 3]
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,(node_s, node_v, edge_s, edge_v))
            
            data = torch_geometric.data.Data(x=X_ca, seq=seq,
                                            node_s=node_s, node_v=node_v,
                                            edge_s=edge_s, edge_v=edge_v,
                                            edge_index=edge_index, mask=mask)
            data_all.append(data)
        return data_all
    
    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 
    
    def collate(self, batch):
        batch = self.featurize(batch)
        if (batch is None) or (len(batch)==0):
            return None
        
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))


def featurize_ProteinMPNN(batch, is_testing=False, chain_dict=None, fixed_position_dict=None, omit_AA_dict=None, tied_positions_dict=None, pssm_dict=None, bias_by_res_dict=None):
    """ Pack and pad batch into torch tensors """
    batch = [one for one in batch if one is not None]
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    if B==0:
        return None
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    pssm_coef_all = np.zeros([B, L_max], dtype=np.float32) #1.0 for the bits that need to be predicted
    pssm_bias_all = np.zeros([B, L_max, 21], dtype=np.float32) #1.0 for the bits that need to be predicted
    pssm_log_odds_all = 10000.0*np.ones([B, L_max, 21], dtype=np.float32) #1.0 for the bits that need to be predicted
    chain_M_pos = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.zeros([B, L_max])
    omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)
    # Build the batch
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []
    # shuffle all chains before the main loop
    for i, b in enumerate(batch):
        if chain_dict != None:
            masked_chains, visible_chains = chain_dict[b['name']] #masked_chains a list of chain letters to predict [A, D, F]
        else:
            # masked_chains = [item[-1:] for item in list(b) if item[:10]=='seq_chain_']
            masked_chains = ['']
            visible_chains = []
        # num_chains = b['num_of_chains']
        all_chains = masked_chains + visible_chains
        #random.shuffle(all_chains)
    for i, b in enumerate(batch):
        mask_dict = {}
        a = 0
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1]+chain_length)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0*np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))
            if letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                # chain_seq = b[f'seq_chain_{letter}']
                chain_seq = b[f'seq{letter}']
                chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1]+chain_length)
                masked_chain_length_list.append(chain_length)
                # chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_coords = b
                chain_mask = np.ones(chain_length) #1.0 for masked
                # x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain = np.stack([chain_coords[c] for c in [f'N', f'CA', f'C', f'O']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict!=None:
                    fixed_pos_list = fixed_position_dict[b['name']][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list)-1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_AA_dict!=None:
                    for item in omit_AA_dict[b['name']][letter]:
                        idx_AA = np.array(item[0])-1
                        AA_idx = np.array([np.argwhere(np.array(list(alphabet))== AA)[0][0] for AA in item[1]]).repeat(idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                        omit_AA_mask_temp[idx_[:,0], idx_[:,1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0*np.ones([chain_length, 21])
                if pssm_dict:
                    if pssm_dict[b['name']][letter]:
                        pssm_coef = pssm_dict[b['name']][letter]['pssm_coef']
                        pssm_bias = pssm_dict[b['name']][letter]['pssm_bias']
                        pssm_log_odds = pssm_dict[b['name']][letter]['pssm_log_odds']
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[b['name']][letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

       
        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict!=None:
            tied_pos_list = tied_positions_dict[b['name']]
            if tied_pos_list:
                set_chains_tied = set(list(itertools.chain(*[list(item) for item in tied_pos_list])))
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[np.argwhere(letter_list_np == k)[0][0]]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(start_idx+v[0][v_count]-1)#make 0 to be the first
                                tied_beta[start_idx+v[0][v_count]-1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(start_idx+v_-1)#make 0 to be the first
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)
 
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)
        m_pos = np.concatenate(fixed_position_mask_list,0) #[L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(pssm_coef_list,0) #[L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(pssm_bias_list,0) #[L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(pssm_log_odds_list,0) #[L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(bias_by_res_list, 0)  #[L,21], 0.0 for places where AA frequencies don't need to be tweaked

        l = len(all_sequence)
        x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        if 'score' in b.keys():
            score[i, :l] = b['score']
        else:
            score[i, :l] = 100.0
 
        m_pad = np.pad(m, [[0, L_max-l]], 'constant', constant_values=(0.0, ))
        m_pos_pad = np.pad(m_pos, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        omit_AA_mask_pad = np.pad(np.concatenate(omit_AA_mask_list,0), [[0,L_max-l], [0, 0]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad
        chain_M_pos[i,:] = m_pos_pad
        omit_AA_mask[i,] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        pssm_coef_pad = np.pad(pssm_coef_, [[0, L_max-l]], 'constant', constant_values=(0.0, ))
        pssm_bias_pad = np.pad(pssm_bias_, [[0, L_max-l], [0,0]], 'constant', constant_values=(0.0, ))
        pssm_log_odds_pad = np.pad(pssm_log_odds_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))

        pssm_coef_all[i,:] = pssm_coef_pad
        pssm_bias_all[i,:] = pssm_bias_pad
        pssm_log_odds_all[i,:] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(bias_by_res_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))
        bias_by_res_all[i,:] = bias_by_res_pad

        # Convert to labels
        indices = np.array(tokenizer.encode(b['seq'], add_special_tokens=False))
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)

    
    # mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    # numbers = np.sum(mask, axis=1).astype(np.int)
    # S_new = np.zeros_like(S)
    # X_new = np.zeros_like(X)+np.nan
    # for i, n in enumerate(numbers):
    #     X_new[i,:n,::] = X[i][mask[i]==1]
    #     S_new[i,:n] = S[i][mask[i]==1]

    # X = X_new
    # S = S_new
    
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(dtype=torch.float32)
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(dtype=torch.float32)
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(dtype=torch.float32)

    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32)

    jumps = ((residue_idx[:,1:]-residue_idx[:,:-1])==1).astype(np.float32)
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(dtype=torch.float32)
    phi_mask = np.pad(jumps, [[0,0],[1,0]])
    psi_mask = np.pad(jumps, [[0,0],[0,1]])
    omega_mask = np.pad(jumps, [[0,0],[0,1]])
    dihedral_mask = np.concatenate([phi_mask[:,:,None], psi_mask[:,:,None], omega_mask[:,:,None]], -1) #[B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long)
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    score = torch.from_numpy(score).float()
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long)

    if is_testing is False:
        return {"title": [b['title'] for b in batch],
                "X":X,
                "S":S,
                "score": score,
                "mask":mask,
                "lengths":lengths,
                "chain_M":chain_M,
                "chain_M_pos":chain_M_pos,
                "residue_idx":residue_idx,
                "chain_encoding_all":chain_encoding_all}
    else:
        return {"title": [b['title'] for b in batch],
                "X":X,
                "S":S,
                "score": score,
                "mask":mask,
                "lengths":lengths,
                "chain_M":chain_M,
                "chain_M_pos":chain_M_pos,
                "residue_idx":residue_idx,
                "chain_encoding_all":chain_encoding_all}
        
        



def featurize_Inversefolding(batch, shuffle_fraction=0.):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 3, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.ones([B, L_max]) * 100.0
    chain_mask = np.zeros([B, L_max])-1 # 1:需要被预测的掩码部分 0:可见部分
    chain_encoding = np.zeros([B, L_max])-1

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b[c] for c in ['N', 'CA', 'C']], 1) # [#atom, 4, 3]
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, )) # [#atom, 3, 3]
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.array(tokenizer.encode(b['seq'], add_special_tokens=False))
        if shuffle_fraction > 0.:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices[idx_shuffle]
        else:
            S[i, :l] = indices
        
        chain_mask[i,:l] = b['chain_mask']
        chain_encoding[i,:l] = b['chain_encoding']

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    chain_mask = torch.from_numpy(chain_mask)
    chain_encoding = torch.from_numpy(chain_encoding)
    return {"title": [b['title'] for b in batch],
            "X":X,
            "S":S,
            "score": score,
            "mask":mask,
            "lengths":lengths,
            "chain_mask":chain_mask,
            "chain_encoding":chain_encoding}


class featurize_UniIF:
    def __init__(self) -> None:
        self.tokenizer = MyTokenizer()
        self.virtual_frame_num = 3

    def _get_features_persample(self, batch):
        for key in batch:
            try:
                batch[key] = batch[key][None,...]
            except:
                batch[key] = batch[key]
        S = []
        for seq in batch['seq']:
            S.extend(self.tokenizer.encode(seq))
        S = torch.tensor(S)
        
        X = torch.from_numpy(np.stack([np.concatenate(batch['N']),
                            np.concatenate(batch['CA']),
                            np.concatenate(batch['C']),
                            np.concatenate(batch['O'])], axis=1)).float()

        chain_mask = torch.from_numpy(np.concatenate(batch['chain_mask'])).float()
        chain_encoding = torch.from_numpy(np.concatenate(batch['chain_encoding'])).float()


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
        
        edge_idx = knn_graph(C_a, k=30, batch=batch_id, loop=True, flow='target_to_source')

        
        
        N, CA, C = X[:,0], X[:,1], X[:,2]

        T = Rigid.make_transform_from_reference(N.float(), CA.float(), C.float())
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        T_ts = T[dst_idx,None].invert().compose(T[src_idx,None])

        # global virtual frames
        num_global = self.virtual_frame_num
        
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
        edge_idx_g = torch.stack([global_dst, global_src])
        edge_idx_g_inv = torch.stack([global_src, global_dst])
        edge_idx_g = torch.cat([edge_idx_g, edge_idx_g_inv], dim=1)

        batch_id_g = torch.zeros(num_global,dtype=batch_id.dtype)
        T_all = Rigid.cat([T, T_g], dim=0)

        idx, _ = edge_idx_g.min(dim=0)
        T_gs = T_all[idx,None].invert().compose(T_all[idx,None])

        rbf_ts = rbf(T_ts._trans.norm(dim=-1), 0, 50, 16)[:,0].view(_E.shape[0],-1)
        rbf_gs = rbf(T_gs._trans.norm(dim=-1), 0, 50, 16)[:,0].view(edge_idx_g.shape[1],-1)

        _V_g = torch.arange(num_global)
        _E_g = torch.zeros([edge_idx_g.shape[1], 128])

        mask = torch.masked_select(mask, mask_bool)
        chain_features = (chain_encoding[edge_idx[0]] == chain_encoding[edge_idx[1]]).int()

        batch={
                'T':T,
                'T_g': T_g,
                'T_ts': T_ts,
                'T_gs': T_gs,
                'rbf_ts': rbf_ts,
                'rbf_gs': rbf_gs,
                'X':X,
                'chain_features': chain_features,
                '_V': _V,
                '_E': _E,
                '_V_g': _V_g,
                '_E_g': _E_g,
                'S':seq,
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
    
    def featurize(self,batch):
        res = []
        for one in batch:
            temp = self._get_features_persample(one)
            res.append(temp)
        return self.custom_collate_fn(res)
    
    def custom_collate_fn(self, batch):
        batch = [one for one in batch if one is not None]
        num_nodes = torch.cat([one['num_nodes'] for one in batch])
        shift = num_nodes.cumsum(dim=0)
        shift = torch.cat([torch.tensor([0], device=shift.device), shift], dim=0)
        def shift_node_idx(idx, num_node, shift_real, shift_virtual):
            mask = idx>=num_node
            shift_combine = (~mask)*(shift_real) + (mask)*(shift_virtual)
            return idx+shift_combine

        
        ret = {}
        for key in batch[0].keys():
            if batch[0][key] is None:
                continue
            
            if key in ['T', 'T_g', 'T_ts', 'T_gs']:
                T = Rigid.cat([one[key] for one in batch], dim=0)
                ret[key+'_rot'] = T._rots._rot_mats
                ret[key+'_trans'] = T._trans
            elif key in ['edge_idx']:
                ret[key] = torch.cat([one[key] + shift[idx] for idx, one in enumerate(batch)], dim=1)
            elif key in ['edge_idx_g']:
                edge_idx_g = []
                for idx, one in enumerate(batch):
                    shift_virtual = shift[-1] + idx*one['K_g']-num_nodes[idx]
                    src = shift_node_idx(one['edge_idx_g'][0], num_nodes[idx], shift[idx], shift_virtual)
                    dst_g = shift_node_idx(one['edge_idx_g'][1], num_nodes[idx], shift[idx], shift_virtual) 
                    edge_idx_g.append(torch.stack([src, dst_g]))
                ret[key] = torch.cat(edge_idx_g, dim=1)
                # edge_idx_g = torch.cat(edge_idx_g, dim=1)
                # edge_idx_g_inv = edge_idx_g.flip((0,))
                # ret[key] = torch.cat([edge_idx_g, edge_idx_g_inv], dim=1)
            elif key in ['batch_id', 'batch_id_g']:
                ret[key] = torch.cat([one[key] + idx for idx, one in enumerate(batch)])
            elif key in ['K_g']:
                pass
            else:
                ret[key] = torch.cat([one[key] for one in batch], dim=0)

        return ret
        
        
    
# def uniif_collate_fn(batch):
#     batch = [one for one in batch if one is not None]
#     num_nodes = torch.tensor([one['CA'].shape[0] for one in batch])
#     shift = num_nodes.cumsum(dim=0)
#     shift = torch.cat([torch.tensor([0], device=shift.device), shift], dim=0)
#     def shift_node_idx(idx, num_node, shift_real, shift_virtual):
#         mask = idx>=num_node
#         shift_combine = (~mask)*(shift_real) + (mask)*(shift_virtual)
#         return idx+shift_combine

    
#     ret = {}
#     for key in batch[0].keys():
#         if key in ['T', 'T_g', 'T_ts', 'T_gs']:
#             T = Rigid.cat([one[key] for one in batch], dim=0)
#             ret[key+'_rot'] = T._rots._rot_mats
#             ret[key+'_trans'] = T._trans
#         elif key in ['edge_idx']:
#             ret[key] = torch.cat([one[key] + shift[idx] for idx, one in enumerate(batch)], dim=1)
#         elif key in ['edge_idx_g']:
#             edge_idx_g = []
#             for idx, one in enumerate(batch):
#                 shift_virtual = shift[-1] + idx*one['K_g']-num_nodes[idx]
#                 src = shift_node_idx(one['edge_idx_g'][0], num_nodes[idx], shift[idx], shift_virtual)
#                 dst_g = shift_node_idx(one['edge_idx_g'][1], num_nodes[idx], shift[idx], shift_virtual) 
#                 edge_idx_g.append(torch.stack([src, dst_g]))
#             ret[key] = torch.cat(edge_idx_g, dim=1)
#             # edge_idx_g = torch.cat(edge_idx_g, dim=1)
#             # edge_idx_g_inv = edge_idx_g.flip((0,))
#             # ret[key] = torch.cat([edge_idx_g, edge_idx_g_inv], dim=1)
#         elif key in ['batch_id', 'batch_id_g']:
#             ret[key] = torch.cat([one[key] + idx for idx, one in enumerate(batch)])
#         elif key in ['K_g']:
#             pass
#         else:
#             try:
#                 ret[key] = torch.cat([one[key] for one in batch], dim=0)
#             except:
#                 ret[key] = [one[key] for one in batch]

#     return ret
