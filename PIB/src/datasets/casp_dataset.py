import os
import json
import numpy as np
import torch.utils.data as data


class CASPDataset(data.Dataset):
    def __init__(self, path = './', split='test'):
        if not os.path.exists(path):
            raise "no such file:{} !!!".format(path)
        else:
            with open(os.path.join(path,'casp15.jsonl')) as f:
                lines = f.readlines()
                
            # casp15_data = json.load(open(path+'casp15.json', 'r'))

        alphabet='ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])
        
        self.data = []
        for line in lines:
            entry = json.loads(line)
            seq = entry['seq']

            for key, val in entry['coords'].items():
                entry['coords'][key] = np.asarray(val)
            
            bad_chars = set([s for s in seq]).difference(alphabet_set)

            if len(bad_chars) == 0:
                chain_length = len(entry['seq'])
                chain_mask = np.ones(chain_length)
                self.data.append({
                    'title':entry['name'],
                    'seq':entry['seq'],
                    'CA':entry['coords']['CA'],
                    'C':entry['coords']['C'],
                    'O':entry['coords']['O'],
                    'N':entry['coords']['N'],
                    'chain_mask': chain_mask,
                    'chain_encoding': 1*chain_mask,
                    'classification': entry['classification']
                })

    def __len__(self):
        return len(self.data)
    
    def get_item(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self.data[index]
    
if __name__ == '__main__':
    dataset = CASPDataset('/gaozhangyang/experiments/OpenCPD/data/casp15/')
    for data in dataset:
        print(data)