import os
import json
import numpy as np
import torch.utils.data as data


class TSDataset(data.Dataset):
    def __init__(self, path = './', split='test'):
        if not os.path.exists(path):
            raise "no such file:{} !!!".format(path)
        else:
            ts50_data = json.load(open(path+'/ts50.json'))
            ts500_data = json.load(open(path+'/ts500.json'))

        # TS500 has proteins with lengths of 500+
        # TS50 only contains proteins with lengths less than 500
        self.data = []
        for temp in ts50_data:
            coords = np.array(temp['coords'])
            self.data.append({'title':temp['name'],
                                'seq':temp['seq'],
                                'CA':coords[:,1,:],
                                'C':coords[:,2,:],
                                'O':coords[:,3,:],
                                'N':coords[:,0,:],
                                'category': 'ts50'
                                })

        for temp in ts500_data:
            coords = np.array(temp['coords'])
            self.data.append({'title':temp['name'],
                                'seq':temp['seq'],
                                'CA':coords[:,1,:],
                                'C':coords[:,2,:],
                                'O':coords[:,3,:],
                                'N':coords[:,0,:],
                                'category': 'ts500'
                                })

    def __len__(self):
        return len(self.data)
    
    def get_item(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self.data[index]