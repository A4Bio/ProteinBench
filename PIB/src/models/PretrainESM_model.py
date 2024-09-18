import time
import torch
import math
import torch.nn as nn
from transformers import AutoTokenizer, EsmForMaskedLM # EsmForMaskedLM, 1041 line
import torch


class PretrainESM_Model(nn.Module):
    def __init__(self, args):
        """ Graph labeling network """
        super(PretrainESM_Model, self).__init__()
        self.args=args
        # {0: '<cls>', 1: '<pad>', 2: '<eos>', 3: '<unk>', 4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C', 24: 'X', 25: 'B', 26: 'U', 27: 'Z', 28: 'O', 29: '.', 30: '-', 31: '<null_1>', 32: '<mask>'}
        if args.is_colab == False:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers")
            self.model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="main/")
            self.model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="main/")
        
    
    def forward(self,batch):
        outputs = self.model(input_ids=batch['pred_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits

        prop = logits.softmax(dim=-1)
        confidences, pred_ids = prop.max(dim=-1)
            
        ret = {"pred_ids": pred_ids,
               "confs": confidences,
               "embeds": outputs.hidden_states,
               "attention_mask": batch['attention_mask']}
        return ret
    
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers")
    tokenizer.convert_ids_to_tokens
    print()