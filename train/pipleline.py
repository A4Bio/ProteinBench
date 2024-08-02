import sys
sys.path.append('/gaozhangyang/experiments/ProteinInvBench/')
import os
import argparse
import torch
import numpy as np
from Bio import PDB
from torch.utils.data import DataLoader
from design_interface import MInterface  
import pytorch_lightning as pl
import json
from src.datasets.deploy_utils import parsePDB
from src.datasets.featurizer import featurize_GTrans
from transformers import AutoTokenizer


# 加载模型
def reload_model(model_path, model_name):
    default_params = json.load(open(f'/gaozhangyang/experiments/OpenCPD/results/CATH4.3/KWDesign/model_param.json'))
    # default_params = json.load(open(f'/gaozhangyang/experiments/OpenCPD/results/MPNN/{ex_name}/model_param.json'))
    # default_params = json.load(open(f'./results/PiFold/{ex_name}/model_param.json'))
    config = {}
    config.update(default_params)
    config['load_memory'] = False
    # config['ex_name'] = model_name
    config['model_name'] = model_name
    config['res_dir'] = '/gaozhangyang/experiments/OpenCPD/results'
    config['data_root'] = '/gaozhangyang/experiments/ProteinInvBench/data/cath4.3'
    model = MInterface(**config)
    model.eval()
    return model

def inference(model, protein, model_name):
    from src.datasets.featurizer import featurize_GTrans
    protein = featurize_GTrans([protein])
    tocuda = lambda x: x.to(model.device) if isinstance(x, torch.Tensor) else x
    protein = {key: tocuda(val) for key, val in protein.items()}
    
    X, S, mask, score, lengths, chain_mask = protein['X'], protein['S'], protein['mask'], protein['score'], protein['lengths'], protein['chain_mask']
        
    with torch.no_grad():   
        if model_name == 'KWDesign':
            # 调用 _get_features 方法
            features = model.model.Design1.design_model.PretrainPiFold._get_features(protein)
            X, S, score, h_V, h_E, E_idx, batch_id, chain_mask, chain_encoding = features['X'], features['S'], features['score'], features['_V'], features['_E'], features['E_idx'], features['batch_id'], features['chain_mask'], features['chain_encoding']

            seq_mask = None
            batch = {
                "title": protein['title'], 
                "h_V": h_V, 
                "h_E": h_E, 
                "E_idx": E_idx, 
                "batch_id": batch_id, 
                "alphabet": 'ACDEFGHIKLMNPQRSTVWYX', 
                "S": S, 
                'position': X, 
                'seq_mask': seq_mask
            }

            results = model.model(batch)
            log_probs = results['log_probs']
    
    return log_probs

# 主程序
def main():
    parser = argparse.ArgumentParser(description="Protein Structure Inference")
    parser.add_argument('--pdb_file', type=str, default='/gaozhangyang/experiments/ProteinInvBench/train/1a2b.pdb', help="Path to the PDB file")
    parser.add_argument('--model_path', type=str, default='/gaozhangyang/experiments/ProteinInvBench/model_zoom/CATH4.2/KWDesign/checkpoint.pth', help="Path to the model checkpoint")
    parser.add_argument('--model_name', type=str, default='KWDesign', help="Model name")

    args = parser.parse_args()

    model = reload_model(args.model_path, args.model_name)
    
    protein = parsePDB(args.pdb_file)
    log_probs = inference(model, protein,args.model_name)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers")
    probs = log_probs.exp()
    temperature = 0.01

    if temperature == 0:
        pred_seq_id = torch.argmax(probs, 1).squeeze(-1)
    else:
        pred_seq_id = torch.multinomial(torch.softmax(probs / temperature, dim=-1), 1).squeeze(-1)

    pred_S = probs.argmax(dim=-1)

    pred_seq = "".join(tokenizer.decode(pred_S).split(" "))
    print(pred_seq)
    print(protein['seq'])

if __name__ == "__main__":
    main()
