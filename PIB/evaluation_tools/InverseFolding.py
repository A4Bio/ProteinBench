from .tools import inference, reload_model, parsePDB, calculate_metrics, download_and_unzip_model, save_fasta
import argparse
import os

def pdb2fasta(pdb_path, sv_fasta_path, topk, model_name='UniIF', temp=1.0):
    download_and_unzip_model(model_name)
    model = reload_model('UniIF', model_name)
    all_pred_seqs = []
    all_names = []
    for pdb_file in os.listdir(pdb_path):
        protein = parsePDB(pdb_path+'/'+pdb_file)
        
        pred_seqs, scores, true_seq = inference(model, protein, model_name, topk=topk, temp=temp)
        names = [pdb_file.split('.')[0]+f'pred_{i}' for i in range(len(pred_seqs))]
        all_pred_seqs.extend(pred_seqs)
        all_names.extend(names)
    
    save_fasta(all_pred_seqs, all_names, sv_fasta_path)

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--model_name', default='UniIF', type=str)
    parser.add_argument('--pdb_path', default='PIB/evaluation_tools/test', type=str)
    parser.add_argument('--sv_fasta_path', default='PIB/evaluation_tools/test.fasta', type=str)
    parser.add_argument('--topk', default=1, type=int)
    parser.add_argument('--temp', default=1.0, type=float)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = create_parser()
    download_and_unzip_model(args.model_name)
    model = reload_model('UniIF', args.model_name)
    all_pred_seqs = []
    all_names = []
    for pdb_file in os.listdir(args.pdb_path):
        protein = parsePDB(args.pdb_path+'/'+pdb_file)
        
        pred_seqs, scores, true_seq = inference(model, protein, args.model_name, topk=args.topk, temp=args.temp)
        names = [pdb_file.split('.')[0]+f'pred_{i}' for i in range(len(pred_seqs))]
        all_pred_seqs.extend(pred_seqs)
        all_names.extend(names)
    
    save_fasta(all_pred_seqs, all_names, args.sv_fasta_path)