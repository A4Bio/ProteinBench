from evaluation_tools.tools import inference, reload_model, parsePDB, calculate_metrics, download_and_unzip_model, save_fasta
import argparse
import os

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--model_name', default='UniIF', type=str)
    parser.add_argument('--pdb_path', default='evaluation_tools/test', type=str)
    parser.add_argument('--sv_fasta_path', default='evaluation_tools/test.fasta', type=str)
    parser.add_argument('--topk', default=1, type=int)
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
        
        pred_seqs, scores, true_seq = inference(model, protein, args.model_name, topk=args.topk)
        names = [pdb_file.split('.')[0]+f'pred_{i}' for i in range(len(pred_seqs))]
        all_pred_seqs.extend(pred_seqs)
        all_names.extend(names)
    
    save_fasta(all_pred_seqs, all_names, args.sv_fasta_path)