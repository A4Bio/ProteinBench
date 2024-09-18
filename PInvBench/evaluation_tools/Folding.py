from PInvBench.evaluation_tools.tools import fold_sequences_with_esmfold, install_esmfold
import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--fasta_path', default='evaluation_tools/test.fasta', type=str)
    parser.add_argument('--sv_pdb_path', default='evaluation_tools/test', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = create_parser()
    # install_esmfold()
    fold_sequences_with_esmfold(args.fasta_path, output_dir=args.sv_pdb_path, model_name='esmfold.model')