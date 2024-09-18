import os
import gzip
import numpy as np
from collections import defaultdict

AAMAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'SEC': 'U', 'PYL': 'O', 'XLE': 'J', '': '-'
}


def getSequence(resnames):
    """Returns polypeptide sequence as from list of *resnames* (residue
    name abbreviations)."""

    get = AAMAP.get
    return ''.join([get(rn, 'X') for rn in resnames])


def gzip_open(filename, *args, **kwargs):
    if args and "t" in args[0]:
        args = (args[0].replace("t", ""), ) + args[1:]
    if isinstance(filename, str):
        return gzip.open(filename, *args, **kwargs)
    else:
        return gzip.GzipFile(filename, *args, **kwargs)


def parsePDB(pdb, chain=['A']):
    title, ext = os.path.splitext(os.path.split(pdb)[1])
    title, ext = os.path.splitext(title)
    if pdb[-3:] == '.gz':
        pdb = gzip_open(pdb, 'rt')
        lines = defaultdict(list)
        for loc, line in enumerate(pdb):
            line = line.decode('ANSI_X3.4-1968')
            startswith = line[0:6]
            lines[startswith].append((loc, line))
    else:
        pdb = open(pdb)
        lines = defaultdict(list)
        for loc, line in enumerate(pdb):
            # line = line.decode('ANSI_X3.4-1968')
            startswith = line[0:6]
            lines[startswith].append((loc, line))

    pdb.close()

    sequence = ''

    CA_coords, C_coords, O_coords, N_coords = [], [], [], []

    # chain_id = []
    for idx, line in lines['ATOM  ']:
        if line[21:22].strip() not in chain:
            continue
        if line[13:16].strip() == 'CA':
            CA_coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            CA_coords.append(CA_coord)
            sequence += ''.join(getSequence([line[17:20]]))
        elif line[13:16].strip() == 'C':
            C_coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            C_coords.append(C_coord)
        elif line[13:16].strip() == 'O':
            O_coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            O_coords.append(O_coord)
        elif line[13:16].strip() == 'N':
            N_coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            N_coords.append(N_coord)

    chain_length = len(sequence)
    chain_mask = np.ones(chain_length)

    return {'title': title,
            'seq': sequence,
            'CA': np.array(CA_coords),
            'C': np.array(C_coords),
            'O': np.array(O_coords),
            'N': np.array(N_coords),
            'score': 100.0,
            'chain_mask': chain_mask,
            'chain_encoding': 1 * chain_mask
            }


def get_parser():
  import argparse
  parser = argparse.ArgumentParser()
  # Set-up parameters
  parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
  parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
  parser.add_argument('--res_dir', default='ProDesign/results', type=str)
  parser.add_argument('--ex_name', default='ProDesign', type=str)
  parser.add_argument('--use_gpu', default=True, type=bool)
  parser.add_argument('--gpu', default=0, type=int)
  parser.add_argument('--seed', default=111, type=int)

  # CATH
  # dataset parameters
  parser.add_argument('--data_name', default='CATH', choices=['CATH', 'TS50'])
  parser.add_argument('--data_root', default='ProDesign/data/')
  parser.add_argument('--batch_size', default=8, type=int)
  parser.add_argument('--num_workers', default=8, type=int)

  # method parameters
  parser.add_argument('--method', default='ProDesign', choices=['ProDesign'])
  parser.add_argument('--config_file', '-c', default=None, type=str)
  parser.add_argument('--hidden_dim',  default=128, type=int)
  parser.add_argument('--node_features',  default=128, type=int)
  parser.add_argument('--edge_features',  default=128, type=int)
  parser.add_argument('--k_neighbors',  default=30, type=int)
  parser.add_argument('--dropout',  default=0.1, type=int)
  parser.add_argument('--num_encoder_layers', default=10, type=int)

  # Training parameters
  parser.add_argument('--epoch', default=100, type=int, help='end epoch')
  parser.add_argument('--log_step', default=1, type=int)
  parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
  parser.add_argument('--patience', default=100, type=int)

  # ProDesign parameters
  parser.add_argument('--updating_edges', default=4, type=int)
  parser.add_argument('--node_dist', default=1, type=int)
  parser.add_argument('--node_angle', default=1, type=int)
  parser.add_argument('--node_direct', default=1, type=int)
  parser.add_argument('--edge_dist', default=1, type=int)
  parser.add_argument('--edge_angle', default=1, type=int)
  parser.add_argument('--edge_direct', default=1, type=int)
  parser.add_argument('--virtual_num', default=3, type=int)
  args = parser.parse_args([])
  return args