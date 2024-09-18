import os
import os.path as osp
import torch
import numpy as np
from PIB.evaluation_tools.design_interface import MInterface
import json
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
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

def beam_search(prob_matrix, beam_width):
    """
    Beam search algorithm for sequence prediction.

    Args:
    - prob_matrix (numpy.ndarray): The probability matrix of shape [L, K].
    - beam_width (int): The number of top sequences to keep (beam size).

    Returns:
    - sequences (list of lists): The top-k sequences.
    - sequence_scores (list of floats): The corresponding scores for the top-k sequences.
    """
    L, K = prob_matrix.shape  # L: sequence length, K: number of classes
    
    # Initialize the beam with an empty sequence and zero log probability
    sequences = [[[], 0.0]]  # Each element is a tuple of (sequence, log_prob)
    
    for t in range(L):
        all_candidates = []
        
        # Expand each sequence in the current beam
        for seq, score in sequences:
            for i in range(K):
                # Get the probability of the i-th class at time step t
                prob = prob_matrix[t, i]
                
                # Calculate new log probability (additive in log space)
                new_score = score + np.log(prob + 1e-10)  # Avoid log(0) by adding a small epsilon
                
                # Create a new candidate sequence
                candidate = (seq + [i], new_score)
                
                # Append to all candidates
                all_candidates.append(candidate)
        
        # Sort candidates by score (log probability) and select top beam_width sequences
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]
    
    # Extract the sequences and their scores
    final_sequences = [seq for seq, score in sequences]
    final_scores = [score for seq, score in sequences]
    
    return final_sequences, final_scores

def save_fasta(seqs, names, pred_fasta_path='pred.fasta', true_fasta_path='true.fasta'):
    with open(pred_fasta_path, 'w') as pred_file:
        for seq, name in zip(seqs, names):
            pred_file.write(f">{name}\n")
            pred_file.write(seq + "\n")

def reload_model(data_name,model_name):
    if model_name != 'UniIF':
      default_params = json.load(open(f'/train/results/{data_name}/{model_name}/model_param.json'))

      config = {}
      config.update(default_params)
    else:
      config = {}
    config['load_memory'] = False
    config['is_colab'] = True
    config['ex_name'] = f'{model_name}'
    config['model_name'] = model_name
    config['res_dir'] = 'PIB/evaluation_tools'
    config['data_root'] = '/data/cath4.3'
    config['pretrained_path'] = osp.join(config['res_dir'],  config['ex_name'],'checkpoint.pth')
    config['config_path'] = osp.join(config['res_dir'],  config['ex_name'],f'{model_name}.yaml')
    model = MInterface(**config)
    model.eval()
    return model

def inference(model, protein_input, model_name, topk=5, temp=1.0):
    from PIB.src.datasets.featurizer import MyTokenizer, featurize_GTrans, featurize_GVP, featurize_ProteinMPNN,featurize_UniIF


    with torch.no_grad():
        if model_name == 'KWDesign':
            protein = featurize_GTrans([protein_input])
            tocuda = lambda x: x.to(model.device) if isinstance(x, torch.Tensor) else x
            protein = {key: tocuda(val) for key, val in protein.items()}
            features = model.model.Design1.design_model.PretrainPiFold._get_features(protein)
            print(model.device)
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

        elif model_name =='PiFold' or model_name == 'GraphTrans' or model_name =='GCA' or model_name =='StructGNN':
            protein = featurize_GTrans([protein_input])
            tocuda = lambda x: x.to(model.device) if isinstance(x, torch.Tensor) else x
            protein = {key: tocuda(val) for key, val in protein.items()}
            batch = model.model._get_features(protein)

        elif model_name == 'GVP':
            featurizer = featurize_GVP()
            protein = featurizer.featurize([protein_input])[0]
            # tocuda = lambda x: x.to(model.device) if isinstance(x, torch.Tensor) else x
            # protein = {key: tocuda(val) for key, val in protein.items()}
            protein = protein.to(model.device)
            batch = model.model._get_features(protein)

        elif model_name == 'ProteinMPNN':
            protein = featurize_ProteinMPNN([protein_input])
            tocuda = lambda x: x.to(model.device) if isinstance(x, torch.Tensor) else x
            protein = {key: tocuda(val) for key, val in protein.items()}
            batch = model.model._get_features(protein)

        elif model_name == 'UniIF':
            featurizer = featurize_UniIF()
            protein = featurizer.featurize([protein_input])
            tocuda = lambda x: x.to(model.device) if isinstance(x, torch.Tensor) else x
            protein = {key: tocuda(val) for key, val in protein.items()}
            # protein = protein.to(model.device)
            batch = model.model._get_features(protein)

        results = model.model(batch)
        logits = results['logits']

    probs = torch.softmax(logits/temp, dim=-1)
    if model_name == 'UniIF':
        tokenizer = MyTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    # probs = log_probs.exp()


    if model_name in ['GCA', 'StructGNN', 'GraphTrans','ProteinMPNN']:
        probs = probs.squeeze(0)

    results = beam_search(probs, topk)
    # pred_S = probs.argmax(dim=-1)
    pred_seqs = []
    for seq in results[0]:
        pred_seq = "".join(tokenizer.decode(seq).split(" "))
        pred_seqs.append(pred_seq)
    scores = results[1]
    return pred_seqs, scores, protein_input['seq']

def calculate_metrics(true_seq, pred_seq, pred_probs):

    true_seq_encoded = [ord(c) for c in true_seq]
    pred_seq_encoded = [ord(c) for c in pred_seq]


    true_seq_tensor = torch.tensor(true_seq_encoded, dtype=torch.long)
    pred_seq_tensor = torch.tensor(pred_seq_encoded, dtype=torch.long)


    recovery = (true_seq_tensor == pred_seq_tensor).float().mean().item()

    confidence = pred_probs.max(dim=-1)[0].mean().item()

    macro_f1 = f1_score(true_seq_encoded, pred_seq_encoded, average='macro')

    return {
        "Recovery": recovery,
        "Confidence": confidence,
        "Macro_F1": macro_f1
    }


def download_and_unzip_model(model_name):
    model_urls = {
        'GVP': 'https://zenodo.org/records/13630171/files/GVP.zip?download=1',
        'GCA': 'https://zenodo.org/records/13630171/files/GCA.zip?download=1',
        'GraphTrans': 'https://zenodo.org/records/13630171/files/GraphTrans.zip?download=1',
        'ProteinMPNN': 'https://zenodo.org/records/13630171/files/ProteinMPNN.zip?download=1',
        'StructGNN': 'https://zenodo.org/records/13630171/files/StructGNN.zip?download=1',
        'KWDesign': 'https://zenodo.org/records/13251279/files/KWDesign.zip?download=1',
        'PiFold': 'https://zenodo.org/records/13630171/files/PiFold.zip?download=1',
        'UniIF':'https://zenodo.org/records/13738616/files/UniIF.zip?download=1'
    }

    model_dir = f'PIB/evaluation_tools/{model_name}'
    zip_path = f'{model_dir}/{model_name}.zip'

    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        print(f"Downloading {model_name} model...")
        os.system(f'wget -O {zip_path} {model_urls[model_name]}')

    if not os.path.exists(f'{model_dir}/checkpoint.pth'):
        print(f"Unzipping {model_name} model...")
        os.system(f'unzip {zip_path} -d {model_dir}')
        
#@title Run ESMFold to make the structure prediction, saved as pred.pdb
from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import softmax
import gc

def parse_output(output):
    pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
    plddt = output["plddt"][0,:,1]

    bins = np.append(0,np.linspace(2.3125,21.6875,63))
    sm_contacts = softmax(output["distogram_logits"],-1)[0]
    sm_contacts = sm_contacts[...,bins<8].sum(-1)
    xyz = output["positions"][-1,0,:,1]
    mask = output["atom37_atom_exists"][0,:,1] == 1
    o = {"pae":pae[mask,:][:,mask],
        "plddt":plddt[mask],
        "sm_contacts":sm_contacts[mask,:][:,mask],
        "xyz":xyz[mask]}
    return o

def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()

def read_fasta(fasta_file):
    with open(fasta_file, 'r') as f:
        lines = f.readlines()

    seq_name = ""
    sequence = ""
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if sequence:
                break
            seq_name = line[1:]
        else:
            sequence += line.strip()

    return seq_name, sequence

def fold_sequences_with_esmfold(fasta_file, output_dir='./pred_pdbs/', model_name='esmfold_v1.model'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_name, sequence = read_fasta(fasta_file)
    sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
    sequence = re.sub(":+",":",sequence)
    sequence = re.sub("^[:]+","",sequence)
    sequence = re.sub("[:]+$","",sequence)
    copies = 1
    if copies == "" or copies <= 0: copies = 1
    sequence = ":".join([sequence] * copies)
    num_recycles = 3 #@param ["0", "1", "2", "3"] {type:"raw"}
    chain_linker = 25

    ID = seq_name + "_" + get_hash(sequence)[:5]
    seqs = sequence.split(":")
    lengths = [len(s) for s in seqs]
    length = sum(lengths)

    if "model" not in dir() or model_name != model_name_:
        if "model" in dir():
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model = torch.load(model_name)
        model.eval().cuda().requires_grad_(False)
        model_name_ = model_name

    if length > 700:
        model.set_chunk_size(64)
    else:
        model.set_chunk_size(128)

    torch.cuda.empty_cache()
    output = model.infer(sequence,
                         num_recycles=num_recycles,
                         chain_linker="X"*chain_linker,
                         residue_index_offset=512)

    pdb_str = model.output_to_pdb(output)[0]
    output = output.cpu().numpy()
    ptm = output["ptm"][0]
    plddt = output["plddt"][0,...,1].mean()
    O = parse_output(output)
    print(f'ptm: {ptm:.3f} plddt: {plddt:.3f}')

    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{output_dir}/{ID}_ptm{ptm:.3f}_r{num_recycles}_default"
    np.savetxt(f"{prefix}.pae.txt",O["pae"],"%.3f")
    with open("pred.pdb","w") as out:
        out.write(pdb_str)
    print(f"Saved PDB for {seq_name} to pred.pdb")


def install_esmfold():
    #@title Install ESMFold, adapted from the official ESMFold Colab https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb#scrollTo=CcyNpAvhTX6q
    #@markdown Install ESMFold, OpenFold and download Params (~2min 30s)
    version = "1"
    model_name = "esmfold_v0.model" if version == "0" else "esmfold.model"
    import os, time
    if not os.path.isfile(model_name):
        # download esmfold params
        os.system("apt-get install aria2 -qq")
        os.system(f"aria2c -q -x 16 https://colabfold.steineggerlab.workers.dev/esm/{model_name} &")

        if not os.path.isfile("finished_install"):
            # install libs
            print("installing libs...")
            os.system("pip install -q omegaconf pytorch_lightning biopython ml_collections einops py3Dmol modelcif")
            os.system("pip install -q git+https://github.com/NVIDIA/dllogger.git")

            print("installing openfold...")
            # install openfold
            os.system(f"pip install -q git+https://github.com/sokrypton/openfold.git")

            print("installing esmfold...")
            # install esmfold
            os.system(f"pip install -q git+https://github.com/sokrypton/esm.git")
            os.system("touch finished_install")

    # wait for Params to finish downloading...
    while not os.path.isfile(model_name):
        time.sleep(5)
    if os.path.isfile(f"{model_name}.aria2"):
        print("downloading params...")
    while os.path.isfile(f"{model_name}.aria2"):
        time.sleep(5)

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