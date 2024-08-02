import sys
sys.path.append('/huyuqi/xmyu/DiffSDS')
import torch
from torch.utils.data import DataLoader
import numpy as np
from Bio import PDB
from src.tools.utils import cuda
from src.interface.model_interface import MInterface_base
from omegaconf import OmegaConf

class InferenceInterface(MInterface_base):
    def __init__(self, model_name=None, **kargs):
        super().__init__(**kargs)
        self.save_hyperparameters()
        self.load_model()
        self.cross_entropy = torch.nn.NLLLoss(reduction='none')

    def forward(self, batch):
        batch = self.model._get_features(batch)
        results = self.model(batch)
        log_probs, mask = results['log_probs'], batch['mask']
        cmp = log_probs.argmax(dim=-1) == batch['S']
        recovery = (cmp * mask).sum() / (mask.sum())
        return recovery

    def load_model(self):
        params = OmegaConf.load(f'./src/models/configs/{self.hparams.model_name}.yaml')
        params.update(self.hparams)

        if self.hparams.model_name == 'GraphTrans':
            from src.models.graphtrans_model import GraphTrans_Model
            self.model = GraphTrans_Model(params)
        
        elif self.hparams.model_name == 'StructGNN':
            from src.models.structgnn_model import StructGNN_Model
            self.model = StructGNN_Model(params)
            
        elif self.hparams.model_name == 'GVP':
            from src.models.gvp_model import GVP_Model
            self.model = GVP_Model(params)

        elif self.hparams.model_name == 'GCA':
            from src.models.gca_model import GCA_Model
            self.model = GCA_Model(params)

        elif self.hparams.model_name == 'AlphaDesign':
            from src.models.alphadesign_model import AlphaDesign_Model
            self.model = AlphaDesign_Model(params)

        elif self.hparams.model_name == 'ProteinMPNN':
            from src.models.proteinmpnn_model import ProteinMPNN_Model
            self.model = ProteinMPNN_Model(params)

        elif self.hparams.model_name == 'ESMIF':
            pass

        elif self.hparams.model_name == 'PiFold':
            from src.models.pifold_model import PiFold_Model
            self.model = PiFold_Model(params)

        elif self.hparams.model_name == 'KWDesign':
            from src.models.kwdesign_model import Design_Model
            self.model = Design_Model(params)
        
        elif self.hparams.model_name == 'E3PiFold':
            from src.models.E3PiFold_model import E3PiFold
            self.model = E3PiFold(params)

# 特征提取方法
def extract_features_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    coords = {'CA': [], 'C': [], 'O': [], 'N': []}
    sequence = []
    for chain in model:
        for residue in chain:
            if PDB.is_aa(residue):
                sequence.append(PDB.Polypeptide.three_to_one(residue.get_resname()))
                for atom_name in coords.keys():
                    if atom_name in residue:
                        coords[atom_name].append(residue[atom_name].get_coord())
                    else:
                        coords[atom_name].append(np.zeros(3))

    # 转换为numpy数组
    for key in coords:
        coords[key] = np.array(coords[key])
        
    chain_length = len(sequence)
    chain_mask = np.ones(chain_length)
    
    return {
        'seq': ''.join(sequence),
        'CA': coords['CA'],
        'C': coords['C'],
        'O': coords['O'],
        'N': coords['N'],
        'chain_mask': chain_mask,
        'chain_encoding': 1 * chain_mask
    }

# 加载模型
def load_model(model_path):
    # 从检查点加载模型
    model = InferenceInterface.load_from_checkpoint(checkpoint_path=model_path)
    model.eval()
    return model

# 推理函数
def inference(model, input_data):
    # 转换为PyTorch张量
    for key in input_data:
        if isinstance(input_data[key], np.ndarray):
            input_data[key] = torch.tensor(input_data[key], dtype=torch.float32)
    
    # 创建数据加载器
    dataset = [input_data]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results = []
    with torch.no_grad():
        for batch in dataloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(batch)  # 使用 InferenceInterface 中的 forward 方法
            results.append(output)
    return results

# 主程序
def main():
    parser = argparse.ArgumentParser(description="Protein Structure Inference")
    parser.add_argument('--pdb_file', type=str, required=True, help="Path to the PDB file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--model_name', type=str, required=True, help="Model name")

    args = parser.parse_args()

    # 提取特征
    input_data = extract_features_from_pdb(args.pdb_file)
    
    # 加载模型
    model = load_model(args.model_path)
    
    # 进行推理
    results = inference(model, input_data)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
