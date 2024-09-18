import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, EsmForMaskedLM
import torch.nn.functional as F

class PretrainInterface(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == "ESM35M":
            self.esm_dim = 480
            self.tokenizer = AutoTokenizer.from_pretrained("/huyuqi/model_zoom/transformers/models--facebook--esm2_t12_35M_UR50D")
            self.pretrain_model = EsmForMaskedLM.from_pretrained("/huyuqi/model_zoom/transformers/models--facebook--esm2_t12_35M_UR50D")
        if name == "ESM650M":
            self.esm_dim = 1280
            self.tokenizer = AutoTokenizer.from_pretrained("/huyuqi/model_zoom/transformers/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c")
            self.pretrain_model = EsmForMaskedLM.from_pretrained("/huyuqi/model_zoom/transformers/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c")
        if name == "ESM3B":
            self.esm_dim = 2560
            self.tokenizer = AutoTokenizer.from_pretrained("/huyuqi/model_zoom/transformers/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc")
            self.pretrain_model = EsmForMaskedLM.from_pretrained("/huyuqi/model_zoom/transformers/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc")
        
        if name == "vanilla":
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMVQ/base/configs/10-18T01-15-36-project.yaml")
            pretrain_args.diffusion = False
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMVQ/base/checkpoints/best-epoch=14-val_loss=0.314.pth', map_location=torch.device('cpu'))
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict, strict=False)
        
        if name == 'GearNet':
            from model.PretrainGearNet import PretrainGearNet_Model
            self.pretrain_model = PretrainGearNet_Model()
        
        self.pretrain_model.eval()
    
    def get_vq_id(self, seqs, angles, attn_mask):
        # if ('softgroup' in self.name) or ('LFQ' in self.name):
        #     h_input = self.pretrain_model.model.input(seqs.squeeze(-1), angles)
        #     h_enc = self.pretrain_model.model.ProteinEnc(h_input, attn_mask, None).last_hidden_state          
        #     vq_id, e_enc = self.pretrain_model.model.VQLayer.get_vq(h_enc, attn_mask, temperature=1e-5)
        #     return F.pad(vq_id, [0,1,0,0])

        h_input = self.pretrain_model.model.input(seqs.squeeze(-1), angles)
        h_enc = self.pretrain_model.model.ProteinEnc(h_input, attn_mask, None).last_hidden_state
        vq_id, e_enc = self.pretrain_model.model.VQLayer.get_vq(h_enc, attn_mask, temperature=1e-5)
        return vq_id
    
    def forward(self, batch):
        if self.name in ["ESM35M", "ESM650M", "ESM3B"]:
            seqs, attn_mask = batch['seqs'], batch['attn_mask']
            outputs = self.pretrain_model.model(input_ids=seqs[:,:,0], attention_mask=attn_mask)
            pretrain_embedding = outputs.hidden_states
            pretrain_embedding = pretrain_embedding.reshape(-1,self.esm_dim)[attn_mask.view(-1)==1]
            return pretrain_embedding
        if self.name in ["softgroup_128_group"]:
            seqs, angles, attn_mask = batch['seqs'], batch['angles'] , batch['attn_mask']
            vq_id = self.pretrain_model.model.get_vqid(seqs[...,0], angles, attn_mask)
            return vq_id
        if self.name in ["ProGLM"]:
            vq_id, attn_mask, seg, pos = batch['vq_id'], batch['attn_mask'], batch['seg'], batch['pos']
            feat = self.pretrain_model.model.get_feat(vq_id, attn_mask, seg, pos)
            feat = feat.reshape(-1,self.vq_dim)[attn_mask.view(-1)==1]
            return feat
        if self.name in ["GearNet"]:
            seqs = batch['seqs']
            batch = batch['batch']
            attn_mask = batch['attn_mask']
            for idx in range(seqs.shape[0]):
                seq_str = self.pretrain_featurizer.ESM_tokenizer.decode(seqs[idx,attn_mask[idx,:].bool(),0])
                seq_strs.append(seq_str.split(" "))
            seq_strs = sum(seq_strs, [])
            node_index = torch.arange(batch.batch.shape[0], device=batch.batch.device)
            node2graph = batch.batch
            chain_id = torch.ones_like(batch.batch)

            pretrain_embedding = self.pretrain_gearnet_model(seq_strs, node_index, node2graph, chain_id, batch.pos)
            return pretrain_embedding




