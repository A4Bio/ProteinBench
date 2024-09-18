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
        
        # if name == "LFQ":
        #     from step1_VQ.model_interface import MInterface
        #     pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMFVQ/LFQ_seg_linear/configs/10-17T15-46-37-project.yaml")
        #     pretrain_args.diffusion = False
        #     self.pretrain_model = MInterface(**pretrain_args)
        #     ckpt = torch.load('/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMFVQ/LFQ_seg_linear/checkpoints/best-epoch=14-val_loss=0.161.pth', map_location=torch.device('cpu'))
        #     state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
        #     self.pretrain_model.load_state_dict(state_dict, strict=False)
            


        if name == "softgroup-1":
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftGroup/softgroup-1/configs/12-16T14-57-28-project.yaml")
            pretrain_args.diffusion = False
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftGroup/softgroup-1/checkpoints/best-epoch=13-val_loss=0.111.pth', map_location=torch.device('cpu'))
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == "softgroup-2":
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoftGroup/softgroup-2/configs/10-24T12-51-57-project.yaml")
            pretrain_args.diffusion = False
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoftGroup/softgroup-2/checkpoints/best-epoch=14-val_loss=0.067.pth', map_location=torch.device('cpu'))
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == "softgroup-3":
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoftGroup/softgroup-3/configs/10-25T00-04-15-project.yaml")
            pretrain_args.diffusion = False
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoftGroup/softgroup-3/checkpoints/best-epoch=14-val_loss=0.063.pth', map_location=torch.device('cpu'))
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == "softgroup-4":
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoftGroup/softgroup_32_vectors/configs/10-19T01-03-55-project.yaml")
            pretrain_args.diffusion = False
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoftGroup/softgroup_32_vectors/checkpoints/best-epoch=14-val_loss=0.056.pth', map_location=torch.device('cpu'))
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict, strict=False)
        
        if name == "softgroup-5":
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoftGroup/softgroup-5-gzy/configs/10-27T17-15-56-project.yaml")
            pretrain_args.diffusion = False
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoftGroup/softgroup-5-gzy/checkpoints/best-epoch=14-val_loss=0.039.pth', map_location=torch.device('cpu'))
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == "softgroup-6":
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoftGroup/softgroup_128_group/configs/10-28T01-28-50-project.yaml")
            pretrain_args.diffusion = False
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoftGroup/softgroup_128_group/checkpoints/best-epoch=14-val_loss=0.011.pth', map_location=torch.device('cpu'))
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        


        if name == "softgroup_128_group":
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoftGroup/softgroup_128_group/configs/10-28T01-28-50-project.yaml")
            pretrain_args.diffusion = False
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoftGroup/softgroup_128_group/checkpoints/best-epoch=14-val_loss=0.011.pth', map_location=torch.device('cpu'))
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)

        if name == "diff-softgroup-1":
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/DiffESMSoftGroup/diff-softgroup-rm-dist/configs/12-17T14-19-21-project.yaml")
            pretrain_args.diffusion = True
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/DiffESMSoftGroup/diff-softgroup-rm-dist/checkpoints/best-epoch=12-val_loss=0.496.pth', map_location=torch.device('cpu'))
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == "diff-softgroup-4":
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/DiffESMSoftGroup/diff-softgroup-vq32/configs/12-19T01-54-15-project.yaml")
            pretrain_args.diffusion = True
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/DiffESMSoftGroup/diff-softgroup-vq32/checkpoints/best-epoch=13-val_loss=0.184.pth', map_location=torch.device('cpu'))
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == "diff-softgroup-5":
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/DiffESMSoftGroup/diff-softgroup-vq64/configs/12-19T01-57-07-project.yaml")
            pretrain_args.diffusion = True
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/DiffESMSoftGroup/diff-softgroup-vq64/checkpoints/best-epoch=13-val_loss=0.100.pth', map_location=torch.device('cpu'))
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == "diff-softgroup-6":
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/DiffESMSoftGroup/diff-softgroup-vq128/configs/12-19T10-47-37-project.yaml")
            pretrain_args.diffusion = True
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/DiffESMSoftGroup/diff-softgroup-vq128/checkpoints/best-epoch=13-val_loss=0.081.pth', map_location=torch.device('cpu'))
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'vanilla-1':
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMVQ/base/configs/10-18T01-15-37-project.yaml")
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMVQ/base/checkpoints/best-epoch=14-val_loss=0.314.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)

        if name == 'soft-1':
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoft/soft_rerun/configs/12-10T12-38-16-project.yaml")
            pretrain_args.diffusion=False
            pretrain_args.attn_type = 'raw'
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoft/soft_rerun/checkpoints/best-epoch=14-val_loss=0.018.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)

        if name == 'soft_64_vecs':
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoft/soft_vq_num64/configs/10-19T11-11-58-project.yaml")
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMSoft/soft_vq_num64/checkpoints/best-epoch=14-val_loss=8.768.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'LFQ':
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMFVQ/vanilla_L1loss/configs/10-24T01-36-37-project.yaml")
            pretrain_args.diffusion = False
            pretrain_args.attn_type = 'raw'
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/DiffSDS/Pretrain_lightning/results/ESMFVQ/vanilla_L1loss/checkpoints/best-epoch=14-val_loss=11.328.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict, strict=False)
        
        if name == 'SCQ-mlp3-vqdim32':
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-vq16-conditional-mlp3-vqdim32/configs/12-22T07-52-47-project.yaml")
            pretrain_args.diffusion = False
            pretrain_args.vq_dim, pretrain_args.condition_layer, pretrain_args.sphere = 32, 3, False

            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-vq16-conditional-mlp3-vqdim32/checkpoints/best-epoch=14-val_loss=0.376.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'SCQ-mlp3-vqdim32-sphere':
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-mlp3-vqdim32-sphere/configs/12-22T10-44-46-project.yaml")
            pretrain_args.diffusion = False

            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-mlp3-vqdim32-sphere/checkpoints/best-epoch=14-val_loss=0.454.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'SCQ-mlp6-vqdim32-sphere':
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-mlp6BN-vqdim32-sphere/configs/12-22T18-28-04-project.yaml")
            pretrain_args.diffusion = False
            pretrain_args.attn_type = 'raw'
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-mlp6BN-vqdim32-sphere/checkpoints/best-epoch=14-val_loss=0.148.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'SCQ-mlp2-vqdim32':
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-vq16-conditional-mlp2-vqdim32/configs/12-22T00-21-35-project.yaml")
            pretrain_args.diffusion = False
            pretrain_args.vq_dim, pretrain_args.condition_layer, pretrain_args.sphere = 32, 2, False
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-vq16-conditional-mlp2-vqdim32/checkpoints/best-epoch=14-val_loss=0.362.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'SCQ-mlp2-vqdim32-sphere':
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-vq16-conditional-sphere-vqdim32/configs/12-22T00-06-35-project.yaml")
            pretrain_args.diffusion = False
            pretrain_args.vq_dim, pretrain_args.condition_layer, pretrain_args.sphere = 32, 2, True
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-vq16-conditional-sphere-vqdim32/checkpoints/best-epoch=14-val_loss=0.338.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'SCQ-mlp2-vqdim16':
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-vq16-conditional/configs/12-21T13-13-11-project.yaml")
            pretrain_args.diffusion = False
            pretrain_args.vq_dim, pretrain_args.condition_layer, pretrain_args.sphere = 16, 2, False
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-vq16-conditional/checkpoints/best-epoch=14-val_loss=0.094.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'SCQ-mlp2-vqdim16-sphere':
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-vq16-conditional-sphere/configs/12-21T16-38-57-project.yaml")
            pretrain_args.diffusion = False
            pretrain_args.vq_dim, pretrain_args.condition_layer, pretrain_args.sphere = 16, 2, True
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-vq16-conditional-sphere/checkpoints/best-epoch=14-val_loss=1.080.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'SCQ-vq8-mlp6-vqdim16-sphere':
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-vq8-mlp6BN-vqdim32-sphere/configs/12-23T05-15-56-project.yaml")
            pretrain_args.diffusion = False
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-vq8-mlp6BN-vqdim32-sphere/checkpoints/best-epoch=14-val_loss=0.892.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)

        if name == 'SCQ-mlp9-vqdim32-sphere':
            from step1_VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-mlp9BN-vqdim32-sphere/configs/12-23T16-20-07-project.yaml")
            pretrain_args.diffusion = False

            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/VQProteinFormer/step1_VQ/results/ESMSoftBV/SoftBV-mlp9BN-vqdim32-sphere/checkpoints/best-epoch=14-val_loss=0.151.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        

        
        if name == 'AF2VQ':
            from step3_AF2VQ.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step3_AF2VQ/results/AF2VQ_softgroup16/configs/12-13T07-59-50-project.yaml")
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load("/huyuqi/xmyu/VQProteinFormer/step3_AF2VQ/results/AF2VQ_softgroup16/checkpoints/best-epoch=11-val_loss=0.812.pth")
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)

        if name == "ProGLM":
            self.vq_dim=480
            from step2_ProGLM.model.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Inpainting_representation/results/softgroup_bin_1127/version_4/hparams.yaml")
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/DiffSDS/Inpainting_representation/results/softgroup_bin_1127/checkpoints/best-epoch=08-valid_acc=0.804.ckpt', map_location=torch.device('cpu'))['state_dict']
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'ProGLM_softgroup_af2db':
            from step2_ProGLM.model.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/DiffSDS/Inpainting_representation/results/softgroup_bin_2/version_3/hparams.yaml")
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/DiffSDS/Inpainting_representation/results/softgroup_bin_2/checkpoints/best-epoch=13-valid_acc=0.863.ckpt', map_location=torch.device('cpu'))['state_dict']
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'ProGLM_SoftVQ_cath':
            from step2_ProGLM.model.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGLM_SoftVQ_epoch15_pad300/configs/12-25T01-20-35-project.yaml")
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGLM_SoftVQ_epoch15_pad300/checkpoints/best-epoch=27-valid_acc=0.001.pth')
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)


        if name == 'ProGLM_SoftCVQ_cath':
            from step2_ProGLM.model.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGLM_SoftCVQ_epoch15_pad300_BCE/configs/12-25T01-42-37-project.yaml")
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGLM_SoftCVQ_epoch15_pad300_BCE/checkpoints/best-epoch=14-valid_acc=0.614.pth')
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)

        if name == 'ProGLM_SoftCVQ_cath_inpaint':
            from step2_ProGLM.model.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGLM_SoftCVQ_epoch15_pad300_BCE_inpaint/configs/12-25T07-47-52-project.yaml")
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGLM_SoftCVQ_epoch15_pad300_BCE_inpaint/checkpoints/best-epoch=14-valid_acc=0.616.pth')
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'ProGLM_SoftCVQ_AF2DB':
            from step2_ProGLM.model.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGLM_SoftCVQ_epoch15_AF2DB/configs/12-25T13-01-12-project.yaml")
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGLM_SoftCVQ_epoch15_AF2DB/checkpoints/best-epoch=14-valid_acc=0.631.pth')
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)

        if name == 'ProGLM_SoftCVQ_ESM1B_CATH':
            from step2_ProGLM.model.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGLM_SoftCVQ_ESM1B_CATH_lr5e-5/configs/12-25T16-02-35-project.yaml")
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGLM_SoftCVQ_ESM1B_CATH_lr5e-5/checkpoints/best-epoch=14-valid_acc=0.616.pth')
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)

        if name == 'ProGLM_SoftCVQ_CATH':
            from step2_ProGLM.model.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGPT_SoftCVQ_CATH/configs/12-26T08-13-41-project.yaml")
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGPT_SoftCVQ_CATH/checkpoints/best-epoch=14-gpt_acc=0.758.pth')
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        if name == 'ProGLM_SoftCVQ_CATH_epoch10k':
            from step2_ProGLM.model.model_interface import MInterface
            pretrain_args = OmegaConf.load("/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGPT_SoftCVQ_CATH_epoch1000/configs/12-27T02-36-49-project.yaml")
            self.pretrain_model = MInterface(**pretrain_args)
            ckpt = torch.load('/huyuqi/xmyu/VQProteinFormer/step2_ProGLM/results/ProGPT_SoftCVQ_CATH_epoch10000_resume/checkpoints/best-epoch=1887-gpt_loss=0.181.pth')
            state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
            self.pretrain_model.load_state_dict(state_dict)
        
        
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




