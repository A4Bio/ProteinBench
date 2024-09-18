# import esm
import torch.nn as nn
import torch
from esm.inverse_folding.util import CoordBatchConverter
import esm
class PretrainESMIF_Model(nn.Module):
    def __init__(self,args):
        super(PretrainESMIF_Model, self).__init__()
        #  /root/.cache/torch/hub/checkpoints
        if args.is_colab == False:
            model_data = torch.load("/gaozhangyang/model_zoom/transformers/esm_if/esm_if1_gvp4_t16_142M_UR50.pt")
        else:
            model_data = torch.load("main/results/esm_if1_gvp4_t16_142M_UR50.pt")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_core("esm_if1_gvp4_t16_142M_UR50", model_data, None)
    
    def forward(self, coords_list):
        self.model.eval()
        batch_converter = CoordBatchConverter(self.model.decoder.dictionary)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coord, None, None) for coord in coords_list], device=coords_list[0].device)
        )
        with torch.no_grad():
            encoder_out = self.model.encoder(batch_coords, padding_mask, confidence)
            
        feat = encoder_out['encoder_out'][0].permute(1,0,2)[:,1:-1] # 2,1046-2,512
        attention_mask = encoder_out['encoder_padding_mask'][0][:,1:-1]==False # 2,1046-2
        
        return {"feat":feat}

if __name__ == '__main__':
    model = PretrainESMIF_Model(0.1)
    coords1 = torch.rand(1044,3,3)#N, CA, C
    coords2 = torch.rand(500,3,3)
    model([coords1, coords2])
    print()