import sys; sys.path.append('/huyuqi/xmyu/DiffSDS')
import inspect
import torch
from src.tools.utils import cuda
import torch.nn as nn
import os
from torcheval.metrics.text import Perplexity
from src.interface.model_interface import MInterface_base
import math
import torch.nn.functional as F
from omegaconf import OmegaConf

class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.cross_entropy = nn.NLLLoss(reduction='none')
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)

    def forward(self, batch, mode='train', temperature=1.0):
        if self.hparams.augment_eps>0:
            batch['X'] = batch['X'] + self.hparams.augment_eps * torch.randn_like(batch['X'])

        batch = self.model._get_features(batch)
        results = self.model(batch)
        log_probs, mask = results['log_probs'], batch['mask']
        if len(log_probs.shape) == 3:
            loss = self.cross_entropy(log_probs.permute(0,2,1), batch['S'])
            loss = (loss*mask).sum()/(mask.sum())
        elif len(log_probs.shape) == 2:
            if self.hparams.model_name == 'GVP':
                loss = self.cross_entropy(log_probs, batch.seq)
            else:
                loss = self.cross_entropy(log_probs, batch['S'])
            
            if self.hparams.model_name == 'AlphaDesign':
                loss += self.cross_entropy(results['log_probs0'], batch['S'])
            loss = (loss*mask).sum()/(mask.sum())
        
        cmp = log_probs.argmax(dim=-1)==batch['S']
        recovery = (cmp*mask).sum()/(mask.sum())
        return loss, recovery


    def temperature_schedular(self, batch_idx):
        total_steps = self.hparams.steps_per_epoch*self.hparams.epoch
        
        initial_lr = 1.0
        circle_steps = total_steps//100
        x = batch_idx / total_steps
        threshold = 0.48
        if x<threshold:
            linear_decay = 1 - 2*x
        else:
            K = 1 - 2*threshold
            linear_decay = K - K*(x-threshold)/(1-threshold)
        
        new_lr = (1+math.cos(batch_idx/circle_steps*math.pi))/2*linear_decay*initial_lr

        return new_lr
    
    #https://lightning.ai/docs/pytorch/1.9.0/notebooks/lightning_examples/basic-gan.html
    def training_step(self, batch, batch_idx, **kwargs):
        loss, recovery = self(batch)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    


    def validation_step(self, batch, batch_idx):
        loss, recovery = self(batch)
        self.log_dict({"val_loss":loss,
                       "recovery": recovery})
        
        return self.log_dict

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_loss(self):
        def loss_function(pred_angle, angles, pred_seq, seqs, seq_loss_mask, angle_loss_mask):
            angle_loss = self.MSE(torch.cat([angles[...,:1],torch.sin(angles[...,1:3]), torch.cos(angles[...,1:3])],dim=-1),
            torch.cat([pred_angle[...,:1],torch.sin(pred_angle[...,1:3]), torch.cos(pred_angle[...,1:3])],dim=-1))
            
            angle_loss = angle_loss[angle_loss_mask].sum(dim=-1).mean()
            logits = pred_seq.permute(0,2,1)
            seq_loss = self.cross_entropy(logits, seqs)
            seq_loss = seq_loss[seq_loss_mask].mean()

            metric=Perplexity()
            metric.update(pred_seq[seq_loss_mask][None,...].cpu(), seqs[seq_loss_mask][None,...].cpu())
            perp = metric.compute()
            
            return {"angle_loss": angle_loss, "seq_loss": seq_loss, "perp":perp}

        self.loss_function = loss_function
        
    def load_model(self):
        params = OmegaConf.load(f'./src/models/configs/{self.hparams.model_name}.yaml')
        params.update(self.hparams)

        if self.hparams.model_name == 'GraphTrans':
            from src.models.graphtrans_model import GraphTrans_Model
            self.model = GraphTrans_Model(params)
        
        if self.hparams.model_name == 'StructGNN':
            from src.models.structgnn_model import StructGNN_Model
            self.model = StructGNN_Model(params)
            
        if self.hparams.model_name == 'GVP':
            from src.models.gvp_model import GVP_Model
            self.model = GVP_Model(params)

        if self.hparams.model_name == 'GCA':
            from src.models.gca_model import GCA_Model
            self.model = GCA_Model(params)

        if self.hparams.model_name == 'AlphaDesign':
            from src.models.alphadesign_model import AlphaDesign_Model
            self.model = AlphaDesign_Model(params)

        if self.hparams.model_name == 'ProteinMPNN':
            from src.models.proteinmpnn_model import ProteinMPNN_Model
            self.model = ProteinMPNN_Model(params)

        if self.hparams.model_name == 'ESMIF':
            pass

        if self.hparams.model_name == 'PiFold':
            from src.models.pifold_model import PiFold_Model
            self.model = PiFold_Model(params)

        # if self.hparams.model_name == 'KWDesign':
        #     from src.models.kwdesign_model import Design_Model
        #     self.model = Design_Model(params)
        
        if self.hparams.model_name == 'E3PiFold':
            from src.models.E3PiFold_model import E3PiFold
            self.model = E3PiFold(params)
        
        if self.hparams.model_name == 'UniIF':
            from src.models.uniif_model import UniIF_Model
            self.model = UniIF_Model(params)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)