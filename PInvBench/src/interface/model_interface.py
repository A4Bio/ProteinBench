import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
import torch.optim.lr_scheduler as lrs
import inspect

class MInterface_base(pl.LightningModule):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)
        
    def forward(self, input):
        pass
    
    
    def training_step(self, batch, batch_idx, **kwargs):
        pass


    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
    
    def get_schedular(self, optimizer, lr_scheduler='onecycle'):
        if lr_scheduler == 'step':
            scheduler = lrs.StepLR(optimizer,
                                    step_size=self.hparams.lr_decay_steps,
                                    gamma=self.hparams.lr_decay_rate)
        elif lr_scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(optimizer,
                                                T_max=self.hparams.lr_decay_steps,
                                                eta_min=self.hparams.lr_decay_min_lr)
        elif lr_scheduler == 'onecycle':
            scheduler = lrs.OneCycleLR(optimizer, max_lr=self.hparams.lr, steps_per_epoch=self.hparams.steps_per_epoch, epochs=self.hparams.epoch, three_phase=False)
        else:
            raise ValueError('Invalid lr_scheduler type!')

        return scheduler

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
    
        optimizer_g = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-8)

        schecular_g = self.get_schedular(optimizer_g, self.hparams.lr_scheduler)

        return [optimizer_g], [{"scheduler": schecular_g, "interval": "step"}]
        
    def lr_scheduler_step(self, *args, **kwargs):
        scheduler = self.lr_schedulers()
        scheduler.step()
        
    
    def configure_devices(self):
        self.device = torch.device(self.hparams.device)

    def configure_loss(self):
        self.loss_function = nn.CrossEntropyLoss(reduction='none')
        
    def load_model(self):
        self.model = None

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
