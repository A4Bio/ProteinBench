from pytorch_lightning.callbacks import Callback
import os
import shutil
from omegaconf import OmegaConf

class SetupCallback(Callback):
    def __init__(self,  now, logdir, ckptdir, cfgdir, config, argv_content=None):
        super().__init__()
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
    
        self.argv_content = argv_content

    # 在pretrain例程开始时调用。
    def on_fit_start(self, trainer, pl_module):
        # Create logdirs and save configs
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)

        print("Project config")
        print(OmegaConf.to_yaml(self.config))
        OmegaConf.save(self.config,
                        os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))
        
        with open(os.path.join(self.logdir, "argv_content.txt"), "w") as f:
            f.write(str(self.argv_content))

class BackupCodeCallback(Callback):
    def __init__(self, source_dir, backup_dir, ignore_patterns=None):
        super().__init__()
        self.source_dir = source_dir
        self.backup_dir = backup_dir
        self.ignore_patterns = ignore_patterns

    def on_train_start(self, trainer, pl_module):
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            if os.path.exists(self.backup_dir+'/code'):
                shutil.rmtree(self.backup_dir+'/code')
            shutil.copytree(self.source_dir, self.backup_dir+'/code', ignore=self.ignore_patterns)

            print(f"Code file backed up to {self.backup_dir}")
        except:
            print(f"Fail in copying file backed up to {self.backup_dir}")