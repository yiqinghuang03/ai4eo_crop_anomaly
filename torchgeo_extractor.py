from pytorch_lightning import Trainer
import torch
from torchgeo.trainers import SimCLRTask

class TorchGeoSSLExtractor:

    def __init__(self,cfg):
        self.cfg=cfg
        self.task = SimCLRTask(
            model=cfg["backbone"],
            in_channels=cfg["in_channels"],
            temperature=cfg["temperature"],
            output_dim=cfg["output_dim"],
            hidden_dim=cfg["hidden_dim"],
            layers=cfg["layers"],
            size=cfg["crop_size"]
        )

    def fit(self,datamodule):
        ssl_cfg=self.cfg
        train_cfg=self.cfg.get("train", {"enabled": False,"epochs":cfg.get("epochs", 5)})
        if not train_cfg.get("enabled", True):
            return
        trainer=Trainer(
            max_epochs=train_cfg["epochs"],
            accelerator=train_cfg.get("accelerator","auto"),
            logger=False,
            enable_checkpointing=False
        )
        trainer.fit(self.task,datamodule=datamodule)

    def extract_embeddings(self,images,batch_size=64):
        self.task.eval()
        device=next(self.task.parameters()).device
        embs=[]
        with torch.no_grad():
            for i in range(0,images.shape[0],batch_size):
                x=images[i:i+batch_size].to(device)
                out=self.task(x)
                z=out[0]
                z=torch.nn.functional.normalize(z)
                embs.append(z.cpu())
        if not embs:
            dim = int(self.cfg["ssl"].get("output_dim", 1))
            return torch.empty((0, dim))
        return torch.cat(embs,dim=0)
