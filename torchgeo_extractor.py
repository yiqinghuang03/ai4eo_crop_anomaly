from pytorch_lightning import Trainer
import torch
from torchgeo.trainers import SimCLRTask

class TorchGeoSSLExtractor:

    def __init__(self,cfg):
        self.cfg=cfg
        ssl_cfg=self.cfg["ssl"]
        self.task = SimCLRTask(
            model=ssl_cfg["backbone"],
            in_channels=ssl_cfg["in_channels"],
            temperature=ssl_cfg["temperature"],
            output_dim=ssl_cfg["output_dim"],
            hidden_dim=ssl_cfg["hidden_dim"],
            layers=ssl_cfg["layers"],
            size=ssl_cfg["crop_size"]
        )

    def fit(self,datamodule):
        ssl_cfg=self.cfg["ssl"]
        train_cfg=ssl_cfg.get("train", {"enabled": False, "epochs": ssl_cfg.get("epochs", 5)})
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
                z=torch.nn.functional.normalize(z, p=2, dim=1)
                embs.append(z.cpu())
        if not embs:
            dim = int(self.cfg["ssl"].get("output_dim", 1))
            return torch.empty((0, dim))
        return torch.cat(embs,dim=0)
