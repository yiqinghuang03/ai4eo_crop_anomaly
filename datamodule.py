
import kornia.augmentation as K
import torch
import yaml
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import Sentinel2,stack_samples
from torchgeo.samplers import GridGeoSampler,RandomGeoSampler


class Sentinel2DataModule(LightningDataModule):
    def __init__(self,config_path="data.yaml"):
        super().__init__()
        cfg=yaml.safe_load(open(config_path))
        self.root=cfg["root"]
        self.batch_size=cfg["batch_size"]
        self.num_workers=cfg["num_workers"]
        self.patch_size=cfg["patch_size"]
        self.step=cfg["step"]
        self.test_stride=self.patch_size
        self.bands=cfg["bands"]
        
        
        self.mean=torch.tensor([0.10]*13)
        self.std=torch.tensor([0.05]*13)
        self.train_aug=K.AugmentationSequential(
            K.Normalize(self.mean, self.std),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip()
        )
        self.test_aug=K.AugmentationSequential(K.Normalize(self.mean, self.std))

    def setup(self,stage):
        self.dataset=Sentinel2(paths=self.root,bands=self.bands)

        self.train_sampler=RandomGeoSampler(
            self.dataset,
            size=self.patch_size,
            length=self.step
        )
        self.test_sampler=GridGeoSampler(
            self.dataset,
            size=self.patch_size,
            stride=self.test_stride
        )

    def stack_train(self,batch):
        sample=stack_samples(batch)
        x=sample["image"].float()
        sample["image"]=self.train_aug(x)
        return sample
    
    def stack_test(self,batch):
        sample=stack_samples(batch)
        x=sample["image"].float()
        sample["image"]=self.test_aug(x)
        return sample
        
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.stack_train,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            sampler=self.test_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.stack_test,
        )

