from ssl_model.torchgeo_extractor import TorchGeoSSLExtractor
import numpy as np
import yaml
import torch

def main():
    #load configuration from YAML file
    with open('configs/mvp.yaml',"r") as f:
        cfg=yaml.safe_load(f)
        
    mode =cfg.get("run", {}).get("mode","full")
  

    out_dir=Path(cfg["results"]["output_dir"])
    #load patches
    dm=Sentinel2PatchDataModule(cfg)
    dm.setup()

    train_samples=dm.get_patches(split="train")
    test_samples=dm.get_patches(split="test")
    
    train_images=torch.stack([s["image"] for s in train_samples],dim=0)
    test_images=torch.stack([s["image"] for s in test_samples],dim=0)
    
    train_meta=[s['metadata'] for s in train_samples]
    test_meta=[s['metadata'] for s in test_samples]
    # ndvi baseline scores
    baseline_scores={}
    ndvi_stats={}
    if mode == "full" or "baseline_only":
        baseline_result=run_baselines(test_images, test_meta, cfg)
        baseline_scores=baseline_result['scores']
        ndvi_stats=baseline_result['ndvi_statistics']
    
    # ssl
    ssl_scores=np.array([])
    
    if mode == "full" or "ssl_only":
        extractor= TorchGeoSSLExtractor(cfg)
        extractor.fit(dm)

       ...

    band_list=cfg["data"]["bands"]
    red_idx=band(cfg["baseline"]["ndvi"]["red_band"])
    nir_idx=band(cfg["baseline"]["ndvi"]["nir_band"])
    ndvi_means=

   
    field_id = cfg["visualization"]["field_id"]
    
    # knn scores
    # visualization and proxy metrics
