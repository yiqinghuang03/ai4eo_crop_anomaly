import csv
import os

import numpy as np
import torch
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from knn import knn_scores
from baselines import baselines
from torchgeo_extractor import TorchGeoSSLExtractor
from torchgeo.datamodules import EuroSATDataModule
from torchgeo.datasets import EuroSAT
from plots import fig_anomaly_map, fig_score_hist


def collect_images(loader):
    images=[]
    for batch in loader:
        x=batch["image"]
        images.append(x)
    return torch.cat(images, dim=0)


def main():
    with open("data.yaml",encoding="utf-8") as f:
        cfg=yaml.safe_load(f)

    k=cfg["k"]

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    dm=EuroSATDataModule(batch_size=cfg["batch_size"],num_workers=cfg["num_workers"],root="data/EuroSAT",download=True)
    dm.setup("fit")
    dm.setup("test")

    train_tensor = collect_images(dm.train_dataloader())
    test_tensor = collect_images(dm.test_dataloader())

    field_ids = [0] * test_tensor.shape[0]

    ndvi_scores=np.mean(test_tensor.numpy(),axis=(1,2,3))

    ex = TorchGeoSSLExtractor(cfg)
    ex.fit(dm)

    bs = f["batch_size"]
    z_train = ex.extract_embeddings(train_tensor,batch_size=bs)
    z_test = ex.extract_embeddings(test_tensor,batch_size=bs)

    ssl_scores =knn_scores(z_test, z_train, k=k).numpy()

    np.save("results/tables/baseline_scores.npy", ndvi_scores)
    np.save("results/tables/ssl_scores.npy", ssl_scores)

    with open("results/tables/metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "value"])
        w.writerow(["baseline_ndvi", "mean_score", float(np.mean(ndvi_scores))])
        w.writerow(["baseline_ndvi", "std_score", float(np.std(ndvi_scores))])
        w.writerow(["ssl_knn", "mean_score", float(np.mean(ssl_scores))])
        w.writerow(["ssl_knn", "std_score", float(np.std(ssl_scores))])

    fig_score_hist(ndvi_scores, ssl_scores, "results/figures/fig_score_hist.png")
    fig_anomaly_map(field_ids, ndvi_scores, ssl_scores, "results/figures/fig_anomaly_map.png")


if __name__ == "__main__":
    main()
