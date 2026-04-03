import csv
import os
import numpy as np
import torch
import yaml
from knn import knn_scores
from torchgeo_extractor import TorchGeoSSLExtractor
from plots import fig_anomaly_map, fig_score_hist

def load_patches(Sentinel_2_dir):
    files=sorted(os.path.join(Sentinel_2_dir, f))
    patches=[]
    for f in files:
        x=np.load(f)
        patches.append(x)

    patches=np.stack(patches, axis=0)  
    return torch.tensor(patches, dtype=torch.float32)

def time_series(root_dir):
    field_ids=[]
    series_list=[]
    patches=[]
    for f in files:
            x=np.load(f) 
            patches.append(x)

      series=np.stack(patches, axis=0) 
      series=torch.tensor(series, dtype=torch.float32)

      field_ids.append(field_id)
      series_list.append(series)

    return field_ids, series_list


def baseline_score(series):
    arr=series.numpy()
    last=arr[:-1]     
    current=arr[-1]           

    mean_last=np.mean(last,axis=0) 
    diff=np.abs(current-mean_last)    
    score=float(np.mean(diff))
    return score


def extract_embeddings(extractor, series_list,batch_size):
    embeddings=[]
    for series in series_list:
        z_t=extract_embeddings(series, batch_size)
        z_f=z_t.mean(dim=0)
        embeddings.append(z_field)
    z_all=torch.stack(embeddings,dim=0) 
    return z_all


def flatten(series_list):
    images=[]

    for series in series_list:
        images.append(series)
    return torch.cat(all_images, dim=0)


def save_metrics(path, baseline_scores, ssl_scores):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow(["model", "metric", "value"])
        w.writerow(["temporal_baseline", "mean_score", float(np.mean(baseline_scores))])
        w.writerow(["temporal_baseline", "std_score", float(np.std(baseline_scores))])
        w.writerow(["ssl_knn_temporal", "mean_score", float(np.mean(ssl_scores))])
        w.writerow(["ssl_knn_temporal", "std_score", float(np.std(ssl_scores))])


def main():
    with open("data.yaml") as f:
        cfg=yaml.safe_load(f)
    train_dir=cfg["train_dir"]
    test_dir=cfg["test_dir"]
    batch_size=cfg["batch_size"]
    k=cfg["k"]
    train_field_ids, train_series_list=load_field_time_series(train_dir)
    test_field_ids, test_series_list=load_field_time_series(test_dir)
    baseline_scores=[]
    for series in test_series_list:
        s=baseline_score(series)
        baseline_score.append(s)
    train_tensor=flatten(train_series_list)
    extractor=TorchGeoSSLExtractor(cfg)
    extractor.fit(train_tensor)
    z_train=extract_embeddings(extractor, train_series_list, batch_size)
    z_test=extract_embeddings(extractor, test_series_list, batch_size)
    ssl_scores=knn_scores(z_test, z_train, k=5).numpy()
    save_metrics("results/tables/metrics.csv", baseline_scores, ssl_scores)
    fig_score_hist(baseline_scores, ssl_scores, "results/figures/score_hist.png")
    fig_anomaly_map(test_field_ids, baseline_scores, ssl_scores, "results/figures/anomaly_map.png “)
if __name__ == "__main__":
    main()
