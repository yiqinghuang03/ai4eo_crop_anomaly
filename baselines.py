# NDVI-based baseline
# limitation: NDVI is a vegetation index, can miss non_vegetation anomalies

import numpy as np

def ndvi(images,red_idx,nir_idx):
    red=images[:,red_idx,:,:]
    nir=images[:,nir_idx,:,:]
    return (nir-red)/(nir+red+1e-8)


def ndvi_means(images,red_idx,nir_idx):
    ndvi=ndvi(images,red_idx,nir_idx)
    return ndvi.mean(axis=(1,2))


def ndvi_anomaly_scores(ndvi_means):
    if len(ndvi_means)==0:
        return np.array([])
    samples_mean=ndvi_means.mean()
    return np.abs(ndvi_means-samples_mean)



def baselines(images):
    red_idx=2
    nir_idx=3
    ndvi_mean=ndvi_means(images,red_idx,nir_idx)
    scores=ndvi_anomaly_scores(ndvi_mean)
    return {"scores":{"ndvi":scores}}  

