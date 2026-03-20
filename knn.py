# kNN: k=5 is picked.

import torch
def knn_scores(test_features,ref_features,k=5):
    test_features=torch.nn.functional.normalize(test_features)
    ref_features=torch.nn.functional.normalize(ref_features)
    distance=torch.cdist(test_features,ref_features)
    k=min(k,ref_features.shape[0])
    nearest=torch.topk(distance,k,largest=False).values
    return nearest.mean()