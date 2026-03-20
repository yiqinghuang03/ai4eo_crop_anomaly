import matplotlib.pyplot as plt
import numpy as np


def fig_score_hist(scores_baseline,scores_ssl,save_path="results/fig_score_hist.png"):
    fig,axes=plt.subplots(1,2,figsize=(10,4))

    axes[0].hist(scores_baseline,bins=30)
    axes[0].set_title("baseline")
    axes[0].set_xlabel("anomaly score")

    axes[1].hist(scores_ssl,bins=30)
    axes[1].set_title("SSL")
    axes[1].set_xlabel("anomaly score")

    plt.savefig(save_path)


def fig_anomaly_map(field_ids,scores_baseline,scores_ssl,save_path="results/fig_anomaly_map.png"):

    n=64
    s_baseline=scores_baseline[:n]
    s_ssl=scores_ssl[:n]

    m=max(1,np.sqrt(n))
    s_baseline=s_baseline[:m*m].reshape(m,m)
    s_ssl=s_ssl[:m*m].reshape(m,m)

    fig, axes=plt.subplots(1,2,figsize=(10, 4))
    axes[0].imshow(s_baseline,cmap="hot")
    axes[0].set_title("baseline anomaly map")

    axes[1].imshow(s_ssl,cmap="hot")
    axes[1].set_title("SSL anomaly map")

    plt.savefig(save_path)