# Initialize GMM parameters from clustered embeddings

import numpy as np
from pathlib import Path
from config import DATA_DIR, K
from utils import ensure_dir
EMBED_CACHE = DATA_DIR / "embeds.npz"
CLUSTERS = DATA_DIR / "text_clusters.npz"
OUT_STATE = DATA_DIR / "init_state.npz"

def init_state():
    d = np.load(EMBED_CACHE)
    image_embs = None
    if "image_embs" in d.files:
        image_embs = d["image_embs"]
    else:
        # try other keys
        for key in d.files:
            if "image" in key:
                image_embs = d[key]
                break
    clusters = np.load(CLUSTERS)["cluster_ids"]
    assert len(image_embs) == len(clusters)
    N, dim = image_embs.shape
    p = np.zeros(K, dtype=np.float64)
    mu = np.zeros((K, dim), dtype=np.float64)
    Sigma = np.zeros((K, dim, dim), dtype=np.float64)
    for i in range(K):
        idx = np.where(clusters == i)[0]
        if len(idx) == 0:
            # initialize small random gaussian
            p[i] = 1e-6
            mu[i] = np.random.randn(dim) * 1e-2
            Sigma[i] = np.eye(dim) * 0.1
        else:
            p[i] = len(idx) / len(clusters)
            imgs = image_embs[idx]
            mu[i] = imgs.mean(axis=0)
            if len(idx) > 1:
                Sigma[i] = np.cov(imgs.T, bias=False)
            else:
                Sigma[i] = np.eye(dim) * 0.01
            # regularize tiny covariances
            Sigma[i] += 1e-6 * np.eye(dim)
    # normalize p
    p = p / p.sum()
    np.savez_compressed(OUT_STATE, p=p, mu=mu, Sigma=Sigma)
    print("Saved initial state to", OUT_STATE)

if __name__ == "__main__":
    init_state()
