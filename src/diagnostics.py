# Tests image and text diversity and fidelity metrics

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import LOG_DIR
from utils import ensure_dir
ensure_dir(LOG_DIR)

def text_diversity(p):
    # H = 1 - sum p^2  (paper's diversity measure)
    return 1.0 - np.sum(p**2)

def image_diversity(Sigma):
    # simple scalar per-class diversity: trace of sqrtm of Sigma
    # approximate: use trace(Sigma)
    return np.trace(Sigma, axis1=1, axis2=2)  # returns array (K,)

def fidelity(mu, mu0):
    """
    Compute per-class fidelity distances.
    If new tokens were added (len(mu) > len(mu0)),
    compare overlapping ones and assign NaN to new tokens.
    """
    K0 = mu0.shape[0]
    K = mu.shape[0]
    d = mu.shape[1]
    if K == K0:
        return np.linalg.norm(mu - mu0, axis=1)
    elif K > K0:
        fids = np.empty(K)
        fids[:K0] = np.linalg.norm(mu[:K0] - mu0, axis=1)
        # newly injected tokens
        fids[K0:] = np.nan  
        return fids
    else:
        # unlikely (tokens removed)
        return np.linalg.norm(mu - mu0[:K], axis=1)


# Plot diagnostic curves
def plot_scalar_curve(xs, ys, title, outpath):
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel("macro step")
    plt.ylabel(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
