# Provides utility functions for directory management and multivariate normal log PDF computation

import numpy as np
import os, json
from pathlib import Path
from scipy.linalg import cho_solve, cho_factor
import math

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def multivariate_logpdf(x, mu, cov):
    """
    Compute log N(x | mu, cov) for each sample x.
    x: (N, d) or (d,)
    mu: (d,)
    cov: (d, d)
    Returns: (N,) array of log probabilities
    """
    x = np.atleast_2d(x)
    d = mu.shape[0]
    cov = cov + 1e-8 * np.eye(d)

    try:
        c, lower = cho_factor(cov, check_finite=False)
        diff = x - mu.reshape(1, -1)
        # shape (d, N)
        sol = cho_solve((c, lower), diff.T, check_finite=False)  
        # Mahalanobis term per sample
        quad = np.sum(diff * sol.T, axis=1)
        logdet = 2.0 * np.sum(np.log(np.diag(c)))
        const = -0.5 * d * np.log(2 * math.pi)
        return const - 0.5 * (logdet + quad)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)
        diff = x - mu.reshape(1, -1)
        quad = np.einsum('ij,ij->i', diff, diff @ inv)
        sign, logdet = np.linalg.slogdet(cov)
        const = -0.5 * d * np.log(2 * math.pi)
        return const - 0.5 * (logdet + quad)

