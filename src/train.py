# Implement the training loop for co-evolution of text and image distributions (algorithm 1 & 2)

import numpy as np
from tqdm import trange
from pathlib import Path
from config import DATA_DIR, OUTPUTS, CHECKPOINT_DIR, LOG_DIR
from config import N_SAMPLES, MT, NT, T_MACRO, ALPHA_INJECT, EPS_INJECT, N0_USER_INJECT
from config import CHECKPOINT_EVERY, PLOT_EVERY, RANDOM_SEED
from utils import ensure_dir, multivariate_logpdf
from diagnostics import text_diversity, image_diversity, fidelity, plot_scalar_curve
import joblib
import os

ensure_dir(CHECKPOINT_DIR)
ensure_dir(LOG_DIR)

EMBED_CACHE = DATA_DIR / "embeds.npz"
CLUSTERS = DATA_DIR / "text_clusters.npz"
INIT_STATE = DATA_DIR / "init_state.npz"

np.random.seed(RANDOM_SEED)

def load_data():
    d = np.load(EMBED_CACHE)
    # find keys
    image_embs = None
    for key in d.files:
        if "image" in key:
            image_embs = d[key]
            break
    cl = np.load(CLUSTERS)
    cluster_ids = cl["cluster_ids"]
    # optionally load original mu0 per class from initial state (for fidelity)
    init = np.load(INIT_STATE)
    return image_embs, cluster_ids, init

def sample_from_gaussian(mu, Sigma, n=1):
    # mu (d,), Sigma (d,d)
    return np.random.multivariate_normal(mu, Sigma, size=n)

def run_training():
    image_embs, cluster_ids, init = load_data()
    p = init["p"].astype(np.float64)
    mu = init["mu"].astype(np.float64)
    Sigma = init["Sigma"].astype(np.float64)
    K = p.shape[0]
    d = mu.shape[1]
    mu0 = mu.copy()
    # store logs
    text_divs = []
    avg_img_divs = []
    avg_fids = []
    steps = []

    for t in trange(int(T_MACRO), desc="macro"):
        # Corpus Injections (Algorithm 2)
        if np.random.rand() < ALPHA_INJECT:
            # create new token and take EPS_INJECT mass from existing tokens
            eps = EPS_INJECT
            # reduce existing mass proportionally
            p = (1.0 - eps) * p
            # add a new token at the end
            new_p = np.array([eps])
            # init new mu,sigma to global mean + small noise
            global_mean = np.mean(mu, axis=0)
            mu = np.vstack([mu, global_mean + np.random.randn(d) * 1e-3])
            Sigma = np.concatenate([Sigma, np.eye(d).reshape(1, d, d) * 0.1], axis=0)
            p = np.hstack([p, new_p])
            K = p.shape[0]

        # Text Updates (Algorithm 1 text step)
        pcurr = p.copy()
        for m in range(MT):
            # sample N texts according to pcurr
            sampled_texts = np.random.choice(K, size=N_SAMPLES, p=pcurr)
            # sample y_j ~ q(y|x_j)
            y_samples = np.zeros((N_SAMPLES, d))
            for j, xi in enumerate(sampled_texts):
                y_samples[j] = sample_from_gaussian(mu[xi], Sigma[xi], n=1)[0]
            # compute log posterior numerator: log p(x_i) + log N(y|mu_i,Sigma_i)
            log_num = np.zeros((N_SAMPLES, K))
            for i in range(K):
                log_prior = np.log(pcurr[i] + 1e-16)
                lp = multivariate_logpdf(y_samples, mu[i], Sigma[i])  # shape (N,)
                log_num[:, i] = log_prior + lp
            # normalize to get posterior p(x_i | y_j)
            # subtract max for stability
            maxl = np.max(log_num, axis=1, keepdims=True)
            exp_term = np.exp(log_num - maxl)
            posterior = exp_term / np.sum(exp_term, axis=1, keepdims=True)
            # average over samples -> new pcurr
            pcurr = posterior.mean(axis=0)
        p = pcurr.copy()
        # normalize in case of numeric drift
        p = p / p.sum()

        # Image Updates (Algorithm 1 image step)
        qmu = mu.copy()
        qSigma = Sigma.copy()
        for nstep in range(NT):
            # sample N texts ~ p
            sampled_texts = np.random.choice(K, size=N_SAMPLES, p=p)
            # for each sampled text draw from its Gaussian
            # also inject N0 user images from real dataset per class (Algorithm 3)
            perclass_samples = {i: [] for i in range(K)}
            for xi in sampled_texts:
                y = sample_from_gaussian(mu[xi], Sigma[xi], n=1)[0]
                perclass_samples[xi].append(y)
            # user injection: sample real images assigned to that class from dataset
            # we use the precomputed image_embs & cluster ids
            # note: cluster_ids originally length = dataset size (pre-clustering)
            # for efficiency, precompute indices per class (lazy compute first iteration)
            if t == 0 and nstep == 0:
                # compute class->indices mapping once
                global class_indices_map
                class_indices_map = {}
                cl_arr = load_data()[1]
                for i in range(K):
                    class_indices_map[i] = np.where(cl_arr == i)[0]
            # now for each class, sample N0 user images (if enough exist)
            for i in range(K):
                idxs = class_indices_map.get(i, [])
                if len(idxs) > 0:
                    n_user = min(N0_USER_INJECT, len(idxs))
                    # choose with replacement
                    picks = np.random.choice(idxs, size=n_user, replace=True)
                    y_user = image_embs[picks]
                    perclass_samples[i].extend(list(y_user))
            # compute new mean & covariance per class using samples
            for i in range(K):
                Ys = perclass_samples[i]
                if len(Ys) >= 1:
                    Ys = np.stack(Ys)
                    mu_i = Ys.mean(axis=0)
                    if Ys.shape[0] >= 2:
                        Sigma_i = np.cov(Ys.T, bias=False)
                    else:
                        Sigma_i = qSigma[i]  # not enough samples to compute cov
                    # regularize
                    Sigma_i = Sigma_i + 1e-6 * np.eye(Sigma_i.shape[0])
                    qmu[i] = mu_i
                    qSigma[i] = Sigma_i
                # else keep old
        mu = qmu
        Sigma = qSigma

        # Diagnostics & Logging
        if (t % PLOT_EVERY) == 0:
            txt_div = text_diversity(p)
            img_divs = image_diversity(Sigma)  # per-class
            avg_img_div = float(np.mean(img_divs))
            fids = fidelity(mu, mu0)
            avg_fid = float(np.nanmean(fids))
            steps.append(int(t))
            text_divs.append(float(txt_div))
            avg_img_divs.append(avg_img_div)
            avg_fids.append(avg_fid)
        # checkpoint
        if (t % CHECKPOINT_EVERY) == 0:
            fname = CHECKPOINT_DIR / f"state_step_{t}.npz"
            np.savez_compressed(fname, p=p, mu=mu, Sigma=Sigma)
    # final save & plots
    np.savez_compressed(CHECKPOINT_DIR / "final_state.npz", p=p, mu=mu, Sigma=Sigma)
    # plots
    plot_scalar_curve(steps, text_divs, "Text Diversity", LOG_DIR / "text_diversity.png")
    plot_scalar_curve(steps, avg_img_divs, "Avg Image Diversity", LOG_DIR / "avg_image_div.png")
    plot_scalar_curve(steps, avg_fids, "Avg Fidelity", LOG_DIR / "avg_fidelity.png")
    print("Training finished. Plots saved to", LOG_DIR)

if __name__ == "__main__":
    run_training()
