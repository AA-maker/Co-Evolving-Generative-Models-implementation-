# Use K-means to cluster text embeddings into K discrete clusters for paper assumptions

import numpy as np
from sklearn.cluster import KMeans
from config import K, DATA_DIR
from pathlib import Path
EMBED_CACHE = DATA_DIR / "embeds.npz"
OUT_FILE = DATA_DIR / "text_clusters.npz"

def cluster_texts(random_state=42):
    d = np.load(EMBED_CACHE)
    text_embs = d["text_embs"] if "text_embs" in d.files else d["text_embs"] if "text_embs" in d.files else d["text_embs"]
    # compatibility file keys may be text_embs or text_embeds
    if "text_embs" not in d.files:
        # find the nearest key
        for key in d.files:
            if "text" in key:
                text_embs = d[key]
                break
    print("Clustering text embeddings shape:", text_embs.shape)
    km = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    ids = km.fit_predict(text_embs)
    np.savez_compressed(OUT_FILE, cluster_ids=ids)
    print("Saved clusters to", OUT_FILE)

if __name__ == "__main__":
    cluster_texts()
