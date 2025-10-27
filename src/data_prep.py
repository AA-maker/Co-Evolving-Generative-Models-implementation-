# Prepare data: compute and cache text and image embeddings

import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PIL import Image
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm
from config import DATA_DIR, CAPTIONS_FILE, IMAGES_DIR, USE_PCA, PCA_DIM, RANDOM_SEED
from utils import ensure_dir

EMBED_CACHE = DATA_DIR / "embeds.npz"

def load_captions(captions_file):
    # expects tab-separated lines: filename \t caption
    pairs = []
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) < 2: continue
            fname, caption = parts[0].strip(), "\t".join(parts[1:]).strip()
            pairs.append((fname, caption))
    return pairs

def compute_embeddings(model_name="all-MiniLM-L6-v2"):
    # compute both text and image embeddings
    ensure_dir(DATA_DIR)
    pairs = load_captions(CAPTIONS_FILE)
    print(f"Found {len(pairs)} captioned images.")
    text_model = SentenceTransformer(model_name)
    # image encoder (sentence-transformers)
    image_model = SentenceTransformer("clip-ViT-B-32")  
    image_embs = []
    text_embs = []
    fnames = []
    for fname, cap in tqdm(pairs):
        imgpath = IMAGES_DIR / fname
        if not imgpath.exists():
            continue
        try:
            img = Image.open(imgpath).convert("RGB")
        except:
            continue
        # image embedding
        emb_img = image_model.encode(img, convert_to_numpy=True)
        emb_text = text_model.encode(cap, convert_to_numpy=True)
        image_embs.append(emb_img)
        text_embs.append(emb_text)
        fnames.append(str(fname))
    image_embs = np.vstack(image_embs).astype(np.float32)
    text_embs = np.vstack(text_embs).astype(np.float32)
    print("Image embeddings shape:", image_embs.shape)
    print("Text embeddings shape:", text_embs.shape)
    # PCA on image embeddings
    pca = None
    if USE_PCA:
        print(f"Fitting PCA to reduce to {PCA_DIM} dims...")
        pca = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
        image_embs_reduced = pca.fit_transform(image_embs).astype(np.float32)
    else:
        image_embs_reduced = image_embs
    np.savez_compressed(EMBED_CACHE, image_embs=image_embs_reduced, text_embs=text_embs, fnames=np.array(fnames))
    if pca is not None:
        # save PCA parameters
        import joblib
        joblib.dump(pca, DATA_DIR / "pca.joblib")
    print("Saved embeds to", EMBED_CACHE)

if __name__ == "__main__":
    compute_embeddings()
