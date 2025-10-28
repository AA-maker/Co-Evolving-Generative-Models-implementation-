# Co-Evolving Generative Models (implementation)

This repository implements the co-evolving text-image Gaussian system described in Gao & Li (arXiv:2503.08117v1), including corpus injection and user-content injection stabilization.

## Quick start
1. Prepare `data/`:
    Images: [text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
    Captions: [text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)
   - Put images under `data/images/`.
   - Create `data/captions.txt` as tab-separated lines `image_filename<TAB>caption`.
   - Example: `1000268201_693b08cb0e.jpg\tA child is playing with a dog.`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run `./run.sh` (or run steps manually).

## Files of interest
- `src/train.py` — main loop implementing Algorithms 1–3.
- `src/data_prep.py` — computes embeddings with `sentence-transformers` models.
- `outputs/` — stores checkpoints and plots.

## Notes and tips
- For quick debugging lower `T_MACRO` in `src/config.py` to 200.
- To reproduce toy experiments from the paper, set `K=5`, `PCA_DIM=2`, and generate synthetic Gaussians instead of reading data.
- The implementation uses PCA-reduced CLIP-like image embeddings, and clusters captions to discrete tokens, following the paper's discrete text + Gaussian image model mapping.
- To train without corpus text injections edit variables `ALPHA_INJECT`, `EPS_INJECT`, and `NO_USER_INJECT` in `src/config.py` to 0.  

