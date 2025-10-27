# Configuration file for co-evolutionary diffusion model

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS = ROOT / "outputs"
CHECKPOINT_DIR = OUTPUTS / "checkpoints"
LOG_DIR = OUTPUTS / "logs"

# Data selection
CAPTIONS_FILE = DATA_DIR / "captions.txt"  # format: image_filename \t caption
IMAGES_DIR = DATA_DIR / "images"           # images here

# Embedding / latent choices
USE_PCA = True
PCA_DIM = 32            # latent dim d
K = 10                  # number of discrete text tokens
RANDOM_SEED = 42

# Training hyperparams
N_SAMPLES = 500         # N in text sampling / image sampling
MT = 1                  # text updates per macro step
NT = 2                  # image updates per macro step
T_MACRO = 2000          # number of macro steps (reduce to 200 for debugging)
ALPHA_INJECT = 0.05     # prob of corpus injection per macro step
EPS_INJECT = 0.02       # mass for new token on injection
N0_USER_INJECT = 10     # number of user images to inject per class update
CHECKPOINT_EVERY = 200
PLOT_EVERY = 100
