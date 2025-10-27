#!/usr/bin/env bash
set -e
# Activate virtual environment if you use one
# e.g. source venv/bin/activate

# 1) Install dependencies (first time)
# pip install -r requirements.txt

# 2) Step 1: compute embeddings (may take a few minutes)
python3 src/data_prep.py

# 3) Step 2: cluster texts
python3 src/cluster_texts.py

# 4) Step 3: init state
python3 src/init_state.py

# 5) Step 4: run training
python3 src/train.py
