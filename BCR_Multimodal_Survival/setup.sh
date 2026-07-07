#!/usr/bin/env bash

# CUDA first
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio

if [ $? -ne 0 ]; then
    echo "CUDA install failed. Falling back to CPU..."
    python -m pip install torch torchvision torchaudio
fi

python -m pip install -r requirements.txt

mkdir -p \
  data/raw/wsi \
  data/raw/mri \
  data/raw/clinical \
  data/reference \
  data/interim/wsi/patches \
  data/interim/wsi/all_patches \
  data/interim/wsi/embeddings \
  data/interim/wsi/cancer_only \
  data/interim/mri/embeddings \
  data/processed/pathology \
  data/processed/fusion \
  outputs/predictions \
  outputs/heatmaps

# -----------------------------
# WSITools
# -----------------------------
if [ ! -d "WSITools" ]; then
    git clone https://github.com/smujiang/WSITools.git || echo "WARNING: Failed to clone WSITools"
fi

if [ -d "WSITools" ]; then
    (
        cd WSITools || exit
        python setup.py install || echo "WARNING: WSITools installation failed"
    )
else
    echo "WARNING: WSITools directory not found. Skipping."
fi

# -----------------------------
# UNI
# -----------------------------
if [ ! -d "UNI" ]; then
    git clone https://github.com/mahmoodlab/UNI.git || echo "WARNING: Failed to clone UNI"
fi

if [ -d "UNI" ]; then
    (
        cd UNI || exit
        python -m pip install -e . || echo "WARNING: UNI installation failed"
    )
else
    echo "WARNING: UNI directory not found. Skipping."
fi

echo "Setup Complete"