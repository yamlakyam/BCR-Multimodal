#!/bin/bash
set -e

pip install -r requirements.txt

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
    git clone https://github.com/smujiang/WSITools.git
fi

cd WSITools
python setup.py install
cd ..

# -----------------------------
# UNI
# -----------------------------
if [ ! -d "UNI" ]; then
    git clone https://github.com/mahmoodlab/UNI.git
fi

cd UNI
pip install -e .
cd ..

echo "✅ Setup Complete"