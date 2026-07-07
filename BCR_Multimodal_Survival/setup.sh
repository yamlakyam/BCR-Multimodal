#!/bin/bash
set -e

pip install -r requirements.txt

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