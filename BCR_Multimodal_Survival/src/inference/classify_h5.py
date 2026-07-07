
import h5py
import numpy as np
import pandas as pd
import joblib
import argparse
import os
from tqdm import tqdm


def run_classification(h5_path, model_path, output_csv):
    # 1. Load the Model Checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from {model_path}...")
    checkpoint = joblib.load(model_path)
    clf = checkpoint["model"]

    # Use the saved index or default to 0
    pos_idx = checkpoint.get("pos_class_idx", 0)

    # 2. Open Embeddings with high-performance cache
    print(f"Opening embeddings: {h5_path}")

    with h5py.File(h5_path, "r", rdcc_nbytes=1024**3) as f:

        # Support different H5 formats
        feat_key = "train_embeddings" if "train_embeddings" in f else "embeddings"
        name_key = "file_names" if "file_names" in f else "filenames"

        feats = f[feat_key]
        raw_names = f[name_key][:]

        filenames = [
            fn.decode() if isinstance(fn, bytes) else fn
            for fn in raw_names
        ]

        total = len(filenames)
        print(f"Processing {total} patches...")

        all_probs = []
        all_preds = []

        batch_size = 25000

        for start in tqdm(range(0, total, batch_size), desc="Classifying"):

            end = min(start + batch_size, total)

            X_batch = feats[start:end]

            probs = clf.predict_proba(X_batch)[:, pos_idx]
            preds = clf.predict(X_batch)

            all_probs.extend(probs)
            all_preds.extend(preds)

    # 3. Save results
    results = pd.DataFrame({
        "filename": filenames,
        "prediction": all_preds,
        "probability_class0": all_probs,
    })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results.to_csv(output_csv, index=False)

    print(f"Done! Results saved to {output_csv}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--h5",
        type=str,
        required=True,
        help="Path to the patch embeddings H5 file",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/pathology/prostate_uni2_model.joblib",
        help="Path to the pathology classifier checkpoint",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions/patch_predictions.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    run_classification(
        args.h5,
        args.model,
        args.output,
    )
