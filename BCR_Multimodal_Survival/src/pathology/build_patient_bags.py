import os
import re
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


def decode_name(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def parse_case_id(filename: str) -> str:
    """
    Recover the WSI / case identifier from a patch filename.

    Examples
    --------
    0760_11_10208_1024_None.png      -> 0760_11
    CASE_100-1-A-1-H&E_100352_14848_None.png -> CASE_100-1-A-1-H&E

    Falls back gracefully if the filename does not match the expected patch pattern.
    """
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]

    # Common patch format:
    #   <case_id>_<x>_<y>_None
    # Remove trailing numeric coordinate tokens and optional "None".
    parts = stem.split("_")

    # Strip trailing tokens that look like patch metadata.
    while parts:
        last = parts[-1]
        if last.lower() == "none":
            parts.pop()
            continue
        if re.fullmatch(r"\d+", last):
            parts.pop()
            continue
        break

    # If we still have at least two tokens, keep the prefix.
    if len(parts) >= 2:
        return "_".join(parts)

    # Otherwise fallback to the stem.
    return stem


def build_patient_bags_from_h5(h5_path, patient_id_fn=parse_case_id):
    """
    Build patient-level pathology bags from a cancer-only H5.

    Expected H5 keys:
        - embeddings
        - filenames
        - patient_ids   (preferred, if present)

    Returns:
        dict: patient_id -> torch.Tensor[num_patches, emb_dim]
    """
    bags = {}

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())

        if "patient_ids" in keys:
            patient_ids = [decode_name(x) for x in f["patient_ids"][:]]
        else:
            name_key = None
            for candidate in ["filenames", "file_names", "image_path", "paths"]:
                if candidate in keys:
                    name_key = candidate
                    break

            if name_key is None:
                raise KeyError(f"No filename-like key found in H5. Available keys: {keys}")

            patient_ids = [patient_id_fn(decode_name(x)) for x in f[name_key][:]]

        feat_key = None
        for candidate in ["embeddings", "train_embeddings", "test_embeddings"]:
            if candidate in keys:
                feat_key = candidate
                break

        if feat_key is None:
            raise KeyError(f"No embeddings key found in H5. Available keys: {keys}")

        embeddings = f[feat_key][:]

        for pid, emb in zip(patient_ids, embeddings):
            bags.setdefault(pid, []).append(emb)

    for pid in bags:
        bags[pid] = torch.tensor(np.asarray(bags[pid])).float()

    return bags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5",
        type=str,
        required=True,
        help="Path to cancer-only embeddings H5 file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/pathology/patient_bags.pt",
        help="Output path for saving bags as a torch object",
    )
    args = parser.parse_args()

    bags = build_patient_bags_from_h5(args.h5)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bags, out_path)

    print("=" * 60)
    print(f"Built patient bags for {len(bags)} patients")
    print(f"Saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()