import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repo root to Python path so `src.*` imports work when running as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pathology.build_patient_bags import build_patient_bags_from_h5


class PathMILSurv(nn.Module):
    def __init__(self, input_dim=1536, L=512, D=128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, L),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, 1)
        )
        self.risk_head = nn.Linear(L, 1)

    def forward(self, x):
        h = self.feature_extractor(x)          # [N, L]
        A = self.attention(h)                  # [N, 1]
        A = torch.transpose(A, 1, 0)           # [1, N]
        A = F.softmax(A, dim=1)                # attention weights
        m = torch.mm(A, h)                     # [1, L]
        risk = self.risk_head(m)               # [1, 1]
        return risk, A, h


def load_mil_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        input_dim = ckpt.get("input_dim", 1536)
        L = ckpt.get("L", 512)
        D = ckpt.get("D", 128)
        model = PathMILSurv(input_dim=input_dim, L=L, D=D).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = PathMILSurv().to(device)
        model.load_state_dict(ckpt)

    model.eval()
    return model


def extract_representative_features(model, bags_dict, ids_list, device, K=10, key_fn=None):
    model.eval()
    rep_features = {}

    if key_fn is None:
        key_fn = lambda pid: pid

    with torch.no_grad():
        for pid in ids_list:
            bag_key = key_fn(pid)

            if bag_key not in bags_dict:
                continue

            bag = bags_dict[bag_key].to(device)
            _, A, _ = model(bag)
            A = A.squeeze()

            actual_k = min(K, A.size(0))
            weights, indices = torch.topk(A, actual_k)
            weights = weights / weights.sum()

            top_patches = bag[indices]
            weighted_rep = torch.mm(weights.unsqueeze(0), top_patches)

            rep_features[pid] = weighted_rep.squeeze().cpu()

    return rep_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5",
        type=str,
        required=True,
        help="Path to cancer-only embeddings H5",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to best_mil_chimera_surv.pth",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/pathology/pathology_vectors.pt",
        help="Output file for patient-level pathology vectors",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top attended patches to use",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    bags = build_patient_bags_from_h5(args.h5)
    print(f"Built {len(bags)} patient bags")

    model = load_mil_model(args.ckpt, device)

    patient_ids = list(bags.keys())
    rep_features = extract_representative_features(
        model=model,
        bags_dict=bags,
        ids_list=patient_ids,
        device=device,
        K=args.top_k,
        key_fn=lambda pid: pid,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rep_features, str(output_path))

    print("=" * 60)
    print(f"Extracted representative pathology vectors for {len(rep_features)} patients")
    print(f"Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()