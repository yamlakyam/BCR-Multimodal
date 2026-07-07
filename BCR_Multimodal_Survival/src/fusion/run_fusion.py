from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_id(x: object) -> str:
    return str(x).replace(" ", "").replace("-", "").replace("_", "").upper().strip()


def load_torchlike_checkpoint(path: Path, device: torch.device):
    obj = torch.load(str(path), map_location=device)
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def load_pathology_vectors(path: Path) -> Dict[str, np.ndarray]:
    """
    Load pathology vectors saved as:
      - .pt / .pth dict via torch.save(dict)
      - .npy dict via np.save(dict, allow_pickle=True)
    Returns:
      patient_id -> 1D float32 vector
    """
    if path.suffix.lower() in {".pt", ".pth"}:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    elif path.suffix.lower() == ".npy":
        obj = np.load(str(path), allow_pickle=True).item()
    else:
        raise ValueError(f"Unsupported pathology vector file: {path}")

    if not isinstance(obj, dict):
        raise ValueError(f"Expected a dict in pathology file: {path}")

    out = {}
    for k, v in obj.items():
        out[norm_id(k)] = np.asarray(v, dtype=np.float32).reshape(-1)
    return out


def load_mri_vectors(mri_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load MRI vectors from .npy files in a directory.
    Supports common names like:
      <pid>_case_mean_feature.npy
      <pid>_patient_feature.npy
      <pid>.npy
    """
    out = {}
    for f in sorted(mri_dir.glob("*.npy")):
        name = f.name
        if name.endswith("_case_mean_feature.npy"):
            pid = name[: -len("_case_mean_feature.npy")]
        elif name.endswith("_patient_feature.npy"):
            pid = name[: -len("_patient_feature.npy")]
        else:
            pid = f.stem
        out[norm_id(pid)] = np.load(str(f)).astype(np.float32).reshape(-1)
    return out


def infer_patient_id_from_case_id(case_id: str, mri_ids: set[str]) -> str:
    """
    Map WSI case IDs to MRI patient IDs.

    Examples:
      0760_11 -> 0760
      0760_51 -> 0760
    If an exact match exists, keep it. Otherwise try progressively shorter prefixes.
    """
    cid = norm_id(case_id)

    if cid in mri_ids:
        return cid

    raw = str(case_id).strip()
    parts = raw.split("_")

    for i in range(len(parts), 0, -1):
        cand = norm_id("_".join(parts[:i]))
        if cand in mri_ids:
            return cand

    # final fallback: first token
    return norm_id(parts[0]) if parts else cid


def aggregate_pathology_to_patient(
    path_vecs_case: Dict[str, np.ndarray],
    mri_ids: set[str],
) -> Dict[str, np.ndarray]:
    """
    Collapse case-level pathology vectors to patient-level by averaging all
    case vectors that map to the same MRI patient ID.
    """
    buckets: Dict[str, List[np.ndarray]] = {}

    for case_id, vec in path_vecs_case.items():
        pid = infer_patient_id_from_case_id(case_id, mri_ids)
        buckets.setdefault(pid, []).append(vec)

    patient_vecs: Dict[str, np.ndarray] = {}
    for pid, vecs in buckets.items():
        patient_vecs[pid] = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)

    return patient_vecs


class PrognosticBridgeFusion(nn.Module):
    def __init__(self, path_dim=1536, rad_dim=256, latent_dim=32):
        super().__init__()
        self.path_proj = nn.Linear(path_dim, latent_dim)
        self.rad_proj = nn.Linear(rad_dim, latent_dim)

        self.norm_p = nn.LayerNorm(latent_dim)
        self.norm_r = nn.LayerNorm(latent_dim)

        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

        self.risk_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, h_p, h_r):
        z_p_raw = self.norm_p(F.relu(self.path_proj(h_p)))
        z_r_raw = self.norm_r(F.relu(self.rad_proj(h_r)))

        g = self.gate(torch.cat([z_p_raw, z_r_raw], dim=1))
        z_fused = g * z_p_raw + (1 - g) * z_r_raw
        risk = self.risk_head(z_fused)

        z_p = F.normalize(z_p_raw, p=2, dim=1)
        z_r = F.normalize(z_r_raw, p=2, dim=1)
        return risk, z_fused, z_p, z_r


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pathology_vectors",
        type=str,
        default="data/processed/pathology/pathology_vectors.pt",
        help="Path to pathology vectors dict (.pt or .npy)",
    )
    parser.add_argument(
        "--mri_dir",
        type=str,
        default="data/interim/mri/embeddings",
        help="Directory containing MRI .npy feature files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/fusion/best_fusion_chimera_surv.pth",
        help="Fusion checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/fusion",
        help="Output directory",
    )
    parser.add_argument("--path_dim", type=int, default=1536)
    parser.add_argument("--rad_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    pathology_path = root / args.pathology_vectors
    mri_dir = root / args.mri_dir
    checkpoint_path = root / args.checkpoint
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading pathology vectors...")
    path_vecs_case = load_pathology_vectors(pathology_path)

    print("Loading MRI vectors...")
    mri_vecs = load_mri_vectors(mri_dir)

    if len(path_vecs_case) == 0:
        raise RuntimeError(f"No pathology vectors found in {pathology_path}")
    if len(mri_vecs) == 0:
        raise RuntimeError(f"No MRI vectors found in {mri_dir}")

    mri_ids = set(mri_vecs.keys())
    path_vecs_patient = aggregate_pathology_to_patient(path_vecs_case, mri_ids)

    common_ids = sorted(set(path_vecs_patient) & set(mri_vecs))
    if len(common_ids) == 0:
        print("Pathology IDs:", sorted(path_vecs_patient.keys())[:20])
        print("MRI IDs:", sorted(mri_vecs.keys())[:20])
        raise RuntimeError("No matched patient IDs found across pathology and MRI.")

    print(f"Matched patients: {len(common_ids)}")

    for pid in common_ids[:5]:
        if path_vecs_patient[pid].shape[0] != args.path_dim:
            raise ValueError(
                f"{pid}: pathology dim {path_vecs_patient[pid].shape[0]} != expected {args.path_dim}"
            )
        if mri_vecs[pid].shape[0] != args.rad_dim:
            raise ValueError(
                f"{pid}: MRI dim {mri_vecs[pid].shape[0]} != expected {args.rad_dim}"
            )

    x_p = torch.tensor(
        np.stack([path_vecs_patient[pid] for pid in common_ids]),
        dtype=torch.float32,
        device=device,
    )
    x_r = torch.tensor(
        np.stack([mri_vecs[pid] for pid in common_ids]),
        dtype=torch.float32,
        device=device,
    )

    model = PrognosticBridgeFusion(path_dim=args.path_dim, rad_dim=args.rad_dim).to(device)
    state = load_torchlike_checkpoint(checkpoint_path, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    risks, zfused, zp, zr = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(common_ids), args.batch_size):
            xb_p = x_p[i : i + args.batch_size]
            xb_r = x_r[i : i + args.batch_size]
            risk, z_fused, z_p, z_r = model(xb_p, xb_r)
            risks.append(risk.detach().cpu().numpy())
            zfused.append(z_fused.detach().cpu().numpy())
            zp.append(z_p.detach().cpu().numpy())
            zr.append(z_r.detach().cpu().numpy())

    risks = np.concatenate(risks, axis=0).reshape(-1)
    zfused = np.concatenate(zfused, axis=0)
    zp = np.concatenate(zp, axis=0)
    zr = np.concatenate(zr, axis=0)

    df = pd.DataFrame(
        {
            "patient_id": common_ids,
            "risk": risks,
        }
    )

    df.to_csv(out_dir / "fusion_predictions.csv", index=False)
    np.save(out_dir / "fusion_z_fused.npy", zfused)
    np.save(out_dir / "fusion_z_path.npy", zp)
    np.save(out_dir / "fusion_z_mri.npy", zr)

    torch.save(path_vecs_patient, out_dir / "aggregated_pathology_vectors.pt")

    print(f"Saved predictions to: {out_dir / 'fusion_predictions.csv'}")
    print(f"Saved fused latent features to: {out_dir / 'fusion_z_fused.npy'}")
    print(f"Saved aggregated pathology vectors to: {out_dir / 'aggregated_pathology_vectors.pt'}")


if __name__ == "__main__":
    main()