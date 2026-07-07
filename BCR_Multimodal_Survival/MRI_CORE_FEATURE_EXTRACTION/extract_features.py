"""
MRI-CORE Generic Feature Extraction

ROI logic:
    box > gland > tumor > whole image

Image channel logic:
    T2 + ADC + HBV  -> [T2, ADC, HBV]
    T2 + ADC only   -> [T2, ADC, T2]
    1 modality only -> [mod, mod, mod]

Masks are used only for cropping, NOT as image channels.
"""

import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm


# =========================
# Paths
# =========================
ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
DATA_DIR = REPO_ROOT / "data" / "raw" / "mri"
OUT_DIR = REPO_ROOT / "data" / "interim" / "mri" / "embeddings"
MRI_CORE_DIR = ROOT_DIR / "mri_foundation"
CHECKPOINT_PATH = ROOT_DIR / "weights" / "mri_foundation.pth"


OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MRI_CORE_DIR))

from models.sam import sam_model_registry
import cfg


# =========================
# Model
# =========================
args = cfg.parse_args()
args.num_cls = 1
args.image_size = 256

device = "cuda" if torch.cuda.is_available() else "cpu"

model = sam_model_registry["vit_b"](
    args,
    checkpoint=str(CHECKPOINT_PATH),
    num_classes=args.num_cls,
    image_size=args.image_size,
    pretrained_sam=True,
).to(device).eval()


# =========================
# Helpers
# =========================
def load_and_norm(path):
    vol = nib.load(str(path)).get_fdata().astype(np.float32)

    for z in range(vol.shape[2]):
        s = vol[..., z]
        minv, maxv = s.min(), s.max()

        if maxv > minv:
            vol[..., z] = (s - minv) / (maxv - minv)
        else:
            vol[..., z] = 0

    return vol


def find_case_files(case_dir):
    # nii_files = list(case_dir.glob("*.nii.gz"))
    # nii_files = list(case_dir.glob("*.nii")) + list(case_dir.glob("*.nii.gz"))
    nii_files = [
        f for f in (list(case_dir.glob("*.nii")) + list(case_dir.glob("*.nii.gz")))
        if not f.name.startswith("._")
    ]

    files = {
        "t2": None,
        "adc_reg": None,
        "adc": None,
        "hbv": None,
        "gland": None,
        "tumor": None,
        "box": None,
    }

    for f in nii_files:
        name = f.name.lower()

        if "box" in name:
            files["box"] = f

        elif "gland" in name or "prostate" in name:
            files["gland"] = f

        elif "tumor" in name or "lesion" in name or "cspca" in name:
            files["tumor"] = f

        elif "hbv" in name or "dwi" in name:
            files["hbv"] = f

        elif "adc_registered" in name or "adc_reg" in name:
            files["adc_reg"] = f

        elif "adc" in name:
            files["adc"] = f

        elif "t2w" in name or "t2" in name:
            files["t2"] = f

    return files


def choose_roi(files):
    if files["box"] is not None:
        return files["box"], "box"

    if files["gland"] is not None:
        return files["gland"], "gland"

    if files["tumor"] is not None:
        return files["tumor"], "tumor"

    return None, "whole_image"


def choose_modalities(files):
    mods = []

    if files["t2"] is not None:
        mods.append(("t2", files["t2"]))

    if files["adc_reg"] is not None:
        mods.append(("adc_reg", files["adc_reg"]))
    elif files["adc"] is not None:
        mods.append(("adc", files["adc"]))

    if files["hbv"] is not None:
        mods.append(("hbv", files["hbv"]))

    if len(mods) == 0:
        raise ValueError("No usable MRI modality found")

    if len(mods) == 1:
        mods = [mods[0], mods[0], mods[0]]

    elif len(mods) == 2:
        # Important:
        # Do NOT use gland/box as channel 3.
        # Mask is only for cropping.
        # For T2+ADC prostate datasets without HBV, use T2 as third channel.
        mods = [mods[0], mods[1], mods[0]]

    elif len(mods) > 3:
        mods = mods[:3]

    return mods


def get_crop_indices(mask):
    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        return None

    ymin, xmin, zmin = coords.min(axis=0)
    ymax, xmax, zmax = coords.max(axis=0)

    return zmin, zmax, ymin, ymax, xmin, xmax


def clip_crop(crop, shape):
    zmin, zmax, ymin, ymax, xmin, xmax = crop

    zmin = max(0, min(zmin, shape[2] - 1))
    zmax = max(0, min(zmax, shape[2] - 1))

    ymin = max(0, min(ymin, shape[0] - 1))
    ymax = max(0, min(ymax, shape[0] - 1))

    xmin = max(0, min(xmin, shape[1] - 1))
    xmax = max(0, min(xmax, shape[1] - 1))

    return zmin, zmax, ymin, ymax, xmin, xmax


def shapes_match(vols):
    shapes = [v.shape for v in vols]
    return all(s == shapes[0] for s in shapes)

def resample_to_shape(vol, target_shape, is_mask=False):
    vol_t = torch.tensor(vol, dtype=torch.float32)[None, None]  # [1, 1, D, H, W] if 3D
    vol_t = vol_t.permute(0, 1, 4, 2, 3)  # if vol is [H, W, D] -> [1,1,D,H,W]

    target_d, target_h, target_w = target_shape[2], target_shape[0], target_shape[1]

    mode = "nearest" if is_mask else "trilinear"
    vol_t = F.interpolate(
        vol_t,
        size=(target_d, target_h, target_w),
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )

    vol_t = vol_t.permute(0, 1, 3, 4, 2)  # back to [1,1,H,W,D]
    return vol_t[0, 0].cpu().numpy()

# =========================
# Feature extraction
# =========================
summary_rows = []

with torch.no_grad():
    for case_dir in tqdm(sorted(DATA_DIR.iterdir())):
        if not case_dir.is_dir():
            continue

        case = case_dir.name

        try:
            files = find_case_files(case_dir)
            selected_modalities = choose_modalities(files)
            roi_path, roi_type = choose_roi(files)

            vols = []
            modality_names = []

            for mod_name, mod_path in selected_modalities:
                vols.append(load_and_norm(mod_path))
                modality_names.append(mod_name)

            # if not shapes_match(vols):
            #     print(f"Skipping {case}: modality shapes do not match {[v.shape for v in vols]}")
            #     continue

            ref_shape = vols[0].shape

            aligned_vols = [vols[0]]
            for vol in vols[1:]:
                if vol.shape != ref_shape:
                    vol = resample_to_shape(vol, ref_shape, is_mask=False)
                aligned_vols.append(vol)
            
            vols = aligned_vols

            if roi_path is not None:
                roi = nib.load(str(roi_path)).get_fdata()

                if roi.shape != ref_shape:
                    # print(f"Skipping {case}: ROI shape {roi.shape} does not match image shape {ref_shape}")
                    # continue
                    roi = resample_to_shape(roi, ref_shape, is_mask=True)

                crop = get_crop_indices(roi)

                if crop is None:
                    print(f"Skipping {case}: empty ROI mask")
                    continue

                zmin, zmax, ymin, ymax, xmin, xmax = clip_crop(crop, ref_shape)

            else:
                ymin, ymax = 0, ref_shape[0] - 1
                xmin, xmax = 0, ref_shape[1] - 1
                zmin, zmax = 0, ref_shape[2] - 1

            case_feats = []

            for z in range(zmin, zmax + 1):
                slices = [
                    vol[ymin:ymax + 1, xmin:xmax + 1, z]
                    for vol in vols
                ]

                if any(s.size == 0 for s in slices):
                    continue

                img_slice = np.stack(slices, axis=0)

                img_slice = torch.tensor(
                    img_slice,
                    dtype=torch.float32
                ).unsqueeze(0).to(device)

                img_slice = F.interpolate(
                    img_slice,
                    size=(args.image_size, args.image_size),
                    mode="bilinear",
                    align_corners=False,
                )

                feat_map = model.image_encoder(img_slice)
                pooled_feat = feat_map.mean(dim=[2, 3])

                case_feats.append(pooled_feat.cpu().numpy())

            if len(case_feats) == 0:
                print(f"Skipping {case}: no valid slices")
                continue

            case_feats = np.vstack(case_feats)

            np.save(OUT_DIR / f"{case}_slice_features.npy", case_feats)
            np.save(
                OUT_DIR / f"{case}_case_mean_feature.npy",
                case_feats.mean(axis=0)
            )

            summary_rows.append(
                f"{case},{roi_type},{'|'.join(modality_names)},"
                f"{case_feats.shape[0]},{case_feats.shape[1]}"
            )

            print(
                f"{case}: ROI={roi_type}, mods={modality_names}, "
                f"features={case_feats.shape}"
            )

        except Exception as e:
            print(f"Error processing {case}: {e}")


summary_path = OUT_DIR / "feature_extraction_summary.csv"

with open(summary_path, "w") as f:
    f.write("case,roi_type,modalities,num_slices,feature_dim\n")
    for row in summary_rows:
        f.write(row + "\n")

print(f"\nDone. Features saved to: {OUT_DIR}")
print(f"Summary saved to: {summary_path}")
