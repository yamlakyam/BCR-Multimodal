
import os
import argparse
import torch
import timm
import h5py
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login


def get_args():
    parser = argparse.ArgumentParser(description="UNI2-h Feature Extraction")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Login Token")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/interim/wsi/all_patches",
        help="Path to flattened patches",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/interim/wsi/embeddings/patch_embeddings.h5",
        help="Output H5 filename",
    )
    return parser.parse_args()


class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return {"image": image, "filename": os.path.basename(img_path)}
        except (UnidentifiedImageError, OSError, ValueError):
            return None


def collate_fn_safe(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def extract(args):
    login(token=args.token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading UNI2-h on {device}...")
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 5.33334,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval().to(device)

    dataset = PatchDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=4,
        num_workers=1,
        collate_fn=collate_fn_safe,
    )

    all_embeddings = []
    all_filenames = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            if batch is None:
                continue
            imgs = batch["image"].to(device)
            fnames = batch["filename"]

            features = model(imgs)
            all_embeddings.append(features.cpu().numpy())
            all_filenames.extend(fnames)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    ascii_filenames = [n.encode("ascii", "ignore") for n in all_filenames]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with h5py.File(args.output_path, "w") as f:
        f.create_dataset("embeddings", data=all_embeddings)
        f.create_dataset("filenames", data=ascii_filenames)

    print(f"Successfully saved {len(all_filenames)} features to {args.output_path}")


if __name__ == "__main__":
    args = get_args()
    extract(args)
