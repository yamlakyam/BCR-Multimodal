
import os
import cv2
import torch
import torchstain
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


def run_stain_normalization(base_dir, target_path):

    # Setup Device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load reference image
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Reference patch not found at {target_path}"
        )

    target = cv2.cvtColor(
        cv2.imread(target_path),
        cv2.COLOR_BGR2RGB
    )

    # Transformations
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(device))
    ])

    # Fit Normalizer
    print("Fitting Macenko Normalizer to reference...")

    normalizer = torchstain.normalizers.MacenkoNormalizer(
        backend="torch"
    )

    normalizer.fit(T(target))

    # Collect images from the flat directory
    image_files = [
        f for f in os.listdir(base_dir)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    print(f"Found {len(image_files)} images in {base_dir}")

    # Process images
    for image_file in tqdm(image_files):

        image_path = os.path.join(
            base_dir,
            image_file
        )

        try:

            img = cv2.cvtColor(
                cv2.imread(image_path),
                cv2.COLOR_BGR2RGB
            )

            img_tensor = T(img)

            # Normalize
            norm_img_tensor, _, _ = normalizer.normalize(
                I=img_tensor,
                stains=True
            )

            # Convert back to CPU numpy for saving
            norm_img = (
                norm_img_tensor
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            Image.fromarray(norm_img).save(image_path)

        except Exception as e:
            print(f"Skipping {image_file} due to error: {e}")

    print("✅ All images normalized and saved in-place.")


if __name__ == "__main__":

    run_stain_normalization(
        base_dir="./data/interim/wsi/all_patches",
        target_path="./data/reference/reference-patch.png",
    )
