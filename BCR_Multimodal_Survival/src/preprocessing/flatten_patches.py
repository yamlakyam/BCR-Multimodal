
import os
import shutil
from tqdm import tqdm


def flatten_patches(source_root, dest_dir):
    """
    Moves all .png patches from SlideID subdirectories into a single
    flat folder for easier batch processing with UNI2.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Identify all slide folders (ignoring text files in the root)
    subdirs = [f.path for f in os.scandir(source_root) if f.is_dir()]
    print(f"Flattening patches from {len(subdirs)} slide folders...")

    for slide_folder in tqdm(subdirs):
        for entry in os.scandir(slide_folder):

            # Only move PNGs; leave the 'finished.txt' files behind
            if entry.is_file() and entry.name.endswith(".png"):

                src_path = entry.path
                dst_path = os.path.join(dest_dir, entry.name)

                # Move is instantaneous on the same file system
                shutil.move(src_path, dst_path)

    print("Flattening complete.")
    print(f"Total patches ready in {dest_dir}: {len(os.listdir(dest_dir))}")


if __name__ == "__main__":

    # Standard repository paths
    flatten_patches(
        "./data/interim/wsi/patches",
        "./data/interim/wsi/all_patches",
    )
