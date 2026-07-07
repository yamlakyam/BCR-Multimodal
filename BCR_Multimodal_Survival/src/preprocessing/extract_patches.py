
import os
import multiprocessing
from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
import argparse
# ------------------- Configuration -------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--num_processors",
    type=int,
    default=4,
    help="Number of CPU workers for patch extraction",
)

args = parser.parse_args()

num_processors = args.num_processors

wsi_dir = "./data/raw/wsi"
output_dir = "./data/interim/wsi/patches"

# Patch extraction parameters (Matches your Prostate Research setup)
parameters = ExtractorParameters(
    save_dir=output_dir,         # Where the patches will be saved
    save_format='.png',          # Format of extracted patches
    sample_cnt=-1,               # -1 = extract all patches
    # sample_cnt=10000,               # 
    patch_size=224,              # Size of patches
    patch_filter_by_area=0.75,    # Minimum tissue proportion (50%)
    with_anno=False,             # Set to False as these are sample WSIs
    extract_layer=0,             # Full resolution
    stride=224                   # No overlap
)

# Tissue detection method (LAB Thresholding)
tissue_detector = TissueDetector(
    "LAB_Threshold",
    threshold=85
)

# Create the PatchExtractor object
patch_extractor = PatchExtractor(
    tissue_detector,
    parameters
)

# ------------------- Execution -------------------
def get_wsi_files(directory):
    valid_exts = ('.svs', '.ndpi', '.tif', '.tiff')
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(valid_exts)]

if __name__ == '__main__':
    wsi_files = get_wsi_files(wsi_dir)
    print(f"Found {len(wsi_files)} files in {wsi_dir}")

    if len(wsi_files) == 0:
        print(" No WSIs found. Check your symlinks in ./data/WSI")
    else:
        # Using multiprocessing.Pool safely
        print(f" Starting extraction on {len(wsi_files)} slides using {num_processors} cores...")
        with multiprocessing.Pool(processes=num_processors) as pool:
            pool.map(patch_extractor.extract, wsi_files)

        print(" Patch extraction completed!")
