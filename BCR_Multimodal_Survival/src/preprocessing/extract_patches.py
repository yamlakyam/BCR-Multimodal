import os
import multiprocessing
import argparse

from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor

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
# output_dir = "./data/interim/wsi/patches"
output_dir = "./data/interim/wsi/all_patches"



# ------------------- Helpers -------------------

def get_wsi_files(directory):
    valid_exts = (".svs", ".ndpi", ".tif", ".tiff")
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(valid_exts)
    ]


def extract_one_slide(args_tuple):
    slide_path, extractor = args_tuple
    try:
        return slide_path, extractor.extract(slide_path), None
    except Exception as e:
        return slide_path, None, repr(e)


# ------------------- Main -------------------

if __name__ == "__main__":
    wsi_files = get_wsi_files(wsi_dir)
    print(f"Found {len(wsi_files)} files in {wsi_dir}")

    if len(wsi_files) == 0:
        print("No WSIs found. Check your symlinks in ./data/raw/wsi")
        raise SystemExit(0)

    # Keep memory under control:
    # - outer processes handle slides
    # - inner threads handle patch writing
    # With only a few slides, more than 3-4 processes usually hurts more than it helps.
    effective_processes = min(num_processors, len(wsi_files), 4)

    parameters = ExtractorParameters(
        save_dir=output_dir,
        save_format=".png",
        sample_cnt=-1,
        patch_size=224,
        patch_filter_by_area=0.75,
        with_anno=False,
        extract_layer=0,
        stride=224,
        threads=2,        
        rescale_rate=512,  # 
    )

    tissue_detector = TissueDetector("LAB_Threshold", threshold=85)
    patch_extractor = PatchExtractor(tissue_detector, parameters)

    print(f"Starting extraction on {len(wsi_files)} slides using {effective_processes} cores...")

    with multiprocessing.Pool(processes=effective_processes) as pool:
        results = pool.map(extract_one_slide, [(fp, patch_extractor) for fp in wsi_files])

    for slide_path, patches_cnt, err in results:
        if err is None:
            print(f"OK: {os.path.basename(slide_path)} -> {patches_cnt} patches")
        else:
            print(f"FAIL: {os.path.basename(slide_path)} -> {err}")

    print("Patch extraction completed!")