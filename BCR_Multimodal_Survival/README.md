# Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/yamlakyam/BCR-Multimodal.git
cd BCR-Multimodal/BCR_Multimodal_Survival
```

## 2. Install dependencies

```bash
chmod +x setup.sh
bash setup.sh
```

If your system already has PyTorch installed and working, keep that installation as-is. The repository setup installs the remaining packages and prepares the directory structure.

---

## 3. Prepare the data

Copy the raw data into the expected folders:

```text
data/raw/wsi/
data/raw/mri/
data/raw/clinical/
```

### Whole Slide Images

Copy all WSI files into:

```text
data/raw/wsi/
```

Example:

```text
data/raw/wsi/
├── 0760_11.svs
├── 0760_51.svs
├── 0760_52.svs
└── ...
```

### MRI data

Copy each patient’s MRI folder into:

```text
data/raw/mri/
```

Example:

```text
data/raw/mri/
├── 0760/
│   ├── t2.nii.gz
│   ├── adc.nii.gz
│   ├── hbv.nii.gz
│   └── gland.nii.gz
└── ...
```

### Clinical data, if ground truth is available

Copy the clinical CSV into:

```text
data/raw/clinical/
```

Example:

```text
data/raw/clinical/clinical_info.csv
```

The file should contain at least these columns:

```text
patient_id,time,event
```

Where:

* `patient_id` is the matched patient identifier
* `time` is follow-up time, typically in years
* `event` is the event indicator

  * `1` = biochemical recurrence occurred
  * `0` = censored

Example:

```text
patient_id,time,event,age,psa,gleason
0760,3.2,1,66,8.4,7
0761,5.8,0,71,5.3,6
0762,1.4,1,64,9.1,8
```

---

## 4. Expected folder structure

```text
BCR_Multimodal_Survival/
├── checkpoints/
│   ├── pathology/
│   │   ├── prostate_uni2_model.joblib
│   │   └── best_mil_chimera_surv.pth
│   └── fusion/
│       └── best_fusion_chimera_surv.pth
├── data/
│   ├── raw/
│   │   ├── wsi/
│   │   ├── mri/
│   │   └── clinical/
│   ├── reference/
│   │   └── reference-patch.png
│   ├── interim/
│   │   ├── wsi/
│   │   │   ├── patches/
│   │   │   ├── all_patches/
│   │   │   ├── embeddings/
│   │   │   └── cancer_only/
│   │   └── mri/
│   │       └── embeddings/
│   └── processed/
│       ├── pathology/
│       └── fusion/
├── MRI_CORE_FEATURE_EXTRACTION/
├── outputs/
├── scripts/
├── src/
└── setup.sh
```

---

## 5. Run the pathology pipeline

### Step 1: Patch extraction

```bash
python src/preprocessing/extract_patches.py --num_processors 32
```

### Step 2: Stain normalization

```bash
python src/preprocessing/normalize_stains.py
```

### Step 3: UNI2 feature extraction

```bash
python src/inference/extract_features.py --token "HF_Token"
```

### Step 4: Patch classification

```bash
python src/inference/classify_h5.py \
  --h5 data/interim/wsi/embeddings/patch_embeddings.h5 \
  --model checkpoints/pathology/prostate_uni2_model.joblib \
  --output outputs/predictions/patch_predictions.csv
```

### Step 5: Keep cancer patches only

```bash
python src/pathology/filter_cancer.py \
  --h5 data/interim/wsi/embeddings/patch_embeddings.h5 \
  --predictions outputs/predictions/patch_predictions.csv \
  --wsi_dir data/raw/wsi \
  --output data/interim/wsi/cancer_only/cancer_embeddings.h5
```

### Step 6: Build patient bags

```bash
python src/pathology/build_patient_bags.py \
  --h5 data/interim/wsi/cancer_only/cancer_embeddings.h5
```

### Step 7: MIL feature extraction

```bash
python src/pathology/mil_inference.py \
  --h5 data/interim/wsi/cancer_only/cancer_embeddings.h5 \
  --ckpt checkpoints/pathology/best_mil_chimera_surv.pth \
  --output data/processed/pathology/pathology_vectors.pt
```

---

## 6. Run the MRI pipeline

```bash
python MRI_CORE_FEATURE_EXTRACTION/extract_features.py
```

This will create MRI feature files inside:

```text
data/interim/mri/embeddings/
```

---

## 7. Run multimodal fusion

```bash
python src/fusion/run_fusion.py
```

This creates:

```text
data/processed/fusion/fusion_predictions.csv
data/processed/fusion/fusion_z_fused.npy
data/processed/fusion/fusion_z_path.npy
data/processed/fusion/fusion_z_mri.npy
data/processed/fusion/aggregated_pathology_vectors.pt
```

The `fusion_predictions.csv` file contains the final patient-level risk scores.

---

## 8. Evaluate survival performance, if ground truth is available

If you have a clinical CSV with survival labels, run:

```bash
python src/evaluation/evaluate_survival_csv.py \
  --predictions data/processed/fusion/fusion_predictions.csv \
  --clinical_csv data/raw/clinical/clinical_info.csv \
  --save_km_plot
```

This will compute:

* Concordance Index (C-index)
* Kaplan–Meier curves
* Log-rank test
* Merged prediction/clinical table

Outputs will be saved in:

```text
data/processed/fusion/evaluation/
```

---

# What the outputs mean

The final `risk` column in `fusion_predictions.csv` is a **Cox proportional hazards log-risk score**.

Higher values mean higher predicted risk of biochemical recurrence.

This is not a probability. It is a relative prognostic score used for ranking patients.

---

# Notes

* The WSI filenames should preserve the case/patient identifier.
* MRI folders should be organized by patient ID.
* Clinical ground truth is optional for inference, but required for evaluation.
* If multiple WSIs exist for one patient, they will be aggregated before fusion.
