from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def norm_id(x: object) -> str:
    return str(x).replace(" ", "").replace("-", "").replace("_", "").upper().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        default="data/processed/fusion/fusion_predictions.csv",
        help="CSV with patient_id and risk columns.",
    )
    parser.add_argument(
        "--clinical_csv",
        type=str,
        required=True,
        help="Clinical CSV with patient_id, time, and event columns.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/fusion/evaluation",
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--save_km_plot",
        action="store_true",
        help="Save Kaplan-Meier plot as PNG.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    pred_path = root / args.predictions
    clinical_path = root / args.clinical_csv
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    if not clinical_path.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {clinical_path}")

    preds = pd.read_csv(pred_path)
    clinical = pd.read_csv(clinical_path)

    required_pred_cols = {"patient_id", "risk"}
    required_clin_cols = {"patient_id", "time", "event"}

    missing_pred = required_pred_cols - set(preds.columns)
    missing_clin = required_clin_cols - set(clinical.columns)

    if missing_pred:
        raise KeyError(f"Predictions CSV missing columns: {sorted(missing_pred)}")
    if missing_clin:
        raise KeyError(f"Clinical CSV missing columns: {sorted(missing_clin)}")

    preds["patient_id"] = preds["patient_id"].astype(str).map(norm_id)
    clinical["patient_id"] = clinical["patient_id"].astype(str).map(norm_id)

    clinical["time"] = pd.to_numeric(clinical["time"], errors="coerce")
    clinical["event"] = pd.to_numeric(clinical["event"], errors="coerce").fillna(0).astype(int)
    clinical = clinical.dropna(subset=["time"])

    merged = preds.merge(clinical, on="patient_id", how="inner")
    if len(merged) == 0:
        raise RuntimeError("No matched patient IDs found between predictions and clinical CSV.")

    merged = merged.sort_values("risk", ascending=False).reset_index(drop=True)

    cidx = concordance_index(
        merged["time"].values,
        -merged["risk"].values,
        merged["event"].values,
    )

    median_risk = merged["risk"].median()
    merged["risk_group"] = np.where(merged["risk"] >= median_risk, "High risk", "Low risk")

    low = merged[merged["risk_group"] == "Low risk"]
    high = merged[merged["risk_group"] == "High risk"]

    lr = None
    if len(low) > 0 and len(high) > 0 and merged["event"].nunique() > 1:
        lr = logrank_test(
            low["time"],
            high["time"],
            event_observed_A=low["event"],
            event_observed_B=high["event"],
        )

    merged.to_csv(out_dir / "merged_predictions_clinical.csv", index=False)

    metrics = {
        "n_matched": int(len(merged)),
        "c_index": float(cidx),
        "median_risk": float(median_risk),
        "high_risk_n": int(len(high)),
        "low_risk_n": int(len(low)),
        "logrank_p_value": float(lr.p_value) if lr is not None else None,
        "logrank_statistic": float(lr.test_statistic) if lr is not None else None,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 60)
    print(f"Matched patients: {len(merged)}")
    print(f"C-index: {cidx:.4f}")
    if lr is not None:
        print(f"Log-rank p-value: {lr.p_value:.6e}")
    else:
        print("Log-rank test not computed (insufficient group/event variation).")
    print("=" * 60)

    if args.save_km_plot:
        kmf = KaplanMeierFitter()
        fig, ax = plt.subplots(figsize=(8, 6))

        for grp in ["Low risk", "High risk"]:
            mask = merged["risk_group"] == grp
            if mask.sum() == 0:
                continue
            kmf.fit(
                merged.loc[mask, "time"],
                event_observed=merged.loc[mask, "event"],
                label=grp,
            )
            kmf.plot_survival_function(ax=ax, ci_show=True)

        ax.set_title("Kaplan-Meier Curves by Predicted Risk")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival probability")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        plot_path = out_dir / "km_curve.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        print(f"Saved KM plot to: {plot_path}")


if __name__ == "__main__":
    main()