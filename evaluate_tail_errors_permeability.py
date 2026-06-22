"""
Permeability counterpart to evaluate_tail_errors_v4.py. No training here -- loads the
already-trained baseline_model.pth and physics_informed_model.pth, runs inference
once, and checks whether the physics-informed model does meaningfully better on the
most severe low-Vf permeability samples specifically, not just on average (same
question as for elastic, same method).

Unlike the elastic version, low-Vf here is unambiguous: Vf is the real ground-truth
fiber_volume_fraction column, not an Otsu/E11-derived estimate, so there's no
"which Vf definition" caveat to carry through this analysis.

REPRODUCES THE EXACT train/test split used by train_eval_permeability_v1.py /
train_physics_only_permeability.py (same CSV, same SEED, same image-existence filter,
same 70/30 split, same StandardScaler fit in log10 space) by importing
train_eval_permeability_v1.py directly rather than reimplementing it.

HOW TO RUN (PowerShell, same env, CPU is fine -- inference only):
    python evaluate_tail_errors_permeability.py --data-dir "C:\\path\\to\\permeability\\images" `
        --csv Combined_Properties_with_FVF_updated.csv --results-dir results_permeability_lambda025 `
        --out-dir tail_analysis_permeability_lambda025

OUTPUT (--out-dir):
    per_sample_results.csv        -- every test sample, mu1/mu2/mu3, both models'
                                      predictions and RPE, plus Vf
    tail_percentile_summary.csv   -- p50/p90/p95/max RPE for baseline vs physics-
                                      informed, low-Vf subset, per property
    worst_case_comparison.csv     -- the N worst baseline samples (by RPE) in the
                                      low-Vf subset, with the physics-informed model's
                                      RPE on those SAME samples alongside
    fig_tail_rpe_vs_vf.pdf, fig_worst_case_comparison.pdf, fig_percentile_comparison.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import bounds_permeability as bp
import train_eval_permeability_v1 as pv1
from train_physics_only_permeability import rebuild_permeability_split

SEED = pv1.SEED
LOW_VF_THRESHOLD = pv1.LOW_VF_THRESHOLD
ACTIONABLE_PROPERTIES = ["mu1", "mu2", "mu3"]  # all three are meaningful for permeability


@torch.no_grad()
def run_inference(model, test_df, data_dir, scaler, device, target_space):
    ds = pv1.PermeabilityDataset(test_df, data_dir, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    model.eval().to(device)
    all_preds, all_labels, all_vf = [], [], []
    for images, labels, vf in loader:
        images = images.to(device)
        outputs_scaled = model(images).cpu().numpy()
        outputs_space = outputs_scaled * scaler.scale_ + scaler.mean_
        outputs_raw = (10 ** outputs_space) if target_space == "log" else outputs_space
        all_preds.append(outputs_raw)
        all_labels.append(labels.numpy())
        all_vf.append(vf.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_vf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--csv", default="Combined_Properties_with_FVF_updated.csv")
    ap.add_argument("--results-dir", default="results_permeability_lambda025")
    ap.add_argument("--out-dir", default="tail_analysis_permeability_lambda025")
    ap.add_argument("--target-space", choices=["log", "linear"], default="log")
    ap.add_argument("--top-n-worst", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Rebuilding the exact train/test split...")
    train_df, test_df, scaler = rebuild_permeability_split(args.csv, args.data_dir)
    print(f"Test set: {len(test_df)} samples")

    print("Loading saved models...")
    base_model = pv1.PermeabilityCNN(num_outputs=len(bp.PERMEABILITY_NAMES))
    base_model.load_state_dict(torch.load(
        os.path.join(args.results_dir, "baseline_model.pth"), map_location=device))
    phys_model = pv1.PermeabilityCNN(num_outputs=len(bp.PERMEABILITY_NAMES))
    phys_model.load_state_dict(torch.load(
        os.path.join(args.results_dir, "physics_informed_model.pth"), map_location=device))

    print("Running inference (baseline)...")
    base_preds, labels, vf = run_inference(base_model, test_df, args.data_dir, scaler, device, args.target_space)
    print("Running inference (physics-informed)...")
    phys_preds, _labels2, _vf2 = run_inference(phys_model, test_df, args.data_dir, scaler, device, args.target_space)

    rows = []
    for i in range(len(test_df)):
        row = {"Input_File": test_df.iloc[i]["Input_File"], "vf": float(vf[i])}
        for j, name in enumerate(bp.PERMEABILITY_NAMES):
            true_val = labels[i, j]
            base_pred = base_preds[i, j]
            phys_pred = phys_preds[i, j]
            row[f"true_{name}"] = true_val
            row[f"baseline_pred_{name}"] = base_pred
            row[f"baseline_RPE_{name}"] = abs((base_pred - true_val) / true_val) * 100
            row[f"physics_pred_{name}"] = phys_pred
            row[f"physics_RPE_{name}"] = abs((phys_pred - true_val) / true_val) * 100
        rows.append(row)
    per_sample = pd.DataFrame(rows)
    per_sample["baseline_RPE_mean_actionable"] = per_sample[
        [f"baseline_RPE_{p}" for p in ACTIONABLE_PROPERTIES]].mean(axis=1)
    per_sample["physics_RPE_mean_actionable"] = per_sample[
        [f"physics_RPE_{p}" for p in ACTIONABLE_PROPERTIES]].mean(axis=1)
    per_sample.to_csv(os.path.join(args.out_dir, "per_sample_results.csv"), index=False)
    print(f"Saved per_sample_results.csv ({len(per_sample)} rows)")

    low_mask = per_sample["vf"] < LOW_VF_THRESHOLD
    low_df = per_sample[low_mask]
    print(f"Low-Vf subset: {len(low_df)} samples")

    percentile_rows = []
    for name in bp.PERMEABILITY_NAMES:
        b = low_df[f"baseline_RPE_{name}"].values
        p = low_df[f"physics_RPE_{name}"].values
        for pct_name, pct_fn in [("p50", lambda x: np.percentile(x, 50)),
                                  ("p90", lambda x: np.percentile(x, 90)),
                                  ("p95", lambda x: np.percentile(x, 95)),
                                  ("max", np.max)]:
            b_val, p_val = pct_fn(b), pct_fn(p)
            pct_change = 100 * (b_val - p_val) / b_val if b_val else float("nan")
            percentile_rows.append({"property": name, "percentile": pct_name,
                                     "baseline_RPE": b_val, "physics_RPE": p_val,
                                     "pct_change": pct_change})
    percentile_df = pd.DataFrame(percentile_rows)
    percentile_df.to_csv(os.path.join(args.out_dir, "tail_percentile_summary.csv"), index=False)
    print("\nTail percentile summary (low-Vf subset, positive pct_change = physics better):")
    print(percentile_df.to_string(index=False))

    worst = low_df.nlargest(args.top_n_worst, "baseline_RPE_mean_actionable")
    cols = ["Input_File", "vf", "baseline_RPE_mean_actionable", "physics_RPE_mean_actionable"]
    for name in ACTIONABLE_PROPERTIES:
        cols += [f"baseline_RPE_{name}", f"physics_RPE_{name}"]
    worst[cols].to_csv(os.path.join(args.out_dir, "worst_case_comparison.csv"), index=False)
    mean_b = worst["baseline_RPE_mean_actionable"].mean()
    mean_p = worst["physics_RPE_mean_actionable"].mean()
    print(f"\nWorst {args.top_n_worst} baseline samples (low-Vf):")
    print(f"  Mean baseline RPE on these samples: {mean_b:.2f}%")
    print(f"  Mean physics-informed RPE on these SAME samples: {mean_p:.2f}%")
    print(f"  Change: {100*(mean_b-mean_p)/mean_b:+.1f}% (positive = physics-informed better)")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(low_df["vf"], low_df["baseline_RPE_mean_actionable"],
               s=18, alpha=0.6, label="Baseline", color="tab:grey")
    ax.scatter(low_df["vf"], low_df["physics_RPE_mean_actionable"],
               s=18, alpha=0.6, label="Physics-informed", color="tab:blue")
    ax.axvline(LOW_VF_THRESHOLD, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("$V_f$ (ground truth)")
    ax.set_ylabel("Mean RPE across $\\mu_1,\\mu_2,\\mu_3$ (%)")
    ax.set_title("Per-sample error vs.\\ $V_f$, low-$V_f$ region")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_tail_rpe_vs_vf.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(args.top_n_worst)
    width = 0.35
    worst_sorted = worst.sort_values("baseline_RPE_mean_actionable", ascending=False)
    ax.bar(x - width / 2, worst_sorted["baseline_RPE_mean_actionable"], width, label="Baseline", color="tab:grey")
    ax.bar(x + width / 2, worst_sorted["physics_RPE_mean_actionable"], width, label="Physics-informed", color="tab:blue")
    ax.set_xlabel(f"Worst {args.top_n_worst} baseline samples (low-$V_f$), ranked")
    ax.set_ylabel("Mean RPE (%)")
    ax.set_title("Baseline's worst cases: same samples, physics-informed model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_worst_case_comparison.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    pivot_b = percentile_df.groupby("percentile")["baseline_RPE"].mean().reindex(["p50", "p90", "p95", "max"])
    pivot_p = percentile_df.groupby("percentile")["physics_RPE"].mean().reindex(["p50", "p90", "p95", "max"])
    x = np.arange(len(pivot_b))
    ax.bar(x - width / 2, pivot_b.values, width, label="Baseline", color="tab:grey")
    ax.bar(x + width / 2, pivot_p.values, width, label="Physics-informed", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_b.index)
    ax.set_xlabel("Percentile of low-$V_f$ RPE distribution")
    ax.set_ylabel("RPE (%), averaged over mu1/mu2/mu3")
    ax.set_title("Tail behavior: low-$V_f$ RPE percentiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_percentile_comparison.pdf"))
    plt.close(fig)

    print(f"\nDone. Everything saved to {args.out_dir}.")


if __name__ == "__main__":
    main()
