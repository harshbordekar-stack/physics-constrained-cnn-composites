"""
Tail/edge-case analysis for the already-trained v4 elastic models. No training here --
this loads the saved baseline_model.pth and physics_informed_model.pth from results_v4,
re-runs inference once, and checks the specific hypothesis: does the physics-informed
model do much better than the baseline on the WORST individual samples, even though the
mean low-Vf RPE only improved a modest 2.6% (Section 3 of main.tex)?

WHY MEAN RPE CAN HIDE THIS: the Voigt-Reuss/HS hinge penalty is exactly zero for any
prediction already inside the admissible bound, regardless of Vf -- it only ever
contributes gradient for samples that actually violate a bound. In the v4 baseline,
that was a small fraction even of the low-Vf subset (bound-violation rates of
0.2-3.2% for E11/E22/E33/G12/G13, even before restricting to low Vf). So if the
constraint loss helps at all, the effect should be concentrated in a small number of
genuinely bad samples, not spread evenly across the whole low-Vf subset -- exactly the
kind of effect a mean can wash out. This script checks the tail directly instead of
assuming either outcome.

IMPORTANT, ALREADY-KNOWN CONTEXT THAT THIS DOES NOT CONTRADICT OR REPEAT: bound-
violation RATE did not improve in the v4 run (E22 went 3.2%->3.9%, E33 2.7%->3.4%,
slightly worse). That's a different statistic from what's checked here -- violation
rate is binary (in/out of bounds), this script looks at error MAGNITUDE on the worst
specific samples, which could improve even if the violation rate doesn't. Report
whatever this actually shows; do not write a paper subsection claiming edge-case
superiority if these numbers don't support it.

REPRODUCES THE EXACT v4 TRAIN/TEST SPLIT (same CSV, same SEED, same 70/30 split, same
StandardScaler fit on the same train rows) by importing train_eval_v4.py directly
rather than re-implementing it, so there is no risk of a subtly different split
producing a test set that doesn't match what the saved .pth files were evaluated on.

HOW TO RUN (PowerShell, same env, no GPU required for this -- inference on ~3000
images takes a couple of minutes either way):
    python evaluate_tail_errors_v4.py --data-dir "C:\\path\\to\\elastic\\images" `
        --csv properties.csv --results-dir results_v4 --out-dir tail_analysis_v4

OUTPUT (--out-dir), nothing needs recomputing afterward:
    per_sample_results.csv        -- every test sample, every property, both models'
                                      predictions and RPE, plus Vf -- full granularity,
                                      re-derive anything else from this if needed later
    tail_percentile_summary.csv   -- p50/p90/p95/max RPE for baseline vs physics-
                                      informed, low-Vf subset, per property
    worst_case_comparison.csv     -- the N worst baseline samples (by RPE) in the
                                      low-Vf subset, with the physics-informed model's
                                      RPE on those SAME samples alongside
    fig_tail_rpe_vs_vf.pdf        -- paper-ready figure: RPE vs Vf scatter, both models
    fig_worst_case_comparison.pdf -- paper-ready figure: paired bars, worst-N samples
    fig_percentile_comparison.pdf -- paper-ready figure: grouped bars, percentiles
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import bounds
import train_eval_v4 as v4

SEED = v4.SEED
LOW_VF_THRESHOLD = v4.LOW_VF_THRESHOLD
CSV_TO_CANONICAL = {
    "E_11": "E11", "E_22": "E22", "E_33": "E33",
    "G_21": "G12", "G_31": "G13",
    "nu_12": "nu12", "nu_13": "nu13", "nu_23": "nu23",
}

# Properties the constraint loss could plausibly have helped (it does NOT apply to
# nu12/nu13/nu23 -- see bounds.py/main.tex sec:constraint_discussion). Aggregate
# "worst sample" ranking uses these five only, so a high-error sample isn't picked
# just because of the known-unfixable nu23 issue.
ACTIONABLE_PROPERTIES = ["E11", "E22", "E33", "G12", "G13"]


def rebuild_v4_split(csv_path, data_dir):
    """Exactly reproduces train_eval_v4.py's main(): same rename, same shuffle seed,
    same 70/30 split, same scaler fit on the same train rows."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns=CSV_TO_CANONICAL)
    missing = [c for c in bounds.PROPERTY_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing {missing} after rename. Found: {list(df.columns)}")
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    split = int(len(df) * 0.7)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    scaler = StandardScaler()
    scaler.fit(train_df[bounds.PROPERTY_NAMES].values)
    return train_df, test_df, scaler


@torch.no_grad()
def run_inference(model, test_df, data_dir, scaler, device):
    """Runs inference in test_df row order (shuffle=False preserves this regardless of
    num_workers), returns physical-unit predictions, labels, and vf_otsu, all aligned
    to test_df.reset_index(drop=True) row order."""
    ds = v4.CFRPDataset(test_df, data_dir, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    model.eval().to(device)

    all_preds, all_labels, all_vf = [], [], []
    for images, labels, vf_otsu, _vf_e11 in loader:
        images = images.to(device)
        outputs_scaled = model(images).cpu().numpy()
        outputs_phys = outputs_scaled * scaler.scale_ + scaler.mean_
        all_preds.append(outputs_phys)
        all_labels.append(labels.numpy())
        all_vf.append(vf_otsu.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_vf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--csv", default="properties.csv")
    ap.add_argument("--results-dir", default="results_v4")
    ap.add_argument("--out-dir", default="tail_analysis_v4")
    ap.add_argument("--top-n-worst", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Rebuilding the exact v4 train/test split...")
    train_df, test_df, scaler = rebuild_v4_split(args.csv, args.data_dir)
    print(f"Test set: {len(test_df)} samples (should match v4's n_test)")

    print("Loading saved models...")
    base_model = v4.MicrostructureCNN(num_outputs=len(bounds.PROPERTY_NAMES))
    base_model.load_state_dict(torch.load(
        os.path.join(args.results_dir, "baseline_model.pth"), map_location=device))
    phys_model = v4.MicrostructureCNN(num_outputs=len(bounds.PROPERTY_NAMES))
    phys_model.load_state_dict(torch.load(
        os.path.join(args.results_dir, "physics_informed_model.pth"), map_location=device))

    print("Running inference (baseline)...")
    base_preds, labels, vf_otsu = run_inference(base_model, test_df, args.data_dir, scaler, device)
    print("Running inference (physics-informed)...")
    phys_preds, _labels2, _vf2 = run_inference(phys_model, test_df, args.data_dir, scaler, device)
    # _labels2/_vf2 should be identical to labels/vf_otsu (same test_df, shuffle=False);
    # not asserted byte-for-byte since augmentation is off and order is deterministic,
    # but if anything looks off downstream, check this assumption first.

    # ---- Build the full per-sample table -------------------------------------------
    rows = []
    for i in range(len(test_df)):
        row = {"Input_File": test_df.iloc[i]["Input_File"], "vf_otsu": float(vf_otsu[i])}
        for j, name in enumerate(bounds.PROPERTY_NAMES):
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

    # ---- Percentile summary on the low-Vf subset ------------------------------------
    low_mask = per_sample["vf_otsu"] < LOW_VF_THRESHOLD
    low_df = per_sample[low_mask]
    print(f"Low-Vf subset: {len(low_df)} samples")

    percentile_rows = []
    for name in bounds.PROPERTY_NAMES:
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

    # ---- Worst-N baseline samples, compared side by side ----------------------------
    worst = low_df.nlargest(args.top_n_worst, "baseline_RPE_mean_actionable")
    cols = ["Input_File", "vf_otsu", "baseline_RPE_mean_actionable", "physics_RPE_mean_actionable"]
    for name in ACTIONABLE_PROPERTIES:
        cols += [f"baseline_RPE_{name}", f"physics_RPE_{name}"]
    worst[cols].to_csv(os.path.join(args.out_dir, "worst_case_comparison.csv"), index=False)
    mean_b = worst["baseline_RPE_mean_actionable"].mean()
    mean_p = worst["physics_RPE_mean_actionable"].mean()
    print(f"\nWorst {args.top_n_worst} baseline samples (low-Vf, actionable properties only):")
    print(f"  Mean baseline RPE on these samples: {mean_b:.2f}%")
    print(f"  Mean physics-informed RPE on these SAME samples: {mean_p:.2f}%")
    print(f"  Change: {100*(mean_b-mean_p)/mean_b:+.1f}% (positive = physics-informed better)")

    # ---- Figures ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(low_df["vf_otsu"], low_df["baseline_RPE_mean_actionable"],
               s=18, alpha=0.6, label="Baseline", color="tab:grey")
    ax.scatter(low_df["vf_otsu"], low_df["physics_RPE_mean_actionable"],
               s=18, alpha=0.6, label="Physics-informed", color="tab:blue")
    ax.axvline(LOW_VF_THRESHOLD, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("$V_f$ (Otsu estimate)")
    ax.set_ylabel("Mean RPE across $E_{11},E_{22},E_{33},G_{12},G_{13}$ (%)")
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
    actionable_pct = percentile_df[percentile_df.property.isin(ACTIONABLE_PROPERTIES)]
    pivot_b = actionable_pct.groupby("percentile")["baseline_RPE"].mean().reindex(["p50", "p90", "p95", "max"])
    pivot_p = actionable_pct.groupby("percentile")["physics_RPE"].mean().reindex(["p50", "p90", "p95", "max"])
    x = np.arange(len(pivot_b))
    ax.bar(x - width / 2, pivot_b.values, width, label="Baseline", color="tab:grey")
    ax.bar(x + width / 2, pivot_p.values, width, label="Physics-informed", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_b.index)
    ax.set_xlabel("Percentile of low-$V_f$ RPE distribution")
    ax.set_ylabel("RPE (%), averaged over actionable properties")
    ax.set_title("Tail behavior: low-$V_f$ RPE percentiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_percentile_comparison.pdf"))
    plt.close(fig)

    print(f"\nDone. Everything saved to {args.out_dir} -- figures are paper-ready PDFs, "
          f"no need to regenerate.")


if __name__ == "__main__":
    main()
