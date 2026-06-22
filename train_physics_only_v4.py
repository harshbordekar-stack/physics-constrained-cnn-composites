"""
Trains ONLY a new physics-informed model at a specified lambda, reusing the existing
baseline from a prior train_eval_v4.py run instead of retraining it. The baseline
doesn't depend on lambda at all (plain MSE, no physics term) -- with the same CSV,
data-dir, and the fixed SEED=42 used throughout this project, retraining it would
reproduce the exact same model and waste ~70 minutes for an identical result.

WHEN TO USE THIS: you already have a results_v4/ (or similar) folder from a full
train_eval_v4.py run, and want to test one specific new lambda value without redoing
the baseline. This is exactly that case (testing the edge-case hypothesis at lambda
values not in the original scan, e.g. 0.2 or 0.3, instead of the scan-picked 0.1).

WHAT THIS DOES:
  1. Copies baseline_model.pth from --baseline-results-dir into --out-dir unchanged.
  2. Copies the baseline rows from full_dataset_performance.csv, lowvf_comparison.csv,
     and bound_violations.csv, and the baseline rows from training_curves.csv --
     these are valid as-is since the split/model/data are identical.
  3. Trains a fresh physics-informed model at the lambda you specify, full budget.
  4. Writes a complete --out-dir with the SAME file structure as a normal
     train_eval_v4.py run, so evaluate_tail_errors_v4.py can point at it with no
     changes (it just needs baseline_model.pth and physics_informed_model.pth to
     both be present).

HOW TO RUN (PowerShell, same env):
    python train_physics_only_v4.py --data-dir "C:\\path\\to\\elastic\\images" `
        --csv properties.csv --baseline-results-dir results_v4 `
        --out-dir results_v4_lambda02 --lambda-vr 0.2 --lambda-hs 0.2
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler

import bounds
import train_eval_v4 as v4

SEED = v4.SEED
CSV_TO_CANONICAL = {
    "E_11": "E11", "E_22": "E22", "E_33": "E33",
    "G_21": "G12", "G_31": "G13",
    "nu_12": "nu12", "nu_13": "nu13", "nu_23": "nu23",
}


def rebuild_v4_split(csv_path, data_dir):
    """Exactly reproduces train_eval_v4.py's main() preprocessing -- same rename, same
    shuffle seed, same 70/30 split, same scaler fit on the same train rows. Duplicated
    here (rather than imported from evaluate_tail_errors_v4.py) so this script has no
    matplotlib dependency."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--csv", default="properties.csv")
    ap.add_argument("--baseline-results-dir", default="results_v4",
                     help="Existing train_eval_v4.py output to reuse the baseline from.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lambda-vr", type=float, required=True)
    ap.add_argument("--lambda-hs", type=float, required=True)
    ap.add_argument("--vr-properties", type=str, default="E22,E33,G12,G13",
                     help="Default matches v4 (E11 excluded -- still using E11-derived "
                          "Vf in this script, same as v4, so the circularity issue "
                          "still applies unless you also pass --vf-source otsu.")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    args = ap.parse_args()

    vr_properties = set(p.strip() for p in args.vr_properties.split(",") if p.strip())
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    baseline_pth = os.path.join(args.baseline_results_dir, "baseline_model.pth")
    if not os.path.isfile(baseline_pth):
        raise FileNotFoundError(
            f"{baseline_pth} not found -- --baseline-results-dir must point at a "
            f"completed train_eval_v4.py output folder."
        )

    print(f"Reusing baseline from {args.baseline_results_dir} (no retraining)...")
    shutil.copy(baseline_pth, os.path.join(args.out_dir, "baseline_model.pth"))

    old_perf = pd.read_csv(os.path.join(args.baseline_results_dir, "full_dataset_performance.csv"))
    old_lowvf = pd.read_csv(os.path.join(args.baseline_results_dir, "lowvf_comparison.csv"))
    old_violations = pd.read_csv(os.path.join(args.baseline_results_dir, "bound_violations.csv"))
    old_curves = pd.read_csv(os.path.join(args.baseline_results_dir, "training_curves.csv"))
    baseline_perf = old_perf[old_perf.model == "baseline"].copy()
    baseline_lowvf = old_lowvf[old_lowvf.model == "baseline"].copy()
    baseline_violations = old_violations[old_violations.model == "baseline"].copy()
    baseline_curves = old_curves[old_curves.model == "baseline"].copy()
    with open(os.path.join(args.baseline_results_dir, "summary.json")) as f:
        old_summary = json.load(f)
    baseline_train_minutes = old_summary.get("baseline_train_minutes")
    baseline_n_low_vf = old_summary.get("baseline_n_low_vf")
    baseline_violations_pct = old_summary.get("baseline_violations_pct")

    print("Rebuilding the exact v4 train/test split (must match the reused baseline)...")
    train_df, test_df, scaler = rebuild_v4_split(args.csv, args.data_dir)
    from torch.utils.data import DataLoader
    train_ds = v4.CFRPDataset(train_df, args.data_dir, augment=True)
    test_ds = v4.CFRPDataset(test_df, args.data_dir, augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    print(f"\n=== Training physics_informed model (lambda_vr={args.lambda_vr}, "
          f"lambda_hs={args.lambda_hs}, full budget) ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    phys_model = v4.MicrostructureCNN(num_outputs=len(bounds.PROPERTY_NAMES))
    phys_model, history = v4.train_one_model(
        phys_model, train_loader, test_loader, scaler, device, physics_informed=True,
        vf_source="e11", lambda_vr=args.lambda_vr, lambda_hs=args.lambda_hs,
        vr_properties=vr_properties, max_epochs=args.max_epochs, patience=args.patience,
        tag="physics_informed")
    elapsed = time.time() - t0
    print(f"physics_informed trained in {elapsed/60:.1f} min")
    torch.save(phys_model.state_dict(), os.path.join(args.out_dir, "physics_informed_model.pth"))
    for h in history:
        h["model"] = "physics_informed"

    preds, labels, vf_otsu, _vf_e11 = run_inference_v4_style(phys_model, test_df, args.data_dir, scaler, device)
    rows, n_low_vf = v4.compute_metrics(preds, labels, vf_otsu, "physics_informed")
    phys_violations_pct = bounds.check_bound_violations(
        {name: preds[:, i] for i, name in enumerate(bounds.PROPERTY_NAMES)}, vf_otsu)

    full_perf = pd.concat([baseline_perf, pd.DataFrame(rows)], ignore_index=True)
    full_perf.to_csv(os.path.join(args.out_dir, "full_dataset_performance.csv"), index=False)

    phys_lowvf = pd.DataFrame(rows)[["model", "property", "RPE_full", "RPE_lowVf"]]
    full_lowvf = pd.concat([baseline_lowvf, phys_lowvf], ignore_index=True)
    full_lowvf.to_csv(os.path.join(args.out_dir, "lowvf_comparison.csv"), index=False)

    phys_violations_rows = [{"model": "physics_informed", "property": k, "violation_pct": v}
                             for k, v in phys_violations_pct.items()]
    full_violations = pd.concat([baseline_violations, pd.DataFrame(phys_violations_rows)], ignore_index=True)
    full_violations.to_csv(os.path.join(args.out_dir, "bound_violations.csv"), index=False)

    full_curves = pd.concat([baseline_curves, pd.DataFrame(history)], ignore_index=True)
    full_curves.to_csv(os.path.join(args.out_dir, "training_curves.csv"), index=False)

    base_low = baseline_perf["RPE_lowVf"].mean()
    pinn_low = pd.DataFrame(rows)["RPE_lowVf"].mean()
    base_full = baseline_perf["RPE_full"].mean()
    pinn_full = pd.DataFrame(rows)["RPE_full"].mean()

    summary = {
        "n_train": len(train_df), "n_test": len(test_df), "vf_source": "e11",
        "vr_properties": sorted(vr_properties),
        "baseline_violations_pct": baseline_violations_pct,
        "baseline_n_low_vf": baseline_n_low_vf,
        "baseline_train_minutes": baseline_train_minutes,
        "baseline_reused_from": args.baseline_results_dir,
        "physics_informed_violations_pct": phys_violations_pct,
        "physics_informed_n_low_vf": n_low_vf,
        "physics_informed_train_minutes": elapsed / 60,
        "lambda_vr_final": args.lambda_vr, "lambda_hs_final": args.lambda_hs,
        "mean_RPE_reduction_lowVf_pct": 100 * (base_low - pinn_low) / base_low if base_low else None,
        "mean_RPE_reduction_full_pct": 100 * (base_full - pinn_full) / base_full if base_full else None,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Results written to", args.out_dir, "(baseline reused, not retrained)")
    print(json.dumps(summary, indent=2))


def run_inference_v4_style(model, test_df, data_dir, scaler, device):
    """Same as evaluate_tail_errors_v4.run_inference but also returns vf_e11 (unused
    here, kept for signature parity since v4.CFRPDataset yields 4 items per sample)."""
    from torch.utils.data import DataLoader
    ds = v4.CFRPDataset(test_df, data_dir, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    model.eval().to(device)
    all_preds, all_labels, all_vf_otsu, all_vf_e11 = [], [], [], []
    with torch.no_grad():
        for images, labels, vf_otsu, vf_e11 in loader:
            images = images.to(device)
            outputs_scaled = model(images).cpu().numpy()
            outputs_phys = outputs_scaled * scaler.scale_ + scaler.mean_
            all_preds.append(outputs_phys)
            all_labels.append(labels.numpy())
            all_vf_otsu.append(vf_otsu.numpy())
            all_vf_e11.append(vf_e11.numpy())
    return (np.concatenate(all_preds), np.concatenate(all_labels),
            np.concatenate(all_vf_otsu), np.concatenate(all_vf_e11))


if __name__ == "__main__":
    main()
