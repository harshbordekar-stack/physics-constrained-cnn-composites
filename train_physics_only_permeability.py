"""
Permeability counterpart to train_physics_only_v4.py: trains ONLY a new
physics-informed permeability model at a specified lambda, reusing the existing
baseline from a completed train_eval_permeability_v1.py run instead of retraining it
(the baseline doesn't depend on lambda or vr/hs-properties at all -- same data, same
seed, retraining it would reproduce an identical model).

WHY mu3 IS STILL EXCLUDED FROM THE HS PENALTY BY DEFAULT (not the same issue as E11):
Vf for permeability comes from the real fiber_volume_fraction column in
Combined_Properties_with_FVF_updated.csv -- it is not derived from mu1/mu2/mu3, so
there is no E11-style circularity here at all. mu3 (longitudinal/along-fibre
permeability) is excluded from the HS penalty for a separate, already-confirmed
reason: the real ground truth violates the d=2 Hashin-Shtrikman bound 100% of the
time at every Vf (checked directly against all 10,042 samples before train_eval_
permeability_v1.py was written), because axial flow along continuous fibres is a
different transport regime than the cross-sectional flow the d=2 HS treatment
models. mu3 remains in the (looser) Wiener penalty, which it satisfies fine. This is
the default in train_eval_permeability_v1.py already and is unchanged here.

HOW TO RUN (PowerShell, same env):
    python train_physics_only_permeability.py --data-dir "C:\\path\\to\\permeability\\images" `
        --csv Combined_Properties_with_FVF_updated.csv `
        --baseline-results-dir results_permeability `
        --out-dir results_permeability_lambda025 --lambda-wiener 0.25 --lambda-hs 0.25
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
from torch.utils.data import DataLoader

import bounds_permeability as bp
import train_eval_permeability_v1 as pv1

SEED = pv1.SEED


def rebuild_permeability_split(csv_path, data_dir):
    """Exactly reproduces train_eval_permeability_v1.py's main() preprocessing --
    same mu-column rename, same vf_fraction column, same image-existence filter, same
    shuffle seed, same 70/30 split, same scaler fit (in log10 space) on the same train
    rows."""
    df = pd.read_csv(csv_path)
    rename_map = {}
    for col in df.columns:
        if col.strip().lower().lstrip("µμ") in ("1", "2", "3") and len(col.strip()) <= 3:
            rename_map[col] = "mu" + col.strip()[-1]
    df = df.rename(columns=rename_map)
    missing = [c for c in bp.PERMEABILITY_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing {missing} after rename. Found: {list(df.columns)}")
    df["vf_fraction"] = df["fiber_volume_fraction"] / 100.0

    def _image_exists(input_file):
        return os.path.isfile(os.path.join(data_dir, input_file.replace(".csv", ".png")))

    n_before = len(df)
    exists_mask = df["Input_File"].apply(_image_exists)
    df = df.loc[exists_mask].reset_index(drop=True)
    print(f"Usable rows after filtering for existing images: {len(df)} (of {n_before})")

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    split = int(len(df) * 0.7)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    train_targets = train_df[bp.PERMEABILITY_NAMES].values
    train_targets_log = np.log10(train_targets)
    scaler = StandardScaler()
    scaler.fit(train_targets_log)
    return train_df, test_df, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--csv", default="Combined_Properties_with_FVF_updated.csv")
    ap.add_argument("--baseline-results-dir", default="results_permeability")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lambda-wiener", type=float, required=True)
    ap.add_argument("--lambda-hs", type=float, required=True)
    ap.add_argument("--vr-properties", type=str, default="mu1,mu2,mu3")
    ap.add_argument("--hs-properties", type=str, default="mu1,mu2",
                     help="Default excludes mu3 -- see module docstring. This is NOT "
                          "the E11-style circularity issue; Vf here is real, not "
                          "label-derived. mu3 is excluded because the d=2 HS bound "
                          "does not hold for axial flow (confirmed against ground "
                          "truth), independent of how Vf is obtained.")
    ap.add_argument("--target-space", choices=["log", "linear"], default="log")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    args = ap.parse_args()

    vr_properties = set(p.strip() for p in args.vr_properties.split(",") if p.strip())
    hs_properties = set(p.strip() for p in args.hs_properties.split(",") if p.strip())
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"vr-properties={sorted(vr_properties)}  hs-properties={sorted(hs_properties)}")

    baseline_pth = os.path.join(args.baseline_results_dir, "baseline_model.pth")
    if not os.path.isfile(baseline_pth):
        raise FileNotFoundError(f"{baseline_pth} not found.")
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

    print("Rebuilding the exact permeability train/test split (must match the reused baseline)...")
    train_df, test_df, scaler = rebuild_permeability_split(args.csv, args.data_dir)
    train_ds = pv1.PermeabilityDataset(train_df, args.data_dir, augment=True)
    test_ds = pv1.PermeabilityDataset(test_df, args.data_dir, augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    print(f"\n=== Training physics_informed model (lambda_wiener={args.lambda_wiener}, "
          f"lambda_hs={args.lambda_hs}) ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    phys_model = pv1.PermeabilityCNN(num_outputs=len(bp.PERMEABILITY_NAMES))
    phys_model, history = pv1.train_one_model(
        phys_model, train_loader, test_loader, scaler, device, physics_informed=True,
        target_space=args.target_space, lambda_wiener=args.lambda_wiener,
        lambda_hs=args.lambda_hs, vr_properties=vr_properties, hs_properties=hs_properties,
        max_epochs=args.max_epochs, patience=args.patience, tag="physics_informed")
    elapsed = time.time() - t0
    print(f"physics_informed trained in {elapsed/60:.1f} min")
    torch.save(phys_model.state_dict(), os.path.join(args.out_dir, "physics_informed_model.pth"))
    for h in history:
        h["model"] = "physics_informed"

    preds, labels, vf = pv1.predict_all(phys_model, test_loader, scaler, device, args.target_space)
    rows, n_low_vf = pv1.compute_metrics(preds, labels, vf, "physics_informed")
    phys_violations_pct = bp.check_bound_violations_permeability(
        {name: preds[:, i] for i, name in enumerate(bp.PERMEABILITY_NAMES)}, vf)

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
        "n_train": len(train_df), "n_test": len(test_df),
        "target_space": args.target_space,
        "vr_properties": sorted(vr_properties), "hs_properties": sorted(hs_properties),
        "baseline_violations_wiener_pct": old_summary.get("baseline_violations_wiener_pct"),
        "baseline_n_low_vf": old_summary.get("baseline_n_low_vf"),
        "baseline_train_minutes": old_summary.get("baseline_train_minutes"),
        "baseline_reused_from": args.baseline_results_dir,
        "physics_informed_violations_wiener_pct": phys_violations_pct,
        "physics_informed_n_low_vf": n_low_vf,
        "physics_informed_train_minutes": elapsed / 60,
        "lambda_wiener_final": args.lambda_wiener, "lambda_hs_final": args.lambda_hs,
        "mean_RPE_reduction_lowVf_pct": 100 * (base_low - pinn_low) / base_low if base_low else None,
        "mean_RPE_reduction_full_pct": 100 * (base_full - pinn_full) / base_full if base_full else None,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Results written to", args.out_dir, "(baseline reused, not retrained)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
