"""
First physics-informed CNN for directional permeability (mu1, mu2, mu3), mirroring
the elastic-property pipeline (train_eval_v4.py) but with several deliberate
differences specific to permeability -- read this docstring before changing defaults.

DATA: Combined_Properties_with_FVF_updated.csv (in this folder) has REAL ground-truth
fiber_volume_fraction (percent, /100 to get the fraction) alongside mu1/mu2/mu3 and
the 8 elastic constants, for 10,042 RVE images. This sidesteps the whole Vf-estimation
problem that caused trouble in the elastic v2/v3/v4 scripts (no Otsu thresholding, no
deriving Vf from a label) -- just use the real column directly. Image filenames in
this CSV follow the pattern "image_<N>.csv" -> "image_<N>.png", which is a DIFFERENT
naming convention from properties.csv's random-string names used for the elastic-only
v2/v3/v4 runs -- confirm --data-dir points at the image folder that actually contains
files matching THIS naming pattern before running; it may not be the same folder you
used for train_eval_v4.py.

BOUNDS: see bounds_permeability.py. Two things were checked against the real ground
truth in this CSV before writing this script (not assumed):
  1. The HS formula already written in main.tex's Methodology section fails its own
     degenerate-limit check (computed: at Vf=1 it returns ~4.2e-17, not the fibre
     value 1.49e-18). bounds_permeability.py implements and self-tests the corrected
     two-phase scalar HS formula instead -- run `python bounds_permeability.py` to see
     the self-test pass before trusting it.
  2. With the corrected formula, mu1 and mu2 (transverse directions) violate the HS
     bound only 0.12%/0.22% of the time across all 10,042 real FE samples -- excellent
     agreement. mu3 (longitudinal/along-fibre direction) violates the SAME HS bound
     100% of the time, systematically, at every fiber volume fraction -- not a minor
     high-Vf effect. This means the isotropic-cross-section HS treatment (d=2) used
     here, while appropriate for mu1/mu2, is not a valid physical model for axial flow
     along continuous fibres (a fundamentally different transport regime: fluid moves
     through matrix channels parallel to unbroken fibre lengths, not around obstacles
     in cross-section). The looser Wiener bound, by contrast, holds for mu3 (0.01%
     violation), so mu3 keeps the Wiener penalty but is excluded from the HS penalty
     by default -- this is the permeability analogue of excluding nu12/nu13/nu23 from
     the elastic VR penalty in train_eval_v4.py: a case where a bound formula simply
     does not apply to a specific output, found by checking, not assumed.

TARGET SPACE: mu1/mu2/mu3 span roughly 1.5e-18 to 1.3e-17 (about one order of
magnitude) in this dataset. --target-space log (default) trains on log10(mu) rather
than raw mu, which is the standard choice for transport/diffusion properties and
keeps the regression targets in a well-scaled O(1) range. IMPORTANT: the bound penalty
is also computed in log10 space when --target-space log (i.e. compare log10(prediction)
against log10(bound)) -- applying the penalty on raw linear values around 1e-18 would
make the squared-hinge terms ~1e-36 in magnitude, utterly negligible against an O(1)
data loss regardless of lambda. This is analogous to (but a different mechanism than)
the elastic E11 circularity bug in v3/v4: a scale mismatch can silently make a penalty
term inert just as effectively as a logic bug can.

HOW TO RUN (PowerShell, same agenticCNN env):
    python train_eval_permeability_v1.py --data-dir "C:\\path\\to\\permeability\\images" `
        --csv Combined_Properties_with_FVF_updated.csv --out-dir results_permeability `
        --lambda-grid 0.1,0.5,1,2 --scan-max-epochs 80 --scan-patience 15

OUTPUT (--out-dir): full_dataset_performance.csv, lowvf_comparison.csv,
bound_violations.csv, summary.json, training_curves.csv, config_scan.csv,
baseline_model.pth, physics_informed_model.pth -- same structure as the elastic
scripts, but for mu1/mu2/mu3 instead of the 8 elastic constants.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import bounds_permeability as bp

SEED = 42
LOW_VF_THRESHOLD = 0.45
IMAGE_SIZE = 93


class PermeabilityDataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.base_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Input_File"].replace(".csv", ".png"))
        image = Image.open(img_path).convert("RGB")

        if self.augment:
            if np.random.rand() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            rot = np.random.choice([0, 90, 180, 270])
            if rot:
                image = image.rotate(rot)

        image_t = self.base_transform(image)
        labels = row[bp.PERMEABILITY_NAMES].values.astype(np.float32)
        vf = float(row["vf_fraction"])
        return image_t, torch.tensor(labels), torch.tensor(vf, dtype=torch.float32)


class PermeabilityCNN(nn.Module):
    """Same backbone family as MicrostructureCNN (bounds.py/train_eval_v4.py), 3
    outputs instead of 8."""

    def __init__(self, num_outputs=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        return self.regressor(self.features(x))


def compute_penalty_terms(outputs_phys_or_log, vf_batch, target_space,
                           vr_properties, hs_properties):
    """outputs_phys_or_log: predictions already inverse-transformed from the
    StandardScaler, in log10(mu) units if target_space=='log', else raw mu units.
    Bounds are computed in raw mu units from bounds_permeability, then converted to
    the same space as outputs_phys_or_log before the hinge penalty (see module
    docstring -- comparing in mismatched scales silently zeroes the gradient)."""
    device = outputs_phys_or_log.device
    vf_np = vf_batch.detach().cpu().numpy()
    voigt, reuss = bp.voigt_reuss_bounds_permeability(vf_np)
    w_up_raw, w_lo_raw = np.maximum(voigt, reuss), np.minimum(voigt, reuss)
    hs_up_raw, hs_lo_raw = bp.hashin_shtrikman_bounds_permeability(vf_np)

    if target_space == "log":
        w_up, w_lo = np.log10(w_up_raw), np.log10(w_lo_raw)
        hs_up, hs_lo = np.log10(hs_up_raw), np.log10(hs_lo_raw)
    else:
        w_up, w_lo, hs_up, hs_lo = w_up_raw, w_lo_raw, hs_up_raw, hs_lo_raw

    w_up_t = torch.tensor(w_up, device=device, dtype=torch.float32)
    w_lo_t = torch.tensor(w_lo, device=device, dtype=torch.float32)
    hs_up_t = torch.tensor(hs_up, device=device, dtype=torch.float32)
    hs_lo_t = torch.tensor(hs_lo, device=device, dtype=torch.float32)

    loss_wiener = torch.tensor(0.0, device=device)
    loss_hs = torch.tensor(0.0, device=device)
    for i, name in enumerate(bp.PERMEABILITY_NAMES):
        pred = outputs_phys_or_log[:, i]
        if name in vr_properties:
            loss_wiener = loss_wiener + torch.mean(
                torch.clamp(pred - w_up_t, min=0) ** 2 + torch.clamp(w_lo_t - pred, min=0) ** 2
            )
        if name in hs_properties:
            loss_hs = loss_hs + torch.mean(
                torch.clamp(pred - hs_up_t, min=0) ** 2 + torch.clamp(hs_lo_t - pred, min=0) ** 2
            )

    n_w = max(len(vr_properties), 1)
    n_h = max(len(hs_properties), 1)
    return loss_wiener / n_w, loss_hs / n_h


def train_one_model(model, train_loader, val_loader, scaler, device, physics_informed,
                     target_space="log", lambda_wiener=0.2, lambda_hs=0.2,
                     vr_properties=("mu1", "mu2", "mu3"), hs_properties=("mu1", "mu2"),
                     max_epochs=300, patience=40, lr=1e-3, tag=""):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=10)
    mse = nn.MSELoss()

    scale_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale_std = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    def inverse_transform(y_scaled):
        return y_scaled * scale_std + scale_mean

    def to_target_space(labels_phys_raw):
        """labels_phys_raw: real mu values. Convert to log10 if target_space=='log'."""
        return torch.log10(labels_phys_raw) if target_space == "log" else labels_phys_raw

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        n_batches = len(train_loader)
        for batch_idx, (images, labels, vfs) in enumerate(train_loader):
            images, labels, vfs = images.to(device), labels.to(device), vfs.to(device)
            labels_space = to_target_space(labels)
            labels_scaled = (labels_space - scale_mean) / scale_std

            optimizer.zero_grad()
            outputs_scaled = model(images)
            data_loss = mse(outputs_scaled, labels_scaled)

            if physics_informed:
                outputs_space = inverse_transform(outputs_scaled)  # log10(mu) or mu
                loss_w, loss_h = compute_penalty_terms(
                    outputs_space, vfs, target_space, vr_properties, hs_properties)
                loss = data_loss + lambda_wiener * loss_w + lambda_hs * loss_h
            else:
                loss = data_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            if batch_idx == 0 or (batch_idx + 1) % 50 == 0 or batch_idx + 1 == n_batches:
                print(f"  [{tag}] epoch {epoch+1} batch {batch_idx+1}/{n_batches} "
                      f"running_loss={loss.item():.5f}", flush=True)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, vfs in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels_space = to_target_space(labels)
                labels_scaled = (labels_space - scale_mean) / scale_std
                outputs_scaled = model(images)
                loss = mse(outputs_scaled, labels_scaled)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{tag}] epoch {epoch+1}: train={train_loss:.5f} val={val_loss:.5f}", flush=True)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_all(model, loader, scaler, device, target_space):
    model.eval()
    all_preds, all_labels, all_vf = [], [], []
    for images, labels, vfs in loader:
        images = images.to(device)
        outputs_scaled = model(images).cpu().numpy()
        outputs_space = outputs_scaled * scaler.scale_ + scaler.mean_
        outputs_raw = (10 ** outputs_space) if target_space == "log" else outputs_space
        all_preds.append(outputs_raw)
        all_labels.append(labels.numpy())
        all_vf.append(vfs.numpy())
    return (np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_vf))


def compute_metrics(preds, labels, vf, label):
    """preds/labels always in raw mu units regardless of target_space (predict_all
    already converts back), so metrics are comparable across target-space choices."""
    rows = []
    low_mask = vf < LOW_VF_THRESHOLD
    for i, name in enumerate(bp.PERMEABILITY_NAMES):
        p, y = preds[:, i], labels[:, i]
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        mape = float(np.mean(np.abs((p - y) / y)) * 100)
        r2 = float(r2_score(y, p))
        rpe_full = mape
        if low_mask.sum() > 0:
            rpe_low = float(np.mean(np.abs((p[low_mask] - y[low_mask]) / y[low_mask])) * 100)
        else:
            rpe_low = float("nan")
        rows.append({"model": label, "property": name, "R2": r2, "RMSE": rmse,
                     "MAPE": mape, "RPE_full": rpe_full, "RPE_lowVf": rpe_low})
    return rows, int(low_mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--csv", default="Combined_Properties_with_FVF_updated.csv")
    ap.add_argument("--out-dir", default="results_permeability")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lambda-wiener", type=float, default=0.2)
    ap.add_argument("--lambda-hs", type=float, default=0.2)
    ap.add_argument("--lambda-grid", type=str, default="",
                     help="Comma list, e.g. '0.1,0.5,1,2'. Applies the same value to "
                          "both lambda-wiener and lambda-hs for each candidate.")
    ap.add_argument("--scan-max-epochs", type=int, default=80)
    ap.add_argument("--scan-patience", type=int, default=15)
    ap.add_argument("--target-space", choices=["log", "linear"], default="log")
    ap.add_argument("--vr-properties", type=str, default="mu1,mu2,mu3",
                     help="Properties penalized by the Wiener (Voigt-Reuss) bound.")
    ap.add_argument("--hs-properties", type=str, default="mu1,mu2",
                     help="Properties penalized by the HS bound. Default excludes mu3 "
                          "-- see module docstring (the d=2 HS bound does not hold for "
                          "the longitudinal direction; verified against ground truth).")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    vr_properties = set(p.strip() for p in args.vr_properties.split(",") if p.strip())
    hs_properties = set(p.strip() for p in args.hs_properties.split(",") if p.strip())
    for s, label in [(vr_properties, "vr"), (hs_properties, "hs")]:
        unknown = s - set(bp.PERMEABILITY_NAMES)
        if unknown:
            raise ValueError(f"--{label}-properties has unknown names: {unknown}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"target-space={args.target_space}  vr-properties={sorted(vr_properties)}  "
          f"hs-properties={sorted(hs_properties)}")

    df = pd.read_csv(args.csv)
    # The CSV uses mu-with-special-character column names (copy-pasted from a
    # spreadsheet) -- normalize to plain ASCII 'mu1'/'mu2'/'mu3' used throughout this
    # script and bounds_permeability.py, and add a plain 'vf_fraction' column (the CSV
    # stores fiber_volume_fraction as a 0-100 percentage, not a 0-1 fraction).
    rename_map = {}
    for col in df.columns:
        if col.strip().lower().lstrip("µμ") in ("1", "2", "3") and len(col.strip()) <= 3:
            rename_map[col] = "mu" + col.strip()[-1]
    df = df.rename(columns=rename_map)
    missing = [c for c in bp.PERMEABILITY_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Could not find permeability columns {missing} after rename. "
            f"Found columns: {list(df.columns)}. The CSV's mu-column names may use a "
            f"different special character than expected -- check manually and adjust "
            f"rename_map above if so."
        )
    df["vf_fraction"] = df["fiber_volume_fraction"] / 100.0

    # The CSV may reference more images than actually exist in --data-dir (e.g. rows
    # added in an updated version of the CSV without matching images copied over
    # yet). Skip those rows rather than crashing mid-training -- log exactly how many
    # and a few examples, so a large skip count gets noticed rather than silently
    # shrinking the dataset.
    def _image_exists(input_file):
        img_path = os.path.join(args.data_dir, input_file.replace(".csv", ".png"))
        return os.path.isfile(img_path)

    n_before = len(df)
    print(f"Checking {n_before} rows against image files in {args.data_dir} ...", flush=True)
    exists_mask = df["Input_File"].apply(_image_exists)
    missing_examples = df.loc[~exists_mask, "Input_File"].head(5).tolist()
    df = df.loc[exists_mask].reset_index(drop=True)
    n_skipped = n_before - len(df)
    if n_skipped:
        print(f"Skipping {n_skipped} of {n_before} rows -- image file not found in "
              f"{args.data_dir} (examples: {missing_examples})")
    if len(df) == 0:
        raise FileNotFoundError(
            f"No CSV rows matched an existing image file in {args.data_dir}. Check "
            f"--data-dir points at the folder containing image_<N>.png files."
        )
    print(f"Usable rows after filtering for existing images: {len(df)}")

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    split = int(len(df) * 0.7)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    print(f"Train: {len(train_df)}  Test/val: {len(test_df)}")
    print(f"Vf range in test set: {test_df['vf_fraction'].min():.3f} - {test_df['vf_fraction'].max():.3f}")

    target_space = args.target_space
    train_targets = train_df[bp.PERMEABILITY_NAMES].values
    if target_space == "log":
        train_targets = np.log10(train_targets)
    scaler = StandardScaler()
    scaler.fit(train_targets)

    train_ds = PermeabilityDataset(train_df, args.data_dir, augment=True)
    test_ds = PermeabilityDataset(test_df, args.data_dir, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_history = []
    results_rows = []
    summary = {"n_train": len(train_df), "n_test": len(test_df),
               "target_space": target_space, "vr_properties": sorted(vr_properties),
               "hs_properties": sorted(hs_properties)}

    print("\n=== Training baseline model ===")
    t0 = time.time()
    base_model = PermeabilityCNN(num_outputs=len(bp.PERMEABILITY_NAMES))
    base_model, history = train_one_model(
        base_model, train_loader, test_loader, scaler, device, physics_informed=False,
        target_space=target_space, max_epochs=args.max_epochs, patience=args.patience,
        tag="baseline")
    elapsed = time.time() - t0
    print(f"baseline trained in {elapsed/60:.1f} min")
    torch.save(base_model.state_dict(), os.path.join(args.out_dir, "baseline_model.pth"))
    for h in history:
        h["model"] = "baseline"
    all_history.extend(history)
    preds, labels, vf = predict_all(base_model, test_loader, scaler, device, target_space)
    rows, n_low_vf = compute_metrics(preds, labels, vf, "baseline")
    results_rows.extend(rows)
    summary["baseline_violations_wiener_pct"] = bp.check_bound_violations_permeability(
        {name: preds[:, i] for i, name in enumerate(bp.PERMEABILITY_NAMES)}, vf)
    summary["baseline_n_low_vf"] = n_low_vf
    summary["baseline_train_minutes"] = elapsed / 60
    base_low_rpe = np.mean([r["RPE_lowVf"] for r in rows])

    scan_rows = []
    if args.lambda_grid:
        candidates = [float(x) for x in args.lambda_grid.split(",") if x.strip()]
        print(f"\n=== Scanning {len(candidates)} lambda candidates "
              f"(scan budget: {args.scan_max_epochs} epochs, patience {args.scan_patience}) ===")
        for lam in candidates:
            t0 = time.time()
            m = PermeabilityCNN(num_outputs=len(bp.PERMEABILITY_NAMES))
            m, _ = train_one_model(
                m, train_loader, test_loader, scaler, device, physics_informed=True,
                target_space=target_space, lambda_wiener=lam, lambda_hs=lam,
                vr_properties=vr_properties, hs_properties=hs_properties,
                max_epochs=args.scan_max_epochs, patience=args.scan_patience,
                tag=f"scan-lam{lam}")
            scan_elapsed = time.time() - t0
            p, l, vfo = predict_all(m, test_loader, scaler, device, target_space)
            r, _ = compute_metrics(p, l, vfo, f"scan_lam{lam}")
            mean_low_rpe = float(np.mean([x["RPE_lowVf"] for x in r]))
            scan_rows.append({"lambda": lam, "mean_RPE_lowVf": mean_low_rpe,
                               "minutes": scan_elapsed / 60})
            print(f"  lambda={lam}: mean low-Vf RPE = {mean_low_rpe:.3f}% "
                  f"(baseline = {base_low_rpe:.3f}%), {scan_elapsed/60:.1f} min")

        pd.DataFrame(scan_rows).to_csv(os.path.join(args.out_dir, "config_scan.csv"), index=False)
        best = min(scan_rows, key=lambda r: r["mean_RPE_lowVf"])
        best_lambda = best["lambda"]
        summary["config_scan"] = scan_rows
        summary["best_lambda"] = best_lambda
        print(f"\nBest candidate: lambda={best_lambda} "
              f"(mean low-Vf RPE {best['mean_RPE_lowVf']:.3f}% vs baseline {base_low_rpe:.3f}%)")
        lambda_w_final = lambda_h_final = best_lambda
    else:
        lambda_w_final, lambda_h_final = args.lambda_wiener, args.lambda_hs
        summary["lambda_wiener"] = lambda_w_final
        summary["lambda_hs"] = lambda_h_final

    print("\n=== Training physics_informed model (final, full budget) ===")
    t0 = time.time()
    phys_model = PermeabilityCNN(num_outputs=len(bp.PERMEABILITY_NAMES))
    phys_model, history = train_one_model(
        phys_model, train_loader, test_loader, scaler, device, physics_informed=True,
        target_space=target_space, lambda_wiener=lambda_w_final, lambda_hs=lambda_h_final,
        vr_properties=vr_properties, hs_properties=hs_properties,
        max_epochs=args.max_epochs, patience=args.patience, tag="physics_informed")
    elapsed = time.time() - t0
    print(f"physics_informed trained in {elapsed/60:.1f} min")
    torch.save(phys_model.state_dict(), os.path.join(args.out_dir, "physics_informed_model.pth"))
    for h in history:
        h["model"] = "physics_informed"
    all_history.extend(history)

    preds, labels, vf = predict_all(phys_model, test_loader, scaler, device, target_space)
    rows, n_low_vf = compute_metrics(preds, labels, vf, "physics_informed")
    results_rows.extend(rows)
    summary["physics_informed_violations_wiener_pct"] = bp.check_bound_violations_permeability(
        {name: preds[:, i] for i, name in enumerate(bp.PERMEABILITY_NAMES)}, vf)
    summary["physics_informed_n_low_vf"] = n_low_vf
    summary["physics_informed_train_minutes"] = elapsed / 60
    summary["lambda_wiener_final"] = lambda_w_final
    summary["lambda_hs_final"] = lambda_h_final

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(os.path.join(args.out_dir, "full_dataset_performance.csv"), index=False)

    lowvf_df = results_df[["model", "property", "RPE_full", "RPE_lowVf"]]
    lowvf_df.to_csv(os.path.join(args.out_dir, "lowvf_comparison.csv"), index=False)

    violations_rows = []
    for label in ["baseline", "physics_informed"]:
        for name, pct in summary[f"{label}_violations_wiener_pct"].items():
            violations_rows.append({"model": label, "property": name, "violation_pct": pct})
    pd.DataFrame(violations_rows).to_csv(os.path.join(args.out_dir, "bound_violations.csv"), index=False)

    pd.DataFrame(all_history).to_csv(os.path.join(args.out_dir, "training_curves.csv"), index=False)

    base_low = results_df[(results_df.model == "baseline")]["RPE_lowVf"].mean()
    pinn_low = results_df[(results_df.model == "physics_informed")]["RPE_lowVf"].mean()
    base_full = results_df[(results_df.model == "baseline")]["RPE_full"].mean()
    pinn_full = results_df[(results_df.model == "physics_informed")]["RPE_full"].mean()
    summary["mean_RPE_reduction_lowVf_pct"] = 100 * (base_low - pinn_low) / base_low if base_low else None
    summary["mean_RPE_reduction_full_pct"] = 100 * (base_full - pinn_full) / base_full if base_full else None

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Results written to", args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
