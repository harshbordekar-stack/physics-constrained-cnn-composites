"""
v4: fixes a real bug found in the v3 results, and adds an alternative loss
parameterization the user asked about.

WHAT v3 GOT WRONG (diagnosed from the actual results_v3 run, not a guess):
  v3 derives Vf from the E11 label (Vf = (E11-Em)/(Ef1-Em)) and used it to penalize
  ALL five stiffness/shear properties, including E11 itself. That's circular for E11
  specifically: plugging that same Vf back into the Voigt formula for E11 gives back
  (almost exactly) the true E11 label, so the "upper bound" being enforced was
  essentially "don't predict higher than the answer." This badly distorted E11
  training (R^2 dropped from 0.94 to 0.82 in the v3 run) while the other seven
  properties each showed a small genuine improvement in the targeted low-Vf region --
  so the fix is narrow, not a reason to abandon the approach.

  FIX: when --vf-source e11, E11 is automatically dropped from the Voigt-Reuss penalty
  (no circularity issue for E22/E33/G12/G13, since their bounds don't collapse onto
  their own label the way E11's does). You can override this with --vr-properties if
  you explicitly want E11 back in (a warning will print, since you'd be reintroducing
  the known bug).

NEW: --loss-mode {additive, convex}
  additive (default, same as v2/v3): loss = data_loss + lambda_vr*VR + lambda_hs*HS
  convex (new, per a paper the user read): loss = (1-factor)*data_loss + factor*(VR+HS)
  These are the same family mathematically (lambda = factor/(1-factor) converts one to
  the other) EXCEPT for one real difference: convex mode also shrinks the data-loss
  weight as factor grows, deliberately trading off supervision strength for physics
  consistency. Since every training sample here already has a ground-truth FE label,
  that tradeoff is probably not what you want -- but it's now easy to try and compare
  for yourself rather than take that on faith. --factor-grid sweeps this the same way
  --lambda-grid sweeps lambda in additive mode.

HOW TO RUN (same pattern as v3, PowerShell):
    python train_eval_v4.py --data-dir "C:\\path\\to\\images" --csv properties.csv `
        --out-dir results_v4 --loss-mode additive --lambda-grid 0.1,0.2,0.5,1 `
        --scan-max-epochs 80 --scan-patience 15

    # or to try the convex form instead:
    python train_eval_v4.py --data-dir "C:\\path\\to\\images" --csv properties.csv `
        --out-dir results_v4_convex --loss-mode convex --factor-grid 0.1,0.3,0.5,0.7 `
        --scan-max-epochs 80 --scan-patience 15

OUTPUT: same files as v3 (full_dataset_performance.csv, lowvf_comparison.csv,
bound_violations.csv, summary.json, training_curves.csv, vf_estimates_sample.csv),
plus config_scan.csv (renamed from v3's lambda_scan.csv -- now logs whichever knob,
lambda or factor, was actually swept).
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import bounds

SEED = 42
LOW_VF_THRESHOLD = 0.45
IMAGE_SIZE = 93


def estimate_vf_otsu(pil_image):
    gray = np.array(pil_image.convert("L"))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.mean(thresh == 0))


def estimate_vf_from_e11(e11, vf_min=0.30, vf_max=0.75):
    vf = (e11 - bounds.EM) / (bounds.EF1 - bounds.EM)
    return float(np.clip(vf, vf_min, vf_max))


class CFRPDataset(Dataset):
    def __init__(self, df, img_dir, augment=False, vf_clip=(0.30, 0.75)):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.vf_clip = vf_clip
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

        vf_otsu = estimate_vf_otsu(image)
        vf_e11 = estimate_vf_from_e11(row["E11"], *self.vf_clip)

        if self.augment:
            if np.random.rand() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            rot = np.random.choice([0, 90, 180, 270])
            if rot:
                image = image.rotate(rot)

        image_t = self.base_transform(image)
        labels = row[bounds.PROPERTY_NAMES].values.astype(np.float32)
        return (image_t, torch.tensor(labels),
                torch.tensor(vf_otsu, dtype=torch.float32),
                torch.tensor(vf_e11, dtype=torch.float32))


class MicrostructureCNN(nn.Module):
    def __init__(self, num_outputs=8):
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


def compute_penalty_terms(outputs_phys, vf_batch, vr_properties):
    """Returns (loss_vr_normalized, loss_hs_normalized) UNWEIGHTED (no lambda/factor
    applied) so the caller can combine them additively or convexly. Normalized by the
    number of properties contributing to each term, same convention as v2/v3."""
    device = outputs_phys.device
    vf_np = vf_batch.detach().cpu().numpy()
    vr = bounds.voigt_reuss_bounds(vf_np)
    hs = bounds.hashin_shtrikman_bounds(vf_np)

    loss_vr = torch.tensor(0.0, device=device)
    loss_hs = torch.tensor(0.0, device=device)
    for i, name in enumerate(bounds.PROPERTY_NAMES):
        pred = outputs_phys[:, i]
        if name in vr_properties:
            v_up = torch.tensor(np.maximum(*vr[name]), device=device, dtype=torch.float32)
            v_lo = torch.tensor(np.minimum(*vr[name]), device=device, dtype=torch.float32)
            loss_vr = loss_vr + torch.mean(
                torch.clamp(pred - v_up, min=0) ** 2 + torch.clamp(v_lo - pred, min=0) ** 2
            )
        if name in hs:
            h_up = torch.tensor(np.maximum(*hs[name]), device=device, dtype=torch.float32)
            h_lo = torch.tensor(np.minimum(*hs[name]), device=device, dtype=torch.float32)
            loss_hs = loss_hs + torch.mean(
                torch.clamp(pred - h_up, min=0) ** 2 + torch.clamp(h_lo - pred, min=0) ** 2
            )

    n_vr = max(len(vr_properties), 1)
    return loss_vr / n_vr, loss_hs / len(bounds.HS_PROPERTIES)


def train_one_model(model, train_loader, val_loader, scaler, device, physics_informed,
                     vf_source="e11", loss_mode="additive",
                     lambda_vr=0.05, lambda_hs=0.05, factor=0.3,
                     vr_properties=bounds.PROPERTY_NAMES,
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

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for images, labels, vf_otsu, vf_e11 in train_loader:
            images, labels = images.to(device), labels.to(device)
            vf_for_penalty = (vf_e11 if vf_source == "e11" else vf_otsu).to(device)
            labels_scaled = (labels - scale_mean) / scale_std

            optimizer.zero_grad()
            outputs_scaled = model(images)
            data_loss = mse(outputs_scaled, labels_scaled)

            if physics_informed:
                outputs_phys = inverse_transform(outputs_scaled)
                loss_vr, loss_hs = compute_penalty_terms(outputs_phys, vf_for_penalty, vr_properties)
                if loss_mode == "convex":
                    physics_term = loss_vr + loss_hs
                    loss = (1 - factor) * data_loss + factor * physics_term
                else:  # additive
                    loss = data_loss + lambda_vr * loss_vr + lambda_hs * loss_hs
            else:
                loss = data_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, vf_otsu, vf_e11 in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels_scaled = (labels - scale_mean) / scale_std
                outputs_scaled = model(images)
                loss = mse(outputs_scaled, labels_scaled)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{tag}] epoch {epoch+1}: train={train_loss:.5f} val={val_loss:.5f}")

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
def predict_all(model, loader, scaler, device):
    model.eval()
    all_preds, all_labels, all_vf_otsu, all_vf_e11 = [], [], [], []
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


def compute_metrics(preds, labels, vf_otsu, label):
    rows = []
    low_mask = vf_otsu < LOW_VF_THRESHOLD
    for i, name in enumerate(bounds.PROPERTY_NAMES):
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
    ap.add_argument("--csv", default="properties.csv")
    ap.add_argument("--out-dir", default="results_v4")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--loss-mode", choices=["additive", "convex"], default="additive")
    ap.add_argument("--lambda-vr", type=float, default=0.2)
    ap.add_argument("--lambda-hs", type=float, default=0.2)
    ap.add_argument("--lambda-grid", type=str, default="",
                     help="Additive mode only. Comma list, e.g. '0.1,0.2,0.5,1'.")
    ap.add_argument("--factor", type=float, default=0.3)
    ap.add_argument("--factor-grid", type=str, default="",
                     help="Convex mode only. Comma list of values in (0,1), e.g. '0.1,0.3,0.5,0.7'.")
    ap.add_argument("--scan-max-epochs", type=int, default=80)
    ap.add_argument("--scan-patience", type=int, default=15)
    ap.add_argument("--vf-source", choices=["otsu", "e11"], default="e11")
    ap.add_argument("--vf-clip-min", type=float, default=0.30)
    ap.add_argument("--vf-clip-max", type=float, default=0.75)
    ap.add_argument("--vr-properties", type=str, default="",
                     help="Comma list. Default: E22,E33,G12,G13 if --vf-source e11 "
                          "(E11 excluded -- see module docstring), or "
                          "E11,E22,E33,G12,G13 if --vf-source otsu.")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if args.vr_properties:
        vr_properties = set(p.strip() for p in args.vr_properties.split(",") if p.strip())
        if "E11" in vr_properties and args.vf_source == "e11":
            print("WARNING: --vr-properties explicitly includes E11 while --vf-source is "
                  "e11 -- this reintroduces the circular-bound bug described in this "
                  "script's docstring (the E11 Voigt bound at an E11-derived Vf collapses "
                  "onto the E11 label itself). Proceeding because you asked for it explicitly.")
    elif args.vf_source == "e11":
        vr_properties = {"E22", "E33", "G12", "G13"}
    else:
        vr_properties = {"E11", "E22", "E33", "G12", "G13"}

    unknown = vr_properties - set(bounds.PROPERTY_NAMES)
    if unknown:
        raise ValueError(f"--vr-properties has unknown names: {unknown}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"loss-mode={args.loss_mode}  vf-source={args.vf_source}  vr-properties={sorted(vr_properties)}")

    df = pd.read_csv(args.csv)
    csv_to_canonical = {
        "E_11": "E11", "E_22": "E22", "E_33": "E33",
        "G_21": "G12", "G_31": "G13",
        "nu_12": "nu12", "nu_13": "nu13", "nu_23": "nu23",
    }
    df = df.rename(columns=csv_to_canonical)
    missing = [c for c in bounds.PROPERTY_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"properties.csv missing {missing} after rename. Found: {list(df.columns)}")
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    split = int(len(df) * 0.7)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    print(f"Train: {len(train_df)}  Test/val: {len(test_df)}")

    scaler = StandardScaler()
    scaler.fit(train_df[bounds.PROPERTY_NAMES].values)

    vf_clip = (args.vf_clip_min, args.vf_clip_max)
    train_ds = CFRPDataset(train_df, args.data_dir, augment=True, vf_clip=vf_clip)
    test_ds = CFRPDataset(test_df, args.data_dir, augment=False, vf_clip=vf_clip)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_history = []
    results_rows = []
    summary = {"n_train": len(train_df), "n_test": len(test_df),
               "vf_source": args.vf_source, "vr_properties": sorted(vr_properties),
               "loss_mode": args.loss_mode}

    print("\n=== Training baseline model ===")
    t0 = time.time()
    base_model = MicrostructureCNN(num_outputs=len(bounds.PROPERTY_NAMES))
    base_model, history = train_one_model(
        base_model, train_loader, test_loader, scaler, device, physics_informed=False,
        max_epochs=args.max_epochs, patience=args.patience, tag="baseline")
    elapsed = time.time() - t0
    print(f"baseline trained in {elapsed/60:.1f} min")
    torch.save(base_model.state_dict(), os.path.join(args.out_dir, "baseline_model.pth"))
    for h in history:
        h["model"] = "baseline"
    all_history.extend(history)
    preds, labels, vf_otsu, vf_e11 = predict_all(base_model, test_loader, scaler, device)
    rows, n_low_vf = compute_metrics(preds, labels, vf_otsu, "baseline")
    results_rows.extend(rows)
    summary["baseline_violations_pct"] = bounds.check_bound_violations(
        {name: preds[:, i] for i, name in enumerate(bounds.PROPERTY_NAMES)}, vf_otsu)
    summary["baseline_n_low_vf"] = n_low_vf
    summary["baseline_train_minutes"] = elapsed / 60
    base_low_rpe = np.mean([r["RPE_lowVf"] for r in rows])

    grid_str = args.lambda_grid if args.loss_mode == "additive" else args.factor_grid
    knob_name = "lambda" if args.loss_mode == "additive" else "factor"
    scan_rows = []
    if grid_str:
        candidates = [float(x) for x in grid_str.split(",") if x.strip()]
        print(f"\n=== Scanning {len(candidates)} {knob_name} candidates "
              f"(scan budget: {args.scan_max_epochs} epochs, patience {args.scan_patience}) ===")
        for val in candidates:
            t0 = time.time()
            m = MicrostructureCNN(num_outputs=len(bounds.PROPERTY_NAMES))
            kwargs = dict(vf_source=args.vf_source, vr_properties=vr_properties,
                          max_epochs=args.scan_max_epochs, patience=args.scan_patience,
                          loss_mode=args.loss_mode, tag=f"scan-{knob_name}{val}")
            if args.loss_mode == "additive":
                kwargs.update(lambda_vr=val, lambda_hs=val)
            else:
                kwargs.update(factor=val)
            m, _ = train_one_model(m, train_loader, test_loader, scaler, device,
                                    physics_informed=True, **kwargs)
            scan_elapsed = time.time() - t0
            p, l, vfo, _ = predict_all(m, test_loader, scaler, device)
            r, _ = compute_metrics(p, l, vfo, f"scan_{knob_name}{val}")
            mean_low_rpe = float(np.mean([x["RPE_lowVf"] for x in r]))
            scan_rows.append({knob_name: val, "mean_RPE_lowVf": mean_low_rpe,
                               "minutes": scan_elapsed / 60})
            print(f"  {knob_name}={val}: mean low-Vf RPE = {mean_low_rpe:.3f}% "
                  f"(baseline = {base_low_rpe:.3f}%), {scan_elapsed/60:.1f} min")

        pd.DataFrame(scan_rows).to_csv(os.path.join(args.out_dir, "config_scan.csv"), index=False)
        best = min(scan_rows, key=lambda r: r["mean_RPE_lowVf"])
        best_val = best[knob_name]
        summary["config_scan"] = scan_rows
        summary[f"best_{knob_name}"] = best_val
        print(f"\nBest candidate: {knob_name}={best_val} "
              f"(mean low-Vf RPE {best['mean_RPE_lowVf']:.3f}% vs baseline {base_low_rpe:.3f}%)")
        print(f"Retraining winner at full budget ({args.max_epochs} epochs, patience {args.patience})...")
        final_kwargs = dict(lambda_vr=best_val, lambda_hs=best_val) if args.loss_mode == "additive" \
            else dict(factor=best_val)
    else:
        final_kwargs = dict(lambda_vr=args.lambda_vr, lambda_hs=args.lambda_hs) \
            if args.loss_mode == "additive" else dict(factor=args.factor)
        summary.update({k: v for k, v in final_kwargs.items()})

    print("\n=== Training physics_informed model (final, full budget) ===")
    t0 = time.time()
    phys_model = MicrostructureCNN(num_outputs=len(bounds.PROPERTY_NAMES))
    phys_model, history = train_one_model(
        phys_model, train_loader, test_loader, scaler, device, physics_informed=True,
        vf_source=args.vf_source, vr_properties=vr_properties, loss_mode=args.loss_mode,
        max_epochs=args.max_epochs, patience=args.patience, tag="physics_informed",
        **final_kwargs)
    elapsed = time.time() - t0
    print(f"physics_informed trained in {elapsed/60:.1f} min")
    torch.save(phys_model.state_dict(), os.path.join(args.out_dir, "physics_informed_model.pth"))
    for h in history:
        h["model"] = "physics_informed"
    all_history.extend(history)

    preds, labels, vf_otsu, vf_e11 = predict_all(phys_model, test_loader, scaler, device)
    rows, n_low_vf = compute_metrics(preds, labels, vf_otsu, "physics_informed")
    results_rows.extend(rows)
    summary["physics_informed_violations_pct"] = bounds.check_bound_violations(
        {name: preds[:, i] for i, name in enumerate(bounds.PROPERTY_NAMES)}, vf_otsu)
    summary["physics_informed_n_low_vf"] = n_low_vf
    summary["physics_informed_train_minutes"] = elapsed / 60
    summary["final_config"] = final_kwargs

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(os.path.join(args.out_dir, "full_dataset_performance.csv"), index=False)

    lowvf_df = results_df[["model", "property", "RPE_full", "RPE_lowVf"]]
    lowvf_df.to_csv(os.path.join(args.out_dir, "lowvf_comparison.csv"), index=False)

    violations_rows = []
    for label in ["baseline", "physics_informed"]:
        for name, pct in summary[f"{label}_violations_pct"].items():
            violations_rows.append({"model": label, "property": name, "violation_pct": pct})
    pd.DataFrame(violations_rows).to_csv(os.path.join(args.out_dir, "bound_violations.csv"), index=False)

    pd.DataFrame(all_history).to_csv(os.path.join(args.out_dir, "training_curves.csv"), index=False)

    n_sample = min(200, len(vf_otsu))
    pd.DataFrame({
        "vf_otsu": vf_otsu[:n_sample], "vf_e11": vf_e11[:n_sample],
    }).to_csv(os.path.join(args.out_dir, "vf_estimates_sample.csv"), index=False)

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
