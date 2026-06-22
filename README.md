# Physics-Constrained CNN for Composite Microstructure Property Prediction

A CNN surrogate model for predicting homogenized elastic constants and directional
hydrogen permeability of CFRP (carbon-fiber-reinforced polymer) composites directly
from microstructure cross-section images, augmented with a physics-informed
**Analytical Constraint Loss** that penalizes predictions falling outside classical
analytical homogenization bounds (Voigt-Reuss, Hashin-Shtrikman).

This is the modeling component of a larger project (an agentic AI framework for
composite quality assurance) and is structured to be picked back up independently —
the trained models and code here are self-contained.

## What's in this repo

| File | Purpose |
|---|---|
| `bounds.py` | Analytical Voigt-Reuss / Hashin-Shtrikman bounds for the 8 elastic constants. Self-tests on import (`python bounds.py`). |
| `bounds_permeability.py` | Analytical Wiener / Hashin-Shtrikman bounds for directional permeability ($\mu_1,\mu_2,\mu_3$). Includes a corrected HS formula — see "Lessons learned" below. Self-tests on import. |
| `train_eval_v4.py` | Trains the baseline (plain MSE) and a physics-informed elastic-property CNN from scratch, with an optional lambda-grid scan. Produces `results_v4/`. |
| `train_physics_only_v4.py` | Trains **only** a new physics-informed elastic model at a specified penalty weight, reusing an existing baseline (no retraining — the baseline doesn't depend on lambda). This is what produced the final reported result, `results_v4_lambda025/`. |
| `evaluate_tail_errors_v4.py` | Inference-only analysis: loads two already-trained elastic models and checks performance specifically on the most severe low-fiber-volume-fraction (low-$V_f$) test images, where the constraint is designed to matter. Produces `tail_analysis_v4_lambda025/`. |
| `train_eval_permeability_v1.py` | Trains the baseline and physics-informed permeability CNN from scratch, with an optional lambda-grid scan. Produces `results_permeability/`. |
| `train_physics_only_permeability.py` | Permeability counterpart to `train_physics_only_v4.py`: trains only a new physics-informed permeability model at a specified penalty weight, reusing an existing baseline. This produced the final reported result, `results_permeability_lambda025/`. |
| `evaluate_tail_errors_permeability.py` | Permeability counterpart to `evaluate_tail_errors_v4.py`. Produces `tail_analysis_permeability_lambda025/`. |
| `properties.csv` *(not included — proprietary, see "Data you need to supply")* | Elastic-property dataset (`Input_File`, 8 elastic constants). No ground-truth $V_f$ column — see Lessons Learned. |
| `Combined_Properties_with_FVF_updated.csv` *(not included — proprietary, see "Data you need to supply")* | Combined dataset with **real ground-truth $V_f$**, permeability ($\mu_1,\mu_2,\mu_3$), and the 8 elastic constants. Different image-filename convention than `properties.csv` (see below). |
| `results_v4/` | Baseline model + a $\lambda=0.1$ physics-informed model (from an early lambda scan), both fully trained. |
| `results_v4_lambda025/` | The baseline (reused from `results_v4/`) + the final adopted elastic physics-informed model at $\lambda_{VR}=\lambda_{HS}=0.25$. **This is the headline elastic result.** |
| `tail_analysis_v4_lambda025/` | Per-sample predictions and the low-$V_f$-specific comparison for the elastic $\lambda=0.25$ models, including the figures used in the writeup. |
| `results_permeability/` | Baseline + an early-scan physics-informed permeability model at $\lambda=0.5$. |
| `results_permeability_lambda025/` | The baseline (reused) + the final adopted permeability physics-informed model at $\lambda_{\text{Wiener}}=\lambda_{HS}=0.25$, matching the elastic penalty weight. **This is the headline permeability result.** |
| `tail_analysis_permeability_lambda025/` | Per-sample predictions and the low-$V_f$-specific comparison for the permeability $\lambda=0.25$ models. |

## Setup

```bash
conda create -n phycnn python=3.10
conda activate phycnn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130  # match your CUDA driver, or use the cpu index if no GPU
pip install pandas numpy opencv-python pillow scikit-learn matplotlib
```

Verify GPU (optional but strongly recommended — CPU training is viable but slow):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Data you need to supply

**Neither the raw dataset CSVs nor the microstructure images are included in this
repo.** The CSVs (`properties.csv`, `Combined_Properties_with_FVF_updated.csv`) and
the per-sample `per_sample_results.csv` files that the tail-analysis scripts produce
contain proprietary ground-truth measurements and are gitignored -- if you re-run the
tail-analysis scripts yourself, the regenerated `per_sample_results.csv` will stay
local and untracked, not get committed. You'll need to supply, from wherever you keep
the proprietary dataset:

1. `properties.csv` and a matching image folder (random-string filenames like
   `4p5U9ijWr5.png`) — used by `train_eval_v4.py` / `train_physics_only_v4.py` /
   `evaluate_tail_errors_v4.py`.
2. `Combined_Properties_with_FVF_updated.csv` and a matching image folder
   (`image_<N>.png` filenames) — used by all the permeability scripts.

Each image is a $93\times93$ pixel RGB render of a CFRP cross-section RVE.

## How to reproduce the results

**Elastic properties** (PowerShell syntax; adjust for your shell):

```powershell
# Step 1: train baseline + an initial lambda scan from scratch (~2-3 hours)
python train_eval_v4.py --data-dir "C:\path\to\elastic\images" --csv properties.csv --out-dir results_v4 --lambda-grid 0.1,0.2,0.5,1 --scan-max-epochs 80 --scan-patience 15

# Step 2: train the final, adopted physics-informed model at lambda=0.25, reusing the baseline (~1 hour)
python train_physics_only_v4.py --data-dir "C:\path\to\elastic\images" --csv properties.csv --baseline-results-dir results_v4 --out-dir results_v4_lambda025 --lambda-vr 0.25 --lambda-hs 0.25

# Step 3: check whether the constraint actually helps where it's supposed to (a few minutes, inference only)
python evaluate_tail_errors_v4.py --data-dir "C:\path\to\elastic\images" --csv properties.csv --results-dir results_v4_lambda025 --out-dir tail_analysis_v4_lambda025
```

**Permeability** (same three-step pattern):

```powershell
# Step 1: train baseline + an initial lambda scan from scratch (~2-3 hours)
python train_eval_permeability_v1.py --data-dir "C:\path\to\permeability\images" --csv Combined_Properties_with_FVF_updated.csv --out-dir results_permeability --lambda-grid 0.1,0.5,1,2 --scan-max-epochs 80 --scan-patience 15

# Step 2: train the final, adopted physics-informed model at lambda=0.25, reusing the baseline (~1 hour)
python train_physics_only_permeability.py --data-dir "C:\path\to\permeability\images" --csv Combined_Properties_with_FVF_updated.csv --baseline-results-dir results_permeability --out-dir results_permeability_lambda025 --lambda-wiener 0.25 --lambda-hs 0.25

# Step 3: check whether the constraint actually helps where it's supposed to (a few minutes, inference only)
python evaluate_tail_errors_permeability.py --data-dir "C:\path\to\permeability\images" --csv Combined_Properties_with_FVF_updated.csv --results-dir results_permeability_lambda025 --out-dir tail_analysis_permeability_lambda025
```

Read the docstring at the top of each script before running — they explain design
decisions (which properties get the penalty and why, what `--vf-source` does, etc.)
in more detail than this README.

## Headline results

**Elastic, on the 60 most severe low-$V_f$ test images ($V_f=0.122$–$0.319$)** — this
is where the constraint is supposed to matter, and where the comparison should be
made, not the full dataset average (see Lessons Learned #1):

| Property | $R^2$ baseline | $R^2$ physics-informed | RPE baseline (%) | RPE physics-informed (%) |
|---|---|---|---|---|
| $E_{11}$ | 0.728 | 0.762 | 8.36 | 8.16 |
| $E_{22}$ | 0.686 | **0.761** | 2.20 | **1.92** |
| $E_{33}$ | 0.458 | **0.536** | 2.82 | **2.41** |
| $G_{12}$ | 0.605 | 0.614 | 6.46 | 6.44 |
| $G_{13}$ | 0.134 | 0.130 | 9.06 | 10.59 |
| $\nu_{12}$ | 0.446 | 0.399 | 1.29 | 1.36 |
| $\nu_{13}$ | 0.337 | 0.376 | 1.87 | 1.74 |
| $\nu_{23}$ | 0.303 | **0.403** | 1.62 | **1.41** |

6 of 8 properties improve; $G_{13}$ is a consistent exception (cause unconfirmed).
On the full test set the two models are essentially equivalent — the improvement is
real but concentrated, not a blanket gain.

**Permeability, on the 50 most severe low-$V_f$ test images ($V_f=0.096$–$0.302$)**,
same penalty weight ($\lambda=0.25$) and same method as elastic:

| Property | $R^2$ baseline | $R^2$ physics-informed | RPE baseline (%) | RPE physics-informed (%) |
|---|---|---|---|---|
| $\mu_1$ | 0.567 | 0.596 | 4.94 | 5.52 |
| $\mu_2$ | 0.064 | **0.299** | 5.87 | **5.47** |
| $\mu_3$ | 0.288 | 0.358 | 3.45 | 3.32 |

$\mu_2$ improves substantially ($R^2$ nearly 5x), $\mu_3$ improves modestly and
consistently, $\mu_1$ is a partial exception ($R^2$ improves slightly but RPE
worsens — a different kind of exception than $G_{13}$'s outright regression, also
unconfirmed). On the full test set, as with elastic, the two models are essentially
equivalent ($R^2$ changes by <0.003 for every direction) — same mechanism: the
permeability bound is already satisfied almost everywhere in the broader dataset
(0–0.17% baseline violation rate), so the constraint has little to fix outside the
most severe cases.

## Lessons learned (read this before extending the method)

These came from real dead ends during development — keeping them here so they don't
get rediscovered the expensive way.

1. **The constraint's benefit is concentrated at the extremes, not spread across the
   "low-$V_f$" band.** The penalty is exactly zero for any prediction already inside
   the admissible bound. The baseline's bound-violation rate is already low (0.2–3.2%
   for elastic, 0–0.17% for permeability) across most of the conventional low-$V_f$
   band ($V_f<0.45$) — there's nothing there for the constraint to fix. The real
   signal only shows up when you isolate the most severe cases specifically (the
   bottom ~1-2% of $V_f$, not the bottom 8%). This held for both elastic and
   permeability.

2. **$E_{11}$ must be excluded from the penalty if $V_f$ is derived from $E_{11}$
   itself.** This codebase estimates $V_f$ by inverting the Voigt rule of mixtures on
   the $E_{11}$ label: $\hat{V}_f=(E_{11}-E_m)/(E_{f1}-E_m)$. Plugging that estimate
   back into the Voigt bound *for* $E_{11}$ returns $E_{11}$ itself to floating-point
   precision — the constraint becomes "never predict above the answer," which is
   circular and actively harmful. Confirmed empirically: including $E_{11}$ in the
   penalty under this $V_f$ estimator dropped its $R^2$ from 0.94 to 0.82.
   Permeability has no analogous issue: its $V_f$ is real ground truth, not derived
   from any of $\mu_1,\mu_2,\mu_3$, so all three could stay in the Wiener penalty.

3. **Switching to real ground-truth $V_f$ did not help elastic, and revealed a second
   problem.** `Combined_Properties_with_FVF_updated.csv` has real, simulation-derived
   $V_f$ — using it for the elastic model (with $E_{11}$ re-included, since the
   circularity no longer applies) produced a *worse* result than the
   $E_{11}$-derived-$V_f$ approach (−6.4% low-$V_f$ RPE vs. baseline). Root cause,
   checked directly against the ground truth: **27% of the real $E_{11}$ labels
   themselves fall slightly outside the Voigt bound** (e.g. true $E_{11}=132.28$ GPa
   against a Voigt ceiling of 132.24) — almost certainly finite-RVE/stochastic-
   microstructure scatter, not a data error. The bound isn't a true hard constraint
   the real data always obeys, so enforcing it pushes predictions away from correct
   answers for a real fraction of cases. This is a structural issue with the bound
   formulation, not something more data or a better $V_f$ estimate fixes on its own.

4. **The Voigt-Reuss bound is not a meaningful admissibility criterion for Poisson's
   ratio in this material system.** $\nu_{12},\nu_{13},\nu_{23}$ show 90–100% bound
   "violations" in *both* the baseline and physics-informed models, regardless of
   training. Cause: $\nu_{f23}=0.298$ and $\nu_m=0.37$ are close in value, so the
   admissible band is only a few thousandths to ~2% wide — narrower than the natural
   scatter in FE-simulated $\nu$ at fixed $V_f$. The ground truth itself routinely
   lies outside this band. $\nu_{12},\nu_{13},\nu_{23}$ are excluded from the penalty
   for this reason (`bounds.py`'s docstring has the full derivation).

5. **The Hashin-Shtrikman permeability bound formula needs to be the corrected
   version in `bounds_permeability.py`, not a naive one.** An earlier formula failed
   its own degenerate-limit check (at $V_f=1$ it returned a value nowhere near the
   pure-fiber permeability). The corrected two-phase scalar HS formula is
   self-tested on import — run `python bounds_permeability.py` to confirm before
   trusting it if you modify it. Separately, mu3 (the fiber-direction permeability)
   is excluded from the HS penalty specifically (kept in Wiener) because the real
   ground truth violates the 2D cross-sectional HS bound 100% of the time at every
   $V_f$ — axial flow along continuous fibres is a different transport regime than
   the cross-sectional flow this HS formula models. Unrelated to the $E_{11}$
   circularity issue (#2) -- this is a wrong-physical-model problem, not a Vf-source
   problem.

6. **$\lambda$ was tuned, not assumed, and the same value worked for both
   properties.** Values from 0.02 to 1.0 were tried; very small weights ($\le 0.05$)
   are too weak to measurably affect training, and weights above ~0.5 tend to
   degrade full-dataset accuracy without a corresponding tail benefit. $\lambda=0.25$
   gave the clearest most-severe-case benefit for *both* elastic and permeability,
   from a limited shared search — a proper per-property grid or Bayesian search is a
   clear next step (see below).

## Known open issues / next steps

- $G_{13}$ (elastic) and $\mu_1$ (permeability) each show a partial/full exception to
  an otherwise positive pattern under the constraint — no confirmed mechanism for
  either yet. They may or may not be related; worth investigating together.
- A per-property-tuned $\lambda$ (rather than one shared value across all properties,
  and the same value reused across elastic and permeability) may sharpen the benefit
  and could resolve these exceptions.
- A revised Poisson's-ratio admissibility criterion (e.g. via a Hashin-Shtrikman-
  consistent transform of the bulk/shear bounds, rather than a direct linear
  rule-of-mixtures on $\nu$) is needed before that property can be meaningfully
  constrained at all.
- A revised, direction-aware bound formulation for $\mu_3$ (axial permeability) would
  let it benefit from an HS-tightness penalty too, instead of only the looser Wiener
  bound.
