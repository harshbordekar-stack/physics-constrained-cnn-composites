"""
Analytical homogenization bounds for directional hydrogen permeability (mu1, mu2,
mu3), following Ebermann et al. (2022, Composite Structures 291:115616) -- the same
reference already cited in main.tex for the Wiener/Hashin-Shtrikman permeability
bounds, but with a corrected Hashin-Shtrikman formula (see CORRECTION below).

Permeability is a scalar transport property (unlike the 4th-order elastic stiffness
tensor), so its Voigt/Reuss and HS bounds are simpler than bounds.py's elastic ones --
no K/G decomposition, no transverse-isotropy approximation needed. This module
implements the textbook two-phase scalar bounds directly.

CONSTITUENT PERMEABILITY VALUES (Ebermann et al. Table 1, also main.tex Table
tab:permeability_coefficients):
    Epoxy resin (LY556/HY917):  1.4e-17  m^3(STP)*m / (m^2*s*Pa)
    Carbon fibre (Toho HTA7):   1.49e-18 m^3(STP)*m / (m^2*s*Pa)
Note the matrix is MORE permeable than the fibre (opposite of the elastic-stiffness
case, where the fibre is the "stiff" phase) -- this flips which phase gives the upper
vs. lower bound compared to bounds.py's elastic conventions. Voigt/Reuss/HS here are
applied identically across mu1, mu2, mu3 (the same isotropic two-phase treatment used
elsewhere in this project for the cross-sectional problem; main.tex already uses a
single d=2 setting for all three directions, so this module follows that same
convention rather than introducing a direction-specific bound unilaterally).

CORRECTION TO main.tex's CURRENT HS FORMULA:
The Methodology text currently states:
    P_HS^pm = P_0 + [sum_i v_i*(P_i-P_0)^-1] / [1 + (1/(d*P_0))*sum_i v_i*(P_i-P_0)^-1]
Checked numerically against the required degenerate limits (Vf=0 -> matrix value,
Vf=1 -> fibre value): this formula does NOT pass that check (verified: at Vf=1 it
gives ~4.2e-17, not the fibre value 1.49e-18). The correct two-phase scalar
Hashin-Shtrikman formula (Hashin & Shtrikman 1962; matches the same k_a/k_b structure
already used for the elastic K,G bounds in bounds.py's _hs_isotropic_KG, with the
elastic "4/3*g1" depolarization term replaced by the scalar dimensional term
"v1/(d*sigma1)" appropriate for a conductivity-type property) is:

    sigma_HS_upper = sigma_ref_hi + v_lo / (1/(sigma_lo - sigma_ref_hi) + v_hi/(d*sigma_ref_hi))
    sigma_HS_lower = sigma_ref_lo + v_hi / (1/(sigma_hi - sigma_ref_lo) + v_lo/(d*sigma_ref_lo))

where sigma_ref_hi/lo are the higher/lower-permeability phase used as the HS reference
medium. This is implemented below and self-tested against the degenerate limits before
being trusted (see __main__ block) -- do not re-introduce the unverified formula above.
"""

import numpy as np

P_EPOXY = 1.4e-17    # m^3(STP)*m / (m^2*s*Pa), matrix -- the MORE permeable phase
P_FIBER = 1.49e-18   # m^3(STP)*m / (m^2*s*Pa), fibre -- the LESS permeable phase

PERMEABILITY_NAMES = ["mu1", "mu2", "mu3"]

D_DIM = 2  # 2D cross-sectional treatment, matching main.tex's existing convention


def voigt_reuss_bounds_permeability(vf):
    """Wiener bounds: arithmetic mean (Voigt, parallel/upper) and harmonic mean
    (Reuss, series/lower). vf = fiber volume fraction (scalar or array, in [0,1])."""
    vf = np.asarray(vf, dtype=np.float64)
    vm = 1.0 - vf
    voigt = vf * P_FIBER + vm * P_EPOXY          # arithmetic mean -- always the upper
                                                  # bound (AM >= HM for any positive
                                                  # values/weights), regardless of vf.
    reuss = 1.0 / (vf / P_FIBER + vm / P_EPOXY)  # harmonic mean -- always the lower bound
    return voigt, reuss


def hashin_shtrikman_bounds_permeability(vf, d=D_DIM):
    """Corrected two-phase scalar HS bounds -- see module docstring. Returns
    (upper, lower) as a tuple of arrays, already ordered (max, min) at every vf so
    callers don't need to re-sort."""
    vf = np.asarray(vf, dtype=np.float64)
    vm = 1.0 - vf

    # Upper bound: use the MORE permeable phase (epoxy) as the HS reference medium.
    hs_a = P_EPOXY + vf / (1.0 / (P_FIBER - P_EPOXY) + vm / (d * P_EPOXY))
    # Lower bound: use the LESS permeable phase (fiber) as the HS reference medium.
    hs_b = P_FIBER + vm / (1.0 / (P_EPOXY - P_FIBER) + vf / (d * P_FIBER))

    upper = np.maximum(hs_a, hs_b)
    lower = np.minimum(hs_a, hs_b)
    return upper, lower


def check_bound_violations_permeability(predictions, vf, tol=1e-25):
    """predictions: dict {name: array(N,)} or array (N,3) in PERMEABILITY_NAMES order.
    vf: array(N,) in [0,1]. Returns dict {name: violation_rate_percent} against the
    Wiener (Voigt-Reuss) bounds, mirroring bounds.py's check_bound_violations."""
    if not isinstance(predictions, dict):
        predictions = {name: predictions[:, i] for i, name in enumerate(PERMEABILITY_NAMES)}

    voigt, reuss = voigt_reuss_bounds_permeability(vf)
    upper, lower = np.maximum(voigt, reuss), np.minimum(voigt, reuss)
    out = {}
    for name in PERMEABILITY_NAMES:
        pred = np.asarray(predictions[name], dtype=np.float64)
        violates = (pred > upper + tol) | (pred < lower - tol)
        out[name] = 100.0 * np.mean(violates)
    return out


if __name__ == "__main__":
    # 1) Degenerate limits for BOTH Wiener and the corrected HS formula.
    vr0_v, vr0_r = voigt_reuss_bounds_permeability(0.0)
    vr1_v, vr1_r = voigt_reuss_bounds_permeability(1.0)
    ok = (np.isclose(vr0_v, P_EPOXY) and np.isclose(vr0_r, P_EPOXY) and
          np.isclose(vr1_v, P_FIBER) and np.isclose(vr1_r, P_FIBER))
    print(f"Wiener (Voigt-Reuss) degenerate-limit check: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  Vf=0: voigt={vr0_v:.4e} reuss={vr0_r:.4e} (expect {P_EPOXY:.4e})")
        print(f"  Vf=1: voigt={vr1_v:.4e} reuss={vr1_r:.4e} (expect {P_FIBER:.4e})")

    hs0_u, hs0_l = hashin_shtrikman_bounds_permeability(0.0)
    hs1_u, hs1_l = hashin_shtrikman_bounds_permeability(1.0)
    ok2 = (np.isclose(hs0_u, P_EPOXY) and np.isclose(hs0_l, P_EPOXY) and
           np.isclose(hs1_u, P_FIBER) and np.isclose(hs1_l, P_FIBER))
    print(f"Hashin-Shtrikman degenerate-limit check: {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        print(f"  Vf=0: upper={hs0_u:.4e} lower={hs0_l:.4e} (expect {P_EPOXY:.4e})")
        print(f"  Vf=1: upper={hs1_u:.4e} lower={hs1_l:.4e} (expect {P_FIBER:.4e})")

    # 2) HS must be nested inside Wiener at every Vf (HS is a refinement, not an
    #    independent/looser bound).
    vf_range = np.linspace(0.01, 0.99, 25)
    v_up = np.maximum(*voigt_reuss_bounds_permeability(vf_range))
    v_lo = np.minimum(*voigt_reuss_bounds_permeability(vf_range))
    h_up, h_lo = hashin_shtrikman_bounds_permeability(vf_range)
    nested = np.all(h_up <= v_up + 1e-25) and np.all(h_lo >= v_lo - 1e-25)
    print(f"HS-nested-inside-Wiener check: {'PASS' if nested else 'FAIL'}")

    print("\nSample bounds at Vf=0.55 (paper's nominal layup):")
    vv, vr = voigt_reuss_bounds_permeability(0.55)
    hu, hl = hashin_shtrikman_bounds_permeability(0.55)
    print(f"  Wiener = [{min(vv,vr):.4e}, {max(vv,vr):.4e}]")
    print(f"  HS     = [{hl:.4e}, {hu:.4e}]")
