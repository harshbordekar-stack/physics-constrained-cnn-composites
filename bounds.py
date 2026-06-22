"""
Analytical homogenization bounds for the IM7-8552 system, used as the physics-informed
penalty terms in the Analytical Constraint Loss described in main.tex Section 2.3.3
(Equations 6-12).

Material constants are taken directly from the paper's own Table 1 (IM7 fiber) and
Table 2 (8552 matrix) so the constraint loss is consistent with what the manuscript
reports -- this matters because the previous version of train_pgcnn.py used different,
undocumented constants (Em=4.67 vs the paper's 4.23 GPa, a single nu_f=0.2 instead of
the paper's split nu_f12=0.23 / nu_f23=0.298), which would have made any results
computed from it inconsistent with the rest of the manuscript.

SIMPLIFYING ASSUMPTIONS (read this before trusting/extending the HS bounds):

IM7 is transversely isotropic (5 independent elastic constants), and the textbook-exact
Hashin-Shtrikman bounds for a transversely-isotropic fiber in an isotropic matrix
(Hashin & Rosen 1964; Hashin 1979, "Analysis of composite materials -- a survey") require
the full 5-constant treatment. Implementing that exactly is a substantial undertaking on
its own and easy to get subtly wrong. To keep this both correct *and* verifiable, this
module instead:

  - Implements Voigt and Reuss bounds *exactly* (closed-form, no approximation) for all
    eight predicted properties -- these are simple rule-of-mixtures / inverse rule-of-
    mixtures formulas and there is no ambiguity in them.
  - Implements Hashin-Shtrikman bounds using the classical *isotropic* two-phase HS
    formulas (Hashin & Shtrikman, J. Mech. Phys. Solids, 1963) for E22, E33, G12, G13 --
    the properties where HS bounds are standard and most informative -- treating the
    matrix as isotropic (true, 8552 is isotropic) and the fiber's relevant in-plane
    behavior as isotropic using its transverse constants (Ef2, Gf23, nu_f23) or its axial
    shear constant (Gf12) as appropriate. This is a documented simplification, not the
    full transversely-isotropic theory.
  - Does NOT apply an independent HS bound to E11, nu12, nu13, or nu23. For E11 this is
    consistent with classical theory anyway (the axial modulus bounds are essentially
    set by Voigt-Reuss for typical fiber/matrix stiffness ratios). For the Poisson ratios,
    combining HS bounds on K and G into an HS bound on nu does not generally preserve a
    valid bounding interval, so it is safer to omit it than to report something incorrect.
    lambda_HS contributes zero gradient for these four properties as a result -- this
    should be stated explicitly in the Methodology text rather than left implicit.

If a future revision implements the full transversely-isotropic HS-Walpole bounds, this
module is the place to replace the approximations -- the public function signatures
(voigt_reuss_bounds, hashin_shtrikman_bounds) are designed to stay the same.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constituent material properties -- IM7 fiber (Table 1) and 8552 matrix (Table 2)
# ---------------------------------------------------------------------------

EF1 = 275.7     # GPa, longitudinal fiber modulus
EF2 = 12.2      # GPa, transverse fiber modulus (= EF3)
GF12 = 18.3     # GPa, fiber in-plane shear modulus
GF23 = 4.7      # GPa, fiber transverse shear modulus
NUF12 = 0.23    # fiber major Poisson ratio (= NUF13)
NUF23 = 0.298   # fiber minor (transverse) Poisson ratio

EM = 4.23       # GPa, matrix modulus
NUM = 0.37      # matrix Poisson ratio
GM = EM / (2 * (1 + NUM))          # isotropic matrix shear modulus
KM = EM / (3 * (1 - 2 * NUM))      # isotropic matrix bulk modulus

# Fiber transverse cross-section treated as isotropic for the HS bound only
# (see module docstring) -- NOT used for Voigt-Reuss, which uses EF2/GF12/GF23 directly.
KF2_ISO_APPROX = EF2 / (3 * (1 - 2 * NUF23))

PROPERTY_NAMES = ["E11", "E22", "E33", "G12", "G13", "nu12", "nu13", "nu23"]


def _voigt_reuss_scalar(vf, fiber_val, matrix_val):
    """Standard rule-of-mixtures (Voigt) and inverse rule-of-mixtures (Reuss) bounds
    for a single scalar property given fiber and matrix values. Works for any of the
    eight properties -- this is exact, not an approximation, for the two-phase
    parallel/series limiting cases that Voigt and Reuss represent."""
    vm = 1.0 - vf
    voigt = vf * fiber_val + vm * matrix_val
    reuss = 1.0 / (vf / fiber_val + vm / matrix_val)
    return voigt, reuss


def voigt_reuss_bounds(vf):
    """Returns dict {property_name: (voigt_upper, reuss_lower)} for all 8 properties,
    evaluated at fiber volume fraction vf (scalar or numpy array)."""
    vf = np.asarray(vf, dtype=np.float64)
    bounds = {}
    bounds["E11"] = _voigt_reuss_scalar(vf, EF1, EM)
    bounds["E22"] = _voigt_reuss_scalar(vf, EF2, EM)
    bounds["E33"] = _voigt_reuss_scalar(vf, EF2, EM)  # transversely isotropic: E33 = E22
    bounds["G12"] = _voigt_reuss_scalar(vf, GF12, GM)
    bounds["G13"] = _voigt_reuss_scalar(vf, GF12, GM)  # G13 = G12
    bounds["nu12"] = _voigt_reuss_scalar(vf, NUF12, NUM)
    bounds["nu13"] = _voigt_reuss_scalar(vf, NUF12, NUM)
    bounds["nu23"] = _voigt_reuss_scalar(vf, NUF23, NUM)
    return bounds


def _hs_isotropic_KG(vf, k1, g1, k2, g2):
    """Classical Hashin-Shtrikman (1963) bounds for the effective bulk modulus K and
    shear modulus G of a two-phase isotropic composite. Phase 1 is used as the
    "reference" medium for the upper bound, phase 2 for the lower bound -- which phase
    is stiffer determines which bound is actually the upper one; callers should compare
    the two returned values rather than assume order.

    v1, v2 are the volume fractions of phase 1 / phase 2 respectively (v1=vf is the
    convention used throughout this module when phase 1 = fiber, phase 2 = matrix)."""
    v1 = vf
    v2 = 1.0 - vf

    # Standard HS closed forms (Hashin & Shtrikman, J. Mech. Phys. Solids 11, 1963;
    # verified against Christensen, "Mechanics of Composite Materials", Ch. 3):
    k_a = k1 + v2 / (1.0 / (k2 - k1) + v1 / (k1 + 4.0 / 3.0 * g1))
    k_b = k2 + v1 / (1.0 / (k1 - k2) + v2 / (k2 + 4.0 / 3.0 * g2))

    def g_hs(ka, ga, kb, gb, va, vb):
        beta_a = ga * (9 * ka + 8 * ga) / (6 * (ka + 2 * ga))
        return ga + vb / (1.0 / (gb - ga) + va / (ga + beta_a))

    g_a = g_hs(k1, g1, k2, g2, v1, v2)
    g_b = g_hs(k2, g2, k1, g1, v2, v1)

    k_upper, k_lower = np.maximum(k_a, k_b), np.minimum(k_a, k_b)
    g_upper, g_lower = np.maximum(g_a, g_b), np.minimum(g_a, g_b)
    return k_upper, k_lower, g_upper, g_lower


def _kg_to_E_nu(k, g):
    """Isotropic combination: Young's modulus and Poisson ratio from bulk and shear
    modulus. E = 9KG / (3K + G); nu = (3K - 2G) / (2*(3K + G))."""
    E = 9 * k * g / (3 * k + g)
    nu = (3 * k - 2 * g) / (2 * (3 * k + g))
    return E, nu


def hashin_shtrikman_bounds(vf):
    """Returns dict {property_name: (hs_upper, hs_lower)} for the four properties this
    module applies HS bounds to (E22, E33, G12, G13). See module docstring for why E11,
    nu12, nu13, nu23 are intentionally excluded.
    """
    vf = np.asarray(vf, dtype=np.float64)

    # Transverse properties (E22/E33): fiber phase approximated as isotropic using
    # (Kf2_iso_approx, Gf23); matrix is exactly isotropic (Km, Gm).
    k_u, k_l, g_u, g_l = _hs_isotropic_KG(vf, KF2_ISO_APPROX, GF23, KM, GM)
    E_u, _ = _kg_to_E_nu(k_u, g_u)
    E_l, _ = _kg_to_E_nu(k_l, g_l)
    e22_bounds = (np.maximum(E_u, E_l), np.minimum(E_u, E_l))

    # Axial shear (G12/G13): same matrix (Km, Gm), fiber shear taken as the real axial
    # shear constant Gf12 paired with the same transverse-isotropic-approximation bulk
    # modulus (the HS shear bound is far less sensitive to K than to G, so reusing
    # KF2_ISO_APPROX here is a minor approximation, not a major source of error).
    _, _, g12_u, g12_l = _hs_isotropic_KG(vf, KF2_ISO_APPROX, GF12, KM, GM)
    g12_bounds = (np.maximum(g12_u, g12_l), np.minimum(g12_u, g12_l))

    return {
        "E22": e22_bounds,
        "E33": e22_bounds,
        "G12": g12_bounds,
        "G13": g12_bounds,
    }


HS_PROPERTIES = ("E22", "E33", "G12", "G13")


def check_bound_violations(predictions, vf, tol=1e-6):
    """predictions: dict or array-like of shape (N, 8) in PROPERTY_NAMES order, or a
    dict {name: array(N,)}. vf: array(N,). Returns dict {name: violation_rate_percent}
    against the Voigt-Reuss bounds (the absolute physical limits referenced in the
    paper's Table "bound violations") -- HS bounds are tighter and not used for the
    pass/fail violation check, consistent with how the manuscript frames Voigt-Reuss as
    the inadmissible-region boundary.
    """
    if not isinstance(predictions, dict):
        predictions = {name: predictions[:, i] for i, name in enumerate(PROPERTY_NAMES)}

    vr = voigt_reuss_bounds(vf)
    out = {}
    for name in PROPERTY_NAMES:
        pred = np.asarray(predictions[name], dtype=np.float64)
        upper, lower = vr[name]
        upper, lower = np.maximum(upper, lower), np.minimum(upper, lower)
        violates = (pred > upper + tol) | (pred < lower - tol)
        out[name] = 100.0 * np.mean(violates)
    return out


if __name__ == "__main__":
    # Self-check: run `python bounds.py` to verify before trusting these in training.
    # 1) Degenerate limits: at Vf=0 every Voigt/Reuss bound must collapse to the matrix
    #    value; at Vf=1 every bound must collapse to the fiber value. This is the same
    #    physical boundary condition the paper's Methodology motivates the whole
    #    constraint loss with -- if this fails, the loss is enforcing the wrong physics.
    matrix_vals = {"E11": EM, "E22": EM, "E33": EM, "G12": GM, "G13": GM,
                   "nu12": NUM, "nu13": NUM, "nu23": NUM}
    fiber_vals = {"E11": EF1, "E22": EF2, "E33": EF2, "G12": GF12, "G13": GF12,
                  "nu12": NUF12, "nu13": NUF12, "nu23": NUF23}

    vr0, vr1 = voigt_reuss_bounds(0.0), voigt_reuss_bounds(1.0)
    ok = True
    for name in PROPERTY_NAMES:
        v0, r0 = vr0[name]
        v1_, r1 = vr1[name]
        if not (np.isclose(v0, matrix_vals[name]) and np.isclose(r0, matrix_vals[name])):
            print(f"FAIL: {name} at Vf=0 -> Voigt={v0:.4f} Reuss={r0:.4f}, expected {matrix_vals[name]}")
            ok = False
        if not (np.isclose(v1_, fiber_vals[name]) and np.isclose(r1, fiber_vals[name])):
            print(f"FAIL: {name} at Vf=1 -> Voigt={v1_:.4f} Reuss={r1:.4f}, expected {fiber_vals[name]}")
            ok = False
    print("Voigt-Reuss degenerate-limit check:", "PASS" if ok else "FAIL")

    # 2) HS bounds must be tighter than (nested inside) Voigt-Reuss at every Vf, since
    #    HS is a refinement of V-R, not an independent/looser bound. If HS ever falls
    #    outside V-R, the HS formula has a sign or algebra error.
    vf_range = np.linspace(0.01, 0.99, 25)
    vr = voigt_reuss_bounds(vf_range)
    hs = hashin_shtrikman_bounds(vf_range)
    ok2 = True
    for name in HS_PROPERTIES:
        v_up, v_lo = np.maximum(*vr[name]), np.minimum(*vr[name])
        h_up, h_lo = np.maximum(*hs[name]), np.minimum(*hs[name])
        if np.any(h_up > v_up + 1e-6) or np.any(h_lo < v_lo - 1e-6):
            print(f"FAIL: {name} HS bound falls outside Voigt-Reuss at some Vf")
            ok2 = False
    print("HS-nested-inside-Voigt-Reuss check:", "PASS" if ok2 else "FAIL")

    print("\nSample bounds at Vf=0.55 (paper's nominal layup):")
    vr55 = voigt_reuss_bounds(0.55)
    hs55 = hashin_shtrikman_bounds(0.55)
    for name in PROPERTY_NAMES:
        line = f"  {name:5s} Voigt/Reuss = [{min(vr55[name]):.3f}, {max(vr55[name]):.3f}]"
        if name in hs55:
            line += f"   HS = [{min(hs55[name]):.3f}, {max(hs55[name]):.3f}]"
        print(line)
