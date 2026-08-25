"""Phase-2 ground-truth decision flow (WP2.1/WP2.2 engine side).

Instruments, in precedence order (all reuse frozen Phase-1 primitives):

1. Formula oracle: every registered identification formula is attempted and
   ACCEPTED only if it reproduces the true target at machine precision on
   this instance's seeded parameters. A structurally invalid formula fails
   verification with probability 1 under random parameters, so acceptance is
   a sound positive certificate for the instance.
2. LP pinching: if the assumption-free relaxation over full tables matching
   the observed fingerprint has width ~0, the target is uniquely determined
   with NO mechanism assumptions, hence certainly recoverable under the model.
3. Model-valid witness search: two independent rounds of null-space
   root-jumping (multistart least-squares root finding on the observable
   fingerprint). A pair of distinct factorized models with identical
   fingerprints but different target values certifies unrecoverability under
   the model. NOTE (deviation from Phase 1): the SLSQP-based witness path was
   REMOVED after scipy 1.17.1's SLSQP wrapper deterministically corrupted the
   interpreter heap on certain instances (witness: structure n3_s00019_d0);
   Phase 1 itself found root-jumping the most effective witness strategy at
   these sizes.

Anything else is UNDETERMINED (sub-classified as relaxed-fragile when the LP
relaxation itself varies). Undecided rows are reported separately; primary
agreement statistics use decidable rows only.
"""
from __future__ import annotations

import numpy as np

from .lp_ground_truth import (
    IDENTITY_FORMULAS,
    lp_range,
    pack,
    param_bounds,
    root_jump_search,
    target_value_phi,
    unpack,
)


def formula_oracle(inst, theta_true: np.ndarray, target) -> str | None:
    """First registered formula that verifies against the true value."""
    m = unpack(inst, theta_true)
    jt = m.joint_table()
    q = m.observed_laws(jt)
    pp = {}
    for (v, r), p in jt.items():
        pp[r] = pp.get(r, 0.0) + p
    aux = {"pattern_prob": pp}
    true_phi = target_value_phi(m, target)
    for name, fn in IDENTITY_FORMULAS.items():
        try:
            est = fn(m, q, target, aux)
        except Exception:
            continue
        if est is None or not np.isfinite(est):
            continue
        if abs(est - true_phi) <= 1e-8 * max(1.0, abs(true_phi)):
            return name
    return None


def collect_roots_early(inst, theta_ref, patterns, n_starts: int = 48,
                        max_roots: int = 12, seed: int = 0,
                        tol: float = 1e-9):
    """Distinct factorized completions of the observed fingerprint. Runs the
    FULL start budget unless max_roots distinct roots are already found (no
    duplicate-streak early stopping: converging repeatedly to one root does
    not certify uniqueness, and treating it as such produced false positives
    in piloting)."""
    from scipy.optimize import least_squares

    from .lp_ground_truth import observed_vector

    lo, hi = param_bounds(inst)
    free = np.where(hi - lo > 0)[0]
    base = pack(inst)
    f_ref, _ = observed_vector(unpack(inst, base), patterns)

    def expand(xf):
        th = base.copy()
        th[free] = xf
        return th

    roots = []
    if len(free) == 0:
        return [base]
    rng = np.random.default_rng(seed)
    span = hi[free] - lo[free]
    for _ in range(n_starts):
        x0f = lo[free] + 0.02 * span + rng.random(len(free)) * (0.96 * span)
        res = least_squares(
            lambda xf: observed_vector(unpack(inst, expand(xf)), patterns)[0] - f_ref,
            x0f, bounds=(lo[free], hi[free]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if np.max(np.abs(res.fun)) >= tol:
            continue
        full = expand(res.x)
        if all(np.max(np.abs(full - u)) > 1e-6 for u in roots):
            roots.append(full.copy())
            if len(roots) >= max_roots:
                break
    return roots


def fingerprint_jacobian_rank(inst, theta: np.ndarray, patterns,
                              eps: float = 1e-6) -> tuple[int, int]:
    """Local identifiability annotation: rank of the observable-fingerprint
    Jacobian w.r.t. free parameters versus the number of free parameters.
    Descriptive evidence only; never upgrades verdicts."""
    from .lp_ground_truth import observed_vector

    f0, _ = observed_vector(unpack(inst, theta), patterns)
    lo, hi = param_bounds(inst)
    free = np.where(hi - lo > 0)[0]
    J = np.zeros((len(f0), len(free)))
    for k, idx in enumerate(free):
        tp = theta.copy()
        tp[idx] += eps
        tm = theta.copy()
        tm[idx] -= eps
        fp, _ = observed_vector(unpack(inst, tp), patterns)
        fm, _ = observed_vector(unpack(inst, tm), patterns)
        J[:, k] = (fp - fm) / (2 * eps)
    s = np.linalg.svd(J, compute_uv=False)
    return int(np.sum(s > 1e-8)), int(len(free))


def sheaf_fiber_verdict(inst, theta_true: np.ndarray, target,
                        n_starts: int = 48, max_roots: int = 12,
                        spread_tol: float = 1e-6, seed: int = 7) -> dict:
    """B1-side verdict: spread of the target across distinct factorized
    completions of the true observed fingerprint (fiber constancy)."""
    from .lp_ground_truth import observed_vector

    m_true = unpack(inst, theta_true)
    patterns = m_true.realized_patterns(jt=m_true.joint_table())
    phi_ref = target_value_phi(m_true, target)
    roots = collect_roots_early(inst, theta_true, patterns,
                                n_starts=n_starts, max_roots=max_roots,
                                seed=seed)
    phis = [target_value_phi(unpack(inst, r), target) for r in roots]
    spread = float(max(phis) - min(phis)) if phis else 0.0
    rank, n_free = fingerprint_jacobian_rank(inst, theta_true, patterns)
    return {
        "sheaf_verdict": "RECOVERABLE" if spread < spread_tol else "UNRECOVERABLE",
        "phi_spread_over_fiber": spread,
        "n_distinct_completions": len(roots),
        "phi_values_sample": [float(p) for p in phis[:max_roots]],
        "jacobian_rank": rank,
        "n_free_params": n_free,
        "n_patterns": len(patterns),
    }


def decide2(inst, theta_true: np.ndarray, target,
            jump_starts: int = 40,
            lp_pinch_tol: float = 1e-9, lp_width_tol: float = 1e-3,
            seed: int = 0) -> dict:
    """Engine-side ground truth. See module docstring for precedence."""
    out: dict = {}
    fname = formula_oracle(inst, theta_true, target)
    if fname is not None:
        out.update(gt_verdict="RECOVERABLE", gt_evidence=f"formula:{fname}")
        return out

    m_true = unpack(inst, theta_true)
    q = m_true.observed_laws()
    true_phi = target_value_phi(m_true, target)
    out["true_value"] = true_phi

    if target[0] in ("mean", "cell"):
        rng_lp = lp_range(inst, q, target)
        out["lp"] = {"width": rng_lp["width"], "lo": rng_lp["lo"], "hi": rng_lp["hi"]}
        if rng_lp["width"] <= lp_pinch_tol:
            out.update(gt_verdict="RECOVERABLE", gt_evidence="lp_pinched")
            return out

    wit = root_jump_search(inst, theta_true, target,
                           n_starts=jump_starts, seed=seed)
    if not wit["success"]:
        walk = root_jump_search(inst, theta_true, target,
                                n_starts=jump_starts, seed=seed + 101)
        if walk["delta_phi"] > wit["delta_phi"]:
            wit = walk
    out["witness"] = {k: wit[k] for k in ("delta_phi", "dist", "success")}
    if wit["success"]:
        out.update(gt_verdict="UNRECOVERABLE",
                   gt_evidence=f"model_witness(rootjump) dphi={wit['delta_phi']:.4f} "
                               f"dist={wit['dist']:.1e}")
        return out

    if out.get("lp", {}).get("width", 0.0) > lp_width_tol:
        out.update(gt_verdict="UNDETERMINED_RELAXED_FRAGILE",
                   gt_evidence="no model witness; relaxation varies")
    else:
        out.update(gt_verdict="UNDETERMINED",
                   gt_evidence="no certificate either way")
    return out
