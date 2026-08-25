"""Consistency radius r* and bootstrap calibration."""
from __future__ import annotations

import numpy as np

from .poset import PatternPoset
from .sheaf import GaussianMeanSheaf


def minimal_radius(poset: PatternPoset,
                   observations: dict[tuple[int, ...], np.ndarray],
                   obs_weights: dict | None = None,
                   edge_weights: dict | None = None) -> dict:
    """r*_quad: L2 distance from the observed per-pattern summary to the
    nearest global section (weighted least-squares projection residual).

    Returns dict with radius, fused stacked section, per-pattern contributions.
    """
    sheaf = GaussianMeanSheaf(poset)
    g, resid_sq = sheaf.project(observations, weights=obs_weights)
    # decompose residual into observation terms (per pattern) for localization
    contrib = {}
    for k, r in enumerate(poset.patterns):
        if r not in observations:
            continue
        w = 1.0 if obs_weights is None else float(obs_weights.get(("obs", r), 1.0))
        diff = sheaf.block(g, r) - observations[r]
        contrib[r] = float(w * float(diff @ diff))
    total = sum(contrib.values()) or 1.0
    return {
        "radius": float(np.sqrt(max(resid_sq, 0.0))),
        "section": g,
        "contributions": {r: c / total for r, c in contrib.items()},
        "raw_contributions": contrib,
    }


def bootstrap_radius_quantile(fit_and_radius_fn, B: int = 500, alpha: float = 0.05,
                              seed: int = 0) -> dict:
    """Calibration wrapper: `fit_and_radius_fn(resample_rng)` must refit the
    global section on a bootstrap resample and return its radius. The null
    quantile of those radii is the threshold tau_{1-alpha}.

    Protocol note (formalization_v0 B2): resamples should be drawn around the
    fitted consistent model (parametric bootstrap on residuals), which mimics
    the MAR-null sampling distribution of r*.
    """
    rng = np.random.default_rng(seed)
    radii = np.empty(B)
    for b in range(B):
        radii[b] = fit_and_radius_fn(rng)
    return {
        "radii": radii,
        "tau": float(np.quantile(radii, 1.0 - alpha)),
        "alpha": alpha,
        "B": B,
    }
