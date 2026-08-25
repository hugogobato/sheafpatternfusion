"""Fused pattern-projection estimator and diagnostic localization."""
from __future__ import annotations

import numpy as np

from .poset import PatternPoset
from .radius import minimal_radius
from .sheaf import GaussianMeanSheaf


def fuse(poset: PatternPoset,
         means: dict[tuple[int, ...], np.ndarray],
         ns: dict[tuple[int, ...], float] | None = None,
         edge_weights: dict | None = None) -> dict:
    """Projection-based fusion of per-pattern mean estimates.

    weights default to n_r (sample-size weighting); the fused estimate of any
    full-vector functional T is T applied to the stacked global section.
    Returns the section per pattern, r*, and localization diagnostics.
    """
    if len(poset.patterns) == 1:
        r0 = poset.patterns[0]
        mu = np.asarray(means[r0], dtype=float)
        full = np.zeros(poset.n_vars)
        k = 0
        for i in range(poset.n_vars):
            if r0[i] == 1:
                full[i] = mu[k]
                k += 1
        return {
            "fused": {r0: mu.copy()},
            "full_mean": full,
            "radius": 0.0,
            "contributions": {r0: 1.0},
            "prediction_residuals": {},
            "section_stacked": mu.copy(),
        }

    obs_weights = None
    if ns is not None:
        obs_weights = {("obs", r): float(ns[r]) for r in poset.patterns if r in ns}
    res = minimal_radius(poset, means, obs_weights=obs_weights, edge_weights=edge_weights)

    sheaf = GaussianMeanSheaf(poset)
    fused = {r: sheaf.block(res["section"], r).copy() for r in poset.patterns}

    # full-dim mean estimate: average overlapping fused blocks (exact when
    # consistent; a stable read-out otherwise)
    n_vars = poset.n_vars
    acc = np.zeros(n_vars)
    cnt = np.zeros(n_vars)
    for r, v in fused.items():
        for k, i in enumerate([j for j in range(n_vars) if r[j] == 1]):
            acc[i] += v[k]
            cnt[i] += 1
    cnt[cnt == 0] = 1
    full_mean = acc / cnt

    # leave-one-out prediction residuals: refit without pattern r and measure
    # how badly the remaining patterns' consensus predicts r's block. The
    # contaminating pattern is precisely the one the others disagree with.
    pred_resid = {}
    obs_weights = None if ns is None else {("obs", r): float(ns[r]) for r in poset.patterns if r in ns}
    sheaf_g = GaussianMeanSheaf(poset)
    for r_out in list(means.keys()):
        sub_patterns = [p for p in means.keys() if p != r_out]
        if len(sub_patterns) < 2:
            pred_resid[r_out] = 0.0
            continue
        sub_poset = PatternPoset(sub_patterns)
        sub_means = {p: means[p] for p in sub_patterns}
        sub_ns = None if ns is None else {p: ns[p] for p in sub_patterns}
        sub = minimal_radius(sub_poset, sub_means, obs_weights=sub_ns)
        sub_sheaf = GaussianMeanSheaf(sub_poset)
        idx_r = [i for i in range(poset.n_vars) if r_out[i] == 1]
        acc_pred = np.zeros(len(idx_r))
        cnt_pred = 0
        for p in sub_patterns:
            block = sub_sheaf.block(sub["section"], p)
            coords = [i for i in range(poset.n_vars) if p[i] == 1]
            for k, i in enumerate(coords):
                if i in idx_r:
                    acc_pred[idx_r.index(i)] += block[k]
                    cnt_pred += 1
        cnt_pred = max(cnt_pred // max(len(idx_r), 1), 1)
        pred = acc_pred / cnt_pred
        w = 1.0 if ns is None else float(ns[r_out])
        pred_resid[r_out] = float(w * np.sum((pred - means[r_out]) ** 2))

    return {
        "fused": fused,
        "full_mean": full_mean,
        "radius": res["radius"],
        "contributions": res["contributions"],
        "raw_contributions": res["raw_contributions"],
        "prediction_residuals": pred_resid,
        "section_stacked": res["section"],
    }


def localize(result: dict) -> tuple[int, ...]:
    """Top-1 contaminated pattern: primary signal is the leave-one-out
    prediction residual (the pattern the consensus of the others cannot
    predict); contribution share is the fallback."""
    scores = result.get("prediction_residuals") or {}
    if scores and max(scores.values()) > 0:
        return max(scores, key=lambda r: scores.get(r, 0.0))
    contrib = result["contributions"]
    return max(contrib, key=lambda r: contrib[r])
