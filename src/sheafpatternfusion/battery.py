"""WP2.5.1 degeneracy null battery.

Scores null policies against the certificate's labels on the engine-undecided
rows of the frozen Phase-2 merge:

  N0  constant RECOVERABLE
  N1  fraction-observed threshold sweep (predict RECOVERABLE above tau)
  N2  pattern-overlap density threshold sweep (predict RECOVERABLE above tau)
  N3  Frechet-width sign: share-pinned assumption-free interval on the target
      mean; wide interval -> predict UNRECOVERABLE
  N4  constant UNRECOVERABLE control

The share-pinned LP matches each realized pattern's observed conditional law
scaled by its OBSERVED pattern probability P(R=r). This is the honest
Manski-style partial-identification bound given the full fingerprint. It is
deliberately NOT `lp_ground_truth.lp_range`: that Phase-1/2 relaxation omits
the share constraints and its total-mass row sums cells AND scales, which
double-counts stratum mass and squeezes every width by the phantom factor
above (stored Phase-2 `lp_width` values live on that artificial scale, hence
the uniform 0.5s). Verdicts are unaffected (width-0 maps to width-0), but all
Phase-2.5 widths are computed here on the corrected scale.

Headline output is NOT raw agreement (N0 reproduces >=99.9% of labels by
construction): it is the disagreement set S* -- the certificate-RECOVERABLE
rows with the widest corrected Frechet intervals, i.e. the boldest claims --
which becomes the priority audit sample for WP2.5.2.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
from scipy.optimize import linprog


def frechet_bounds(n_vars: int, q: dict, pp: dict, target) -> dict:
    """Share-pinned assumption-free min/max of a mean target.

    LP variables are joint cells t[v, r] (the honest Manski-style object:
    R-strata are separate population cells, so NO cross-stratum consistency
    is imposed). Constraints: total mass 1; per realized pattern r the share
    sum_v t[v, r] = pp[r]; and the observed conditional law
    sum_{v: v_O = o} t[v, r] = pp[r] * q_r(o). The true P(v, r) is always
    feasible, so bounds bracket the truth by construction."""
    patterns = sorted(q.keys())
    cells = list(itertools.product(
        itertools.product((0, 1), repeat=n_vars), patterns))
    cindex = {c: k for k, c in enumerate(cells)}
    j = target[1]
    if target[0] != "mean":
        raise ValueError("frechet_bounds supports mean targets only")
    if not any(r[j] == 1 for r in patterns):
        return {"lo": 0.0, "hi": 1.0, "width": 1.0, "lp_status": 0,
                "degenerate": "target never observed"}

    rows = [np.ones(len(cells))]
    rhs = [1.0]
    for r in patterns:
        share_row = np.zeros(len(cells))
        for v in itertools.product((0, 1), repeat=n_vars):
            share_row[cindex[(v, r)]] = 1.0
        rows.append(share_row)
        rhs.append(float(pp.get(r, 0.0)))
        Oidx = [i for i in range(n_vars) if r[i] == 1]
        for o in itertools.product((0, 1), repeat=len(Oidx)):
            row = np.zeros(len(cells))
            for v in itertools.product((0, 1), repeat=n_vars):
                if tuple(v[i] for i in Oidx) == tuple(o):
                    row[cindex[(v, r)]] = 1.0
            rows.append(row)
            rhs.append(float(pp.get(r, 0.0)) * float(q[r].get(tuple(o), 0.0)))
    A = np.array(rows)
    b = np.array(rhs)

    c_obj = np.array([float(v[j]) for v, _ in cells])
    vals = {}
    for name, d in (("lo", 1.0), ("hi", -1.0)):
        res = linprog(d * c_obj, A_eq=A, b_eq=b,
                      bounds=[(0.0, 1.0)] * len(cells), method="highs")
        if res.status != 0:
            return {"lo": None, "hi": None, "width": None, "lp_status": res.status}
        vals[name] = float(res.fun * d)
    return {"lo": vals["lo"], "hi": vals["hi"],
            "width": vals["hi"] - vals["lo"], "lp_status": 0}


def instance_from_row(row: dict):
    from .enumerate_structures import instantiate
    from .lp_ground_truth import pack, unpack

    vp = {int(k): tuple(v) for k, v in row["var_parents"].items()}
    structure = (vp, tuple(tuple(p) for p in row["r_parents"]))
    inst = instantiate(structure, seed=row["seed"])
    m = unpack(inst, pack(inst))
    jt = m.joint_table()
    q = m.observed_laws(jt)
    pp: dict = {}
    for (v, r), p in jt.items():
        pp[r] = pp.get(r, 0.0) + p
    return inst, q, pp


def fraction_observed(pp: dict, n_vars: int) -> float:
    tot = sum(pp.values())
    if tot <= 0:
        return 0.0
    acc = 0.0
    for r, w in pp.items():
        acc += w * sum(r)
    return acc / (tot * n_vars)


def overlap_density(patterns: list[tuple]) -> float:
    js = []
    for i, a in enumerate(patterns):
        sa = {k for k, bit in enumerate(a) if bit == 1}
        for b in patterns[i + 1:]:
            sb = {k for k, bit in enumerate(b) if bit == 1}
            union = sa | sb
            if union and sa & sb:
                js.append(len(sa & sb) / len(union))
    return float(np.mean(js)) if js else 0.0


def score_row(row: dict, tau_cfg: dict | None = None) -> dict:
    """Per-row null features + corrected Frechet interval (streaming-friendly;
    used verbatim by the Colab battery and audit notebooks)."""
    inst, q, pp = instance_from_row(row)
    fb = frechet_bounds(inst.n_vars, q, pp, tuple(row["target"]))
    return {
        "instance_id": row["instance_id"],
        "target": list(row["target"]),
        "n_vars": row["n_vars"],
        "sheaf_recoverable": row["sheaf_recoverable"],
        "frac_observed": round(fraction_observed(pp, row["n_vars"]), 6),
        "overlap_density": round(overlap_density(
            [tuple(p) for p in row["patterns"]]), 6),
        "frechet_lo": fb["lo"],
        "frechet_hi": fb["hi"],
        "frechet_width": fb["width"],
        "true_value": row.get("true_value"),
    }


def aggregate_results(scored: list[dict], cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    tau_frac = cfg.get("tau_frac_observed",
                       [round(0.30 + 0.02 * k, 2) for k in range(36)])
    tau_overlap = cfg.get("tau_overlap", [round(0.05 * k, 2) for k in range(21)])
    tau_width = cfg.get("tau_width", [1e-3, 0.05, 0.10, 0.15, 0.20, 0.25,
                                      0.30, 0.35, 0.40, 0.45])
    s_cap = int(cfg.get("priority_sample_cap", 200))
    label = lambda r: r["sheaf_recoverable"] == "RECOVERABLE"  # noqa: E731

    def confusion(pred_rec):
        tp = sum(1 for p, r in zip(pred_rec, scored) if p and label(r))
        tn = sum(1 for p, r in zip(pred_rec, scored) if not p and not label(r))
        fp = sum(1 for p, r in zip(pred_rec, scored) if p and not label(r))
        fn = sum(1 for p, r in zip(pred_rec, scored) if not p and label(r))
        n = max(len(scored), 1)
        return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
                "accuracy": (tp + tn) / n}

    metrics: dict = {"n_rows": len(scored)}
    metrics["N0_constant_recoverable"] = confusion([True] * len(scored))
    metrics["N4_constant_unrecoverable"] = confusion([False] * len(scored))
    best = {"N1": None, "N2": None, "N3": None}
    sweeps = {"N1": [], "N2": [], "N3": []}
    for tau in tau_frac:
        c = confusion([r["frac_observed"] >= tau for r in scored])
        sweeps["N1"].append({"tau": tau, **c})
        if best["N1"] is None or c["accuracy"] > best["N1"]["accuracy"]:
            best["N1"] = {"tau": tau, **c}
    for tau in tau_overlap:
        c = confusion([r["overlap_density"] >= tau for r in scored])
        sweeps["N2"].append({"tau": tau, **c})
        if best["N2"] is None or c["accuracy"] > best["N2"]["accuracy"]:
            best["N2"] = {"tau": tau, **c}
    for tau in tau_width:
        c = confusion([not (r["frechet_width"] is not None
                            and r["frechet_width"] > tau) for r in scored])
        sweeps["N3"].append({"tau": tau, **c})
        if best["N3"] is None or c["accuracy"] > best["N3"]["accuracy"]:
            best["N3"] = {"tau": tau, **c}
    metrics["best_swept"] = best
    metrics["sweeps"] = sweeps

    by_n = {}
    for n in (2, 3, 4):
        sub = [r for r in scored if r["n_vars"] == n]
        ws = [r["frechet_width"] for r in sub if r["frechet_width"] is not None]
        by_n[f"n={n}"] = {
            "rows": len(sub),
            "width_min": float(np.min(ws)) if ws else None,
            "width_median": float(np.median(ws)) if ws else None,
            "width_max": float(np.max(ws)) if ws else None,
        }
    metrics["frechet_width_by_n"] = by_n

    cand = [r for r in scored if label(r) and r["frechet_width"] is not None]
    cand.sort(key=lambda r: (-r["frechet_width"], r["instance_id"],
                             json.dumps(r["target"])))
    priority = [dict(r, reason="widest_frechet_vs_certificate")
                for r in cand[:s_cap]]
    discordant = [dict(r, reason="certificate_unrecoverable_engine_undecided")
                  for r in scored if not label(r)]
    metrics["S_star_size"] = len(priority)
    metrics["S_star_rule"] = (f"top-{s_cap} certificate-RECOVERABLE rows by "
                              "corrected Frechet width (ties: id, target)")
    return {"metrics": metrics, "priority_sample": priority + discordant}


def evaluate_nulls(rows: list[dict], cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    scored = [score_row(row, cfg) for row in rows]
    out = aggregate_results(scored, cfg)
    out["scored"] = scored
    return out
