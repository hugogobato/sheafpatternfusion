"""Phase 3 pivot-gate probes (WP3.0a / WP3.0b / WP3.0c).

Everything the Colab notebook fleet embeds or imports for Phase 3 lives here;
the module is deliberately self-contained relative to its sibling modules so
that scripts/make_colab_phase3.py can concatenate it into standalone runners.

WP3.0a  natural-prevalence scan utilities: realized-pattern counting on raw
        missingness masks, Berge-cyclicity readouts (reusing the frozen
        graham_acyclic), partial-overlap flags, column-permutation negative
        controls, and a fast bootstrap for dataset-level cyclic fractions.
WP3.0b  scaling probe at n=5 (n=6 arm): uniform structure sampling beyond the
        exhaustive Phase-2 space, and a timing-split replica of the exact
        Phase-2 decision pipeline (engine round1+round2 unchanged, fiber
        certificate, share-pinned Frechet features) plus fixed-budget
        attacker runs on undecided x RECOVERABLE rows.
WP3.0c  signal-validity utilities: pin-aware instance rebuilding (cyclic
        stratum rows carry indicator pins that battery.instance_from_row
        ignores), tie-corrected rank AUCs, stratified label-permutation
        nulls, and the downstream spread-vs-naive-pooling-error correlation.

No function here upgrades verdicts: all decision logic reuses the frozen
Phase-1/2 primitives with identical seeds, budgets, and tolerances.
"""
from __future__ import annotations

import json
import time
import zlib
import base64
from collections import Counter

import numpy as np


# --------------------------------------------------------------------------
# pin-aware instance reconstruction (cyclic-stratum rows carry fixed_cpt)
# --------------------------------------------------------------------------

def instance_from_row_fixed(row: dict):
    """battery.instance_from_row, extended to honor fixed_cpt pins.

    Cyclic-stratum records freeze indicator mechanisms at exact 0/1 values;
    ignoring them realizes the wrong pattern family (the full simplex), so
    every Phase-3 rebuild MUST go through this constructor."""
    from .enumerate_structures import instantiate
    from .lp_ground_truth import pack, unpack

    vp = {int(k): tuple(v) for k, v in row["var_parents"].items()}
    structure = (vp, tuple(tuple(p) for p in row["r_parents"]))
    inst = instantiate(structure, seed=row["seed"],
                       fixed_cpt=row.get("fixed_cpt") or [])
    m = unpack(inst, pack(inst))
    jt = m.joint_table()
    q = m.observed_laws(jt)
    pp: dict = {}
    for (v, r), p in jt.items():
        pp[r] = pp.get(r, 0.0) + p
    return inst, q, pp


def attack_row_fixed(row: dict, cfg: dict | None = None) -> dict:
    """attack_row replica for pinned rows: identical attacker stack (A1 -> A2
    -> A3, non-shared oracles, stable seeds) but rebuilding the instance with
    instance_from_row_fixed."""
    from .attackers import (
        completion_enumeration,
        deepened_witness_search,
        frechet_cell_scan,
        _serialize_witness,
        _stable_seed,
    )
    from .lp_ground_truth import pack

    cfg = cfg or {}
    inst, q, pp = instance_from_row_fixed(row)
    theta = pack(inst)
    target = tuple(row["target"])

    rec = {
        "instance_id": row["instance_id"],
        "target": list(target),
        "n_vars": row["n_vars"],
        "mechanism_class": row.get("mechanism_class"),
        "poset_shape": row.get("poset_shape"),
        "certificate_sheaf": row.get("sheaf_recoverable"),
        "strata": row.get("_strata", []),
        "pinned": bool(row.get("fixed_cpt")),
    }

    a1 = deepened_witness_search(inst, theta, target, cfg,
                                 seed=_stable_seed(row, "A1"))
    rec["A1"] = {k: v for k, v in a1.items() if k != "witness"}
    rec["A1"]["has_witness"] = a1["confirmed_false_recoverable"]
    rec["_a1_witness"] = _serialize_witness(a1)

    if a1["confirmed_false_recoverable"]:
        a2 = {"attacker": "A2_completion_enumeration", "skipped": True,
              "reason": "A1 already confirmed"}
    else:
        a2 = completion_enumeration(inst, theta, target, cfg,
                                    seed=_stable_seed(row, "A2"))
        rec["_a2_witness"] = _serialize_witness(a2)
    rec["A2"] = {k: v for k, v in a2.items()
                 if k not in ("witness", "lp_vertices")}
    rec["A2"]["has_witness"] = bool(a2.get("confirmed_false_recoverable"))
    rec["A2"]["lp_vertices"] = a2.get("lp_vertices")

    a3 = frechet_cell_scan(inst, q, pp, target, cfg)
    rec["A3"] = a3

    confirmed = a1["confirmed_false_recoverable"] or \
        a2.get("confirmed_false_recoverable", False)
    wit = None
    if a1["confirmed_false_recoverable"]:
        wit = a1["witness"]
    elif a2.get("confirmed_false_recoverable"):
        wit = a2["witness"]
    rec["verdict"] = "CONFIRMED_FALSE_RECOVERABLE" if confirmed \
        else "NO_FALSE_RECOVERABLE_FOUND"
    rec["confirming_route"] = wit["route"] if wit else None
    rec["total_wall_s"] = sum(a["wall_s"] for a in (rec["A1"], rec["A2"], rec["A3"])
                              if isinstance(a, dict) and "wall_s" in a)
    return rec


# --------------------------------------------------------------------------
# WP3.0b: uniform structure sampling beyond the Phase-2 space
# --------------------------------------------------------------------------

def sample_structures(n_vars: int, count: int, seed: int,
                      prefix: str) -> list[dict]:
    """Uniformly sample structures on n_vars binary variables.

    Variable DAGs are drawn uniformly from the topologically ordered family
    (identical to enumerate_structures.var_dags); each R_i parent set is
    drawn uniformly over ALL 2^n_vars subsets (identical to the
    enumerate_structures.r_mechanisms distribution Phase 2 sampled from).
    Deterministic given seed; duplicate structures rejected."""
    from .enumerate_structures import var_dags

    rng = np.random.default_rng(seed)
    vds = var_dags(n_vars)
    jobs: list[dict] = []
    seen = set()
    guard = 0
    while len(jobs) < count and guard < 200 * count:
        guard += 1
        vd = vds[int(rng.integers(0, len(vds)))]
        rp = tuple(tuple(int(i) for i in np.flatnonzero(rng.random(n_vars) < 0.5))
                   for _ in range(n_vars))
        key = (tuple(sorted((k, tuple(v)) for k, v in vd.items())), rp)
        if key in seen:
            continue
        seen.add(key)
        jobs.append({
            "iid": f"{prefix}_j{len(jobs):04d}",
            "n_vars": n_vars,
            "structure": {
                "var_parents": {str(k): list(v) for k, v in vd.items()},
                "r_parents": [list(p) for p in rp]},
            "draw_seed": int(rng.integers(0, 2 ** 31)),
        })
    return jobs


# --------------------------------------------------------------------------
# WP3.0b: timing-split replica of the Phase-2 decision pipeline
# --------------------------------------------------------------------------

def decide2_timed(inst, theta_true: np.ndarray, target,
                  jump_starts: int = 40, round2_multiplier: int = 2,
                  lp_pinch_tol: float = 1e-9, lp_width_tol: float = 1e-3,
                  seed: int = 0) -> dict:
    """engine2.decide2 with per-instrument wall times. Verdict semantics are a
    line-for-line replica: formula oracle acceptance at 1e-8 relative
    tolerance, LP pinch at lp_pinch_tol, root-jump witness rounds (fallback
    walk kept iff strictly better), relaxed-fragile threshold at
    lp_width_tol. Round 2 reruns with jump_starts * round2_multiplier and
    seed offset exactly like the Phase-2 protocol."""
    from .engine2 import formula_oracle
    from .lp_ground_truth import lp_range, root_jump_search, unpack

    out: dict = {}
    walls = {}
    t0 = time.perf_counter()
    fname = formula_oracle(inst, theta_true, target)
    walls["formula"] = time.perf_counter() - t0
    if fname is not None:
        out.update(gt_verdict="RECOVERABLE", gt_evidence=f"formula:{fname}")
        out["walls"] = walls
        return out

    m_true = unpack(inst, theta_true)
    q = m_true.observed_laws()
    out["true_value"] = _target_value(m_true, target)

    if target[0] in ("mean", "cell"):
        t0 = time.perf_counter()
        rng_lp = lp_range(inst, q, target)
        walls["lp"] = time.perf_counter() - t0
        out["lp"] = {"width": rng_lp["width"],
                     "lo": rng_lp["lo"], "hi": rng_lp["hi"]}
        if rng_lp["width"] <= lp_pinch_tol:
            out.update(gt_verdict="RECOVERABLE", gt_evidence="lp_pinched")
            out["walls"] = walls
            return out

    t0 = time.perf_counter()
    wit = root_jump_search(inst, theta_true, target,
                           n_starts=jump_starts, seed=seed)
    walls["witness_r1"] = time.perf_counter() - t0
    if not wit["success"]:
        walk = root_jump_search(inst, theta_true, target,
                                n_starts=jump_starts, seed=seed + 101)
        if walk["delta_phi"] > wit["delta_phi"]:
            wit = walk
            walls["witness_r1_extra"] = True
    out["witness"] = {k: wit[k] for k in ("delta_phi", "dist", "success")}
    if wit["success"]:
        out.update(gt_verdict="UNRECOVERABLE",
                   gt_evidence=f"model_witness(rootjump) dphi={wit['delta_phi']:.4f} "
                               f"dist={wit['dist']:.1e}")
        out["walls"] = walls
        return out

    t0 = time.perf_counter()
    wit2 = root_jump_search(inst, theta_true, target,
                            n_starts=jump_starts * int(round2_multiplier),
                            seed=seed + 12)
    walls["witness_r2"] = time.perf_counter() - t0
    if not wit2["success"]:
        walk = root_jump_search(inst, theta_true, target,
                                n_starts=jump_starts * int(round2_multiplier),
                                seed=seed + 113)
        if walk["delta_phi"] > wit2["delta_phi"]:
            wit2 = walk
            walls["witness_r2_extra"] = True
    if wit2["delta_phi"] > out["witness"]["delta_phi"]:
        out["witness"] = {k: wit2[k] for k in ("delta_phi", "dist", "success")}
    if wit2["success"]:
        out.update(gt_verdict="UNRECOVERABLE",
                   gt_evidence="round2:" +
                               f"model_witness(rootjump) dphi={wit2['delta_phi']:.4f} "
                               f"dist={wit2['dist']:.1e}")
        out["walls"] = walls
        return out

    if out.get("lp", {}).get("width", 0.0) > lp_width_tol:
        out.update(gt_verdict="UNDETERMINED_RELAXED_FRAGILE",
                   gt_evidence="round2:no model witness; relaxation varies")
    else:
        out.update(gt_verdict="UNDETERMINED",
                   gt_evidence="round2:no certificate either way")
    out["walls"] = walls
    return out


def _target_value(m, target) -> float:
    from .lp_ground_truth import target_value_phi
    return target_value_phi(m, target)


def run_scaling_job(job: dict, cfg: dict) -> list[dict]:
    """Full WP3.0b pipeline for one structure: engine round1(+round2),
    fiber certificate, structural annotations, share-pinned Frechet features,
    and (when job['do_attack']) a fixed-budget attacker run on undecided x
    RECOVERABLE rows. Row schema extends the Phase-2 merge schema with wall
    splits and feature fields; `job['fixed_cpt']` is honored when present."""
    from .battery import frechet_bounds, fraction_observed, overlap_density
    from .engine2 import sheaf_fiber_verdict
    from .enumerate_structures import (
        classify,
        conflict_flags,
        discover_slice_cis,
        graham_acyclic,
        instantiate,
        pick_targets,
        poset_shape,
    )
    from .gluing import marginal_problem_lp
    from .lp_ground_truth import pack, unpack

    budgets = cfg["budgets"]
    t0 = time.perf_counter()
    vp = {int(k): tuple(v) for k, v in job["structure"]["var_parents"].items()}
    structure = (vp, tuple(tuple(p) for p in job["structure"]["r_parents"]))
    inst = instantiate(structure, seed=job["draw_seed"],
                       fixed_cpt=job.get("fixed_cpt") or [])
    info = classify(inst)
    jt = inst.joint_table()
    patterns = inst.realized_patterns(jt=jt)
    q = inst.observed_laws(jt)
    pp: dict = {}
    for (v, r), p in jt.items():
        pp[r] = pp.get(r, 0.0) + p
    fam_w = {r: {o: c * pp[r] for o, c in cells.items()} for r, cells in q.items()}
    completability = marginal_problem_lp(inst.n_vars, fam_w)["feasible"]
    sets = [frozenset(i for i in range(inst.n_vars) if r[i] == 1) for r in patterns]
    shape = poset_shape(patterns)
    conflicts = conflict_flags(inst)
    cis = discover_slice_cis(inst, n_draws=int(budgets.get("ci_discovery_draws", 16)))
    theta_true = pack(inst)
    wall_struct = time.perf_counter() - t0

    records = []
    for tgt in pick_targets(inst):
        eng = decide2_timed(
            inst, theta_true, tgt,
            jump_starts=int(budgets["jump_starts"]),
            round2_multiplier=int(budgets.get("round2_multiplier", 2)),
            lp_pinch_tol=float(budgets.get("lp_pinch_tol", 1e-9)),
            lp_width_tol=float(budgets.get("lp_width_tol", 1e-3)),
            seed=11)
        walls = eng.pop("walls")
        undecided = eng["gt_verdict"].startswith("UNDETERMINED")

        t0 = time.perf_counter()
        fib = sheaf_fiber_verdict(inst, theta_true, tgt,
                                  n_starts=int(budgets["fiber_starts"]),
                                  max_roots=int(budgets.get("max_roots", 12)),
                                  seed=13)
        wall_fiber = time.perf_counter() - t0

        t0 = time.perf_counter()
        fb = frechet_bounds(inst.n_vars, q, pp, tgt)
        wall_frechet = time.perf_counter() - t0

        do_attack = bool(job.get("do_attack")) and undecided \
            and fib["sheaf_verdict"] == "RECOVERABLE"
        attack_rec = None
        if do_attack:
            rowlike = {
                "instance_id": job["iid"],
                "target": list(tgt),
                "n_vars": inst.n_vars,
                "var_parents": {str(k): list(v) for k, v in vp.items()},
                "r_parents": [list(p) for p in structure[1]],
                "seed": job["draw_seed"],
                "fixed_cpt": job.get("fixed_cpt"),
                "mechanism_class": info["mechanism_class"],
                "poset_shape": shape,
                "sheaf_recoverable": fib["sheaf_verdict"],
            }
            try:
                t0 = time.perf_counter()
                attack_rec = attack_row_fixed(rowlike, cfg.get("attack"))
                attack_rec["wall_dispatch_s"] = time.perf_counter() - t0
            except Exception as e:  # never lose the engine row to an attack crash
                attack_rec = {"status": "error", "error": f"{type(e).__name__}: {e}"}

        records.append({
            "instance_id": job["iid"],
            "tag": job.get("tag", f"n{inst.n_vars}"),
            "seed": job["draw_seed"],
            "template": job.get("template"),
            "fixed_cpt": job.get("fixed_cpt"),
            "n_vars": inst.n_vars,
            "var_parents": {str(k): list(v) for k, v in vp.items()},
            "r_parents": [list(p) for p in structure[1]],
            "mechanism_class": info["mechanism_class"],
            "has_self_edge": info["has_self_edge"],
            "poset_shape": shape,
            "graham_acyclic": bool(graham_acyclic(sets)),
            "n_realized_patterns": len(patterns),
            "patterns": [list(p) for p in patterns],
            "always_observed": list(info["always_observed"]),
            "never_observed": list(info["never_observed"]),
            "target": list(tgt),
            "true_value": eng.get("true_value"),
            "gt_recoverable": eng["gt_verdict"],
            "gt_evidence": eng["gt_evidence"],
            "lp_width": eng.get("lp", {}).get("width"),
            "witness_delta_phi": eng.get("witness", {}).get("delta_phi"),
            "sheaf_recoverable": fib["sheaf_verdict"],
            "phi_spread_over_fiber": fib["phi_spread_over_fiber"],
            "n_distinct_completions": fib["n_distinct_completions"],
            "jacobian_rank": fib["jacobian_rank"],
            "n_free_params": fib["n_free_params"],
            "jacobian_rank_deficiency": int(fib["n_free_params"] - fib["jacobian_rank"]),
            "jacobian_full_rank": bool(fib["jacobian_rank"] == fib["n_free_params"]),
            "observed_family_completable": bool(completability),
            "conflict_mcar_style": conflicts["conflict_mcar_style"],
            "max_cross_pattern_marginal_gap": conflicts["max_cross_pattern_marginal_gap"],
            "n_slice_ci_constraints": int(sum(len(v) for v in cis.values())),
            "frechet_lo": fb["lo"],
            "frechet_hi": fb["hi"],
            "frechet_width": fb["width"],
            "frac_observed": round(fraction_observed(pp, inst.n_vars), 6),
            "overlap_density": round(overlap_density([tuple(p) for p in patterns]), 6),
            "attack_requested": bool(job.get("do_attack")) and undecided
            and fib["sheaf_verdict"] == "RECOVERABLE",
            "attack": attack_rec,
            "wall_struct_s": round(wall_struct, 3),
            "wall_formula_s": round(walls.get("formula", 0.0), 3),
            "wall_lp_s": round(walls.get("lp", 0.0), 3),
            "wall_engine_r1_s": round(walls.get("witness_r1", 0.0), 3),
            "wall_engine_r2_s": round(walls.get("witness_r2", 0.0), 3),
            "wall_fiber_s": round(wall_fiber, 3),
            "wall_features_s": round(wall_frechet, 3),
            "wall_attack_s": round(attack_rec.get("total_wall_s", 0.0), 3)
            if isinstance(attack_rec, dict) else 0.0,
        })
    return records


# --------------------------------------------------------------------------
# WP3.0a: prevalence scan utilities
# --------------------------------------------------------------------------

def realized_pattern_counts(obs: np.ndarray) -> dict[tuple, int]:
    """Counts of realized observed-set patterns. `obs` is a boolean matrix
    (n_rows, k), True = observed; pattern tuples follow the engine convention
    (1 = observed)."""
    codes = _pattern_codes(obs)
    counts = np.bincount(codes, minlength=1 << obs.shape[1])
    out = {}
    for code in range(1 << obs.shape[1]):
        if counts[code]:
            pat = tuple((code >> i) & 1 for i in range(obs.shape[1]))
            out[pat] = int(counts[code])
    return out


def _pattern_codes(obs: np.ndarray) -> np.ndarray:
    weights = 1 << np.arange(obs.shape[1])
    return obs.astype(np.int64) @ weights


def scan_subsets(obs: np.ndarray, col_names: list[str],
                 subsets: list[tuple[int, ...]], min_patterns: int = 4,
                 min_support: int = 1) -> list[dict]:
    """Per-subset prevalence records for WP3.0a. Cyclicity is Berge-cyclicity
    of the observed-set hypergraph restricted to patterns whose support is at
    least min_support (main analysis: min_support=1, i.e., any realized
    pattern; robustness: min_support>1)."""
    from .enumerate_structures import graham_acyclic

    records = []
    for sub in subsets:
        sub_obs = obs[:, list(sub)]
        counts = realized_pattern_counts(sub_obs)
        kept = {p: c for p, c in counts.items() if c >= min_support}
        n_patterns = len(kept)
        rec = {
            "cols": [col_names[i] for i in sub],
            "size": len(sub),
            "n_realized_patterns": n_patterns,
            "eligible": n_patterns >= min_patterns,
        }
        if rec["eligible"]:
            sets = [frozenset(i for i, bit in enumerate(p) if bit == 1)
                    for p in sorted(kept)]
            acyclic = graham_acyclic(sets)
            nested = _nested_only(sorted(kept))
            rec.update({
                "graham_acyclic": bool(acyclic),
                "cyclic": not acyclic,
                "nested_only": bool(nested),
                "partial_overlap": bool(not nested),
                "max_support_gap": int(max(kept.values()) - min(kept.values())),
            })
        records.append(rec)
    return records


def _nested_only(patterns: list[tuple]) -> bool:
    sets = [frozenset(i for i, bit in enumerate(p) if bit == 1) for p in patterns]
    return all(a <= b or b <= a for a, b in
               [(x, y) for i, x in enumerate(sets) for y in sets[i + 1:]])


def column_permutation_control(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Negative control: independently permute each column's missingness mask,
    destroying co-missingness dependence while preserving marginals."""
    out = np.empty_like(obs)
    for j in range(obs.shape[1]):
        out[:, j] = obs[rng.permutation(obs.shape[0]), j]
    return out


def cyclic_fraction_bootstrap(obs: np.ndarray, subsets: list[tuple[int, ...]],
                              B: int, min_patterns: int, min_support: int,
                              seed: int) -> dict:
    """Dataset-level bootstrap of the cyclic fraction over pre-selected
    eligible subsets (eligibility frozen at the full sample; realized patterns
    recount per resample). Returns fraction quantiles and per-subset stability."""
    from .enumerate_structures import graham_acyclic

    rng = np.random.default_rng(seed)
    n = obs.shape[0]
    fracs = []
    stable = np.zeros(len(subsets))
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        obs_b = obs[idx]
        cyc = 0
        elig = 0
        for si, sub in enumerate(subsets):
            counts = realized_pattern_counts(obs_b[:, list(sub)])
            kept = {p: c for p, c in counts.items() if c >= min_support}
            if len(kept) < min_patterns:
                continue
            elig += 1
            sets = [frozenset(i for i, bit in enumerate(p) if bit == 1)
                    for p in sorted(kept)]
            if not graham_acyclic(sets):
                cyc += 1
                stable[si] += 1
        fracs.append(cyc / max(elig, 1))
    fracs = np.array(fracs)
    return {
        "B": B,
        "fraction_q05": float(np.quantile(fracs, 0.05)),
        "fraction_median": float(np.median(fracs)),
        "fraction_q95": float(np.quantile(fracs, 0.95)),
        "per_subset_cyclic_stability": [round(float(s / B), 4) for s in stable],
    }


# --------------------------------------------------------------------------
# WP3.0c: signal-validity utilities
# --------------------------------------------------------------------------

def naive_pooling_mean(m, j: int) -> float:
    """MCAR-plugin estimate of E[V_j]: probability-weighted average of the
    pattern-conditional means over the patterns that observe j (weights are
    the population pattern probabilities, normalized among observers)."""
    jt = m.joint_table()
    q = m.observed_laws(jt)
    pp: dict = {}
    for (v, r), p in jt.items():
        pp[r] = pp.get(r, 0.0) + p
    num = den = 0.0
    for r, cells in q.items():
        if r[j] != 1:
            continue
        w = pp.get(r, 0.0)
        pos = sum(1 for kk in range(len(r)) if r[kk] == 1 and kk < j)
        mg: dict[int, float] = {}
        for o, c in cells.items():
            mg[o[pos]] = mg.get(o[pos], 0.0) + c
        num += w * sum(k * c for k, c in mg.items())
        den += w * sum(mg.values())
    return num / den if den > 0 else float("nan")


def spread_naive_table(rows: list[dict]) -> list[dict]:
    """Downstream add-on inputs: for each rebuildable row, (fiber spread,
    corrected Frechet width, cross-pattern gap) vs absolute naive-pooling
    error against the true estimand."""
    from .lp_ground_truth import target_value_phi

    out = []
    for row in rows:
        try:
            inst, q, pp = instance_from_row_fixed(row)
            m = unpack_model(inst)
            j = int(row["target"][1])
            phi_true = target_value_phi(m, tuple(row["target"]))
            est = naive_pooling_mean(m, j)
            out.append({
                "instance_id": row.get("instance_id"),
                "source": row.get("source_tag", "unspecified"),
                "spread": float(row.get("phi_spread_over_fiber", 0.0) or 0.0),
                "naive_abs_err": abs(est - phi_true),
                "frechet_width": row.get("frechet_width"),
                "max_gap": row.get("max_cross_pattern_marginal_gap"),
            })
        except Exception:
            continue
    return out


def unpack_model(inst):
    from .lp_ground_truth import pack, unpack
    return unpack(inst, pack(inst))


def rank_auc(scores, labels) -> float | None:
    """Tie-corrected Mann-Whitney AUC; None when only one class present."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sp = s[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def permutation_auc_p(scores, labels, strata, B: int,
                      seed: int) -> dict:
    """Stratified label-permutation null for the AUC: labels are shuffled
    within strata (e.g., n_vars x mechanism_class buckets), preserving class
    balance per bucket. One-sided p = P(AUC_perm >= AUC_obs) with add-one
    smoothing."""
    rng = np.random.default_rng(seed)
    y = np.asarray(labels, dtype=int)
    strata = np.asarray(strata)
    auc_obs = rank_auc(scores, y)
    if auc_obs is None:
        return {"auc": None, "p_value": None, "B": B,
                "reason": "single-class labels"}
    ge = 0
    null_aucs = []
    for _ in range(B):
        yp = np.empty_like(y)
        for st in np.unique(strata):
            mask = strata == st
            vals = y[mask]
            perm = rng.permutation(vals)
            yp[mask] = perm
        a = rank_auc(scores, yp)
        if a is not None:
            null_aucs.append(a)
            if a >= auc_obs:
                ge += 1
    return {
        "auc": auc_obs,
        "p_value": (1 + ge) / (B + 1),
        "B": B,
        "null_mean": float(np.mean(null_aucs)) if null_aucs else None,
        "null_sd": float(np.std(null_aucs)) if null_aucs else None,
    }


def permutation_corr_p(x, y, B: int, seed: int, method: str = "spearman") -> dict:
    """One-sided permutation p for |correlation| exceeding the observed value."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    def corr(a, b):
        if method == "spearman":
            a = np.argsort(np.argsort(a)).astype(float)
            b = np.argsort(np.argsort(b)).astype(float)
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    rho = corr(x, y)
    ge = 0
    for _ in range(B):
        if abs(corr(x, rng.permutation(y))) >= abs(rho):
            ge += 1
    return {"rho": rho, "p_two_sided": (1 + ge) / (B + 1),
            "B": B, "method": method, "n": int(len(x))}


# --------------------------------------------------------------------------
# compact payload codec shared by the embedded-notebook fleet
# --------------------------------------------------------------------------

ENGINE_ROW_FIELDS = [
    "instance_id", "tag", "seed", "template", "fixed_cpt", "n_vars",
    "var_parents", "r_parents", "mechanism_class", "has_self_edge",
    "poset_shape", "graham_acyclic", "n_realized_patterns", "patterns",
    "always_observed", "never_observed", "target", "true_value",
    "gt_recoverable", "gt_evidence", "sheaf_recoverable",
    "phi_spread_over_fiber", "n_distinct_completions", "jacobian_rank",
    "n_free_params", "conflict_mcar_style", "max_cross_pattern_marginal_gap",
]


def compact_engine_row(row: dict) -> dict:
    return {k: row[k] for k in ENGINE_ROW_FIELDS if k in row}


def compress_payload(obj, level: int = 9) -> str:
    blob = json.dumps(obj, separators=(",", ":")).encode()
    return base64.b64encode(zlib.compress(blob, level)).decode()


def decompress_payload(b64: str):
    return json.loads(zlib.decompress(base64.b64decode(b64)))
