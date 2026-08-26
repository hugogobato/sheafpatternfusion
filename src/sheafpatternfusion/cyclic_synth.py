"""WP2.5.4 forced cyclic-poset stratum.

Phase 2's sampler could never realize a cyclic overlap hypergraph: r-CPT
entries are drawn in (0.25, 0.75), so EVERY 2^n response pattern has positive
probability, the realized family is the full simplex of observed-sets, and
the simplex is Berge-acyclic. The stratum is therefore forced directly:
indicator mechanisms receive exact 0/1 pins (fixed_cpt), which remove
patterns and can leave a cyclic overlap hypergraph.

Two proposal families:

  templates   hand constructions with guaranteed cyclic realization:
              triangle (n=3): R_k deterministically watches the other two
              variables with an equality/inequality event; an odd number of
              inequality events makes the three pair-strata pairwise
              satisfiable but jointly inconsistent, so the top pattern is
              unrealized while the three pairs are maximal -> Berge cycle.
              square (n=4): four equality/inequality events around a cycle
              of variable pairs; any proper subset is satisfiable, all four
              indicators jointly are not.
  rejection   random structures (var DAG + two-variable watcher parents)
              with per-entry pins applied with probability p_pin; accepted
              iff the realized poset shape is 'cyclic'.

Jobs carry their fixed_cpt so `instantiate` reproduces the instance exactly;
downstream stages run the identical Phase-2 pipeline on them.
"""
from __future__ import annotations

import json

import numpy as np

from .enumerate_structures import instantiate, poset_shape, var_dags


def _eqneq_pins(node: int, neq: bool) -> list[dict]:
    out = []
    for a in (0, 1):
        for b in (0, 1):
            val = float((a != b) if neq else (a == b))
            out.append({"kind": "r", "node": node,
                        "parents": [a, b], "p": float(val)})
    return out


def triangle_template(seed: int) -> dict:
    g = np.random.default_rng(seed)
    signs = [int(g.integers(0, 2)) for _ in range(3)]
    if sum(signs) % 2 == 0:
        signs[0] ^= 1
    watchers = [(1, 2), (0, 2), (0, 1)]
    fixed = []
    for k in range(3):
        fixed += _eqneq_pins(k, bool(signs[k]))
    vp = {0: (), 1: (0,), 2: (0,) if g.random() < 0.5 else (0, 1)}
    return {"template": "triangle_n3", "var_parents": vp,
            "r_parents": watchers, "fixed_cpt": fixed}


def square_template(seed: int) -> dict:
    g = np.random.default_rng(seed)
    signs = [int(g.integers(0, 2)) for _ in range(4)]
    if sum(signs) % 2 == 0:
        signs[0] ^= 1
    watchers = [(1, 2), (2, 3), (3, 0), (0, 1)]
    fixed = []
    for k in range(4):
        fixed += _eqneq_pins(k, bool(signs[k]))
    vp = {0: (), 1: (0,), 2: (0, 1),
          3: (0,) if g.random() < 0.5 else (0, 2)}
    return {"template": "square_n4", "var_parents": vp,
            "r_parents": watchers, "fixed_cpt": fixed}


TEMPLATES = {"triangle_n3": triangle_template, "square_n4": square_template}


def random_pinned_proposal(n_vars: int, p_pin: float,
                           rng: np.random.Generator) -> dict:
    from .lp_ground_truth import param_spec

    vds = var_dags(n_vars)
    vd = vds[int(rng.integers(0, len(vds)))]
    rp = []
    for i in range(n_vars):
        pool = list(range(n_vars))
        pick = rng.choice(len(pool), size=2, replace=False)
        rp.append(tuple(sorted(int(pool[k]) for k in pick)))
    base_seed = int(rng.integers(0, 2**31))
    inst = instantiate((vd, tuple(rp)), seed=base_seed)
    spec = param_spec(inst)
    fixed = []
    for idx, (kind, i, k) in enumerate(spec):
        if kind == "r" and rng.random() < p_pin:
            fixed.append({"kind": "r", "node": i, "parents": list(k),
                          "p": float(rng.integers(0, 2))})
    return {"template": f"rejection_n{n_vars}", "var_parents": vd,
            "r_parents": rp, "fixed_cpt": fixed}


def realize(proposal: dict, draw_seed: int):
    inst = instantiate(
        ({int(k): tuple(v) for k, v in proposal["var_parents"].items()},
         tuple(tuple(p) for p in proposal["r_parents"])),
        seed=draw_seed,
        fixed_cpt=proposal.get("fixed_cpt") or [])
    jt = inst.joint_table()
    patterns = inst.realized_patterns(jt=jt)
    return inst, patterns, poset_shape(patterns)


def signature(proposal: dict) -> str:
    payload = {
        "var_parents": {str(k): sorted(v) for k, v in proposal["var_parents"].items()},
        "r_parents": [list(p) for p in proposal["r_parents"]],
        "fixed_cpt": sorted([str(f["node"]) + ":" + ",".join(map(str, f["parents"]))
                             + ":" + str(int(f["p"]))
                             for f in proposal.get("fixed_cpt", [])]),
    }
    return json.dumps(payload, sort_keys=True)


def make_cyclic_jobs(cfg: dict, shard_idx: int = -1) -> tuple[list[dict], dict]:
    """Deterministic rejection loop; returns accepted job dicts and stats.

    Each job: {iid, n_vars, structure, fixed_cpt, draw_seed, template, tag}.
    `target_per_shard` caps accepted jobs for this shard; `max_attempts` is
    the global attempt budget partitioned across shards (honest-attempt
    floor per plan D4).
    """
    seed_base = int(cfg["seed_base"])
    target_total = int(cfg.get("target_per_shard", 140))
    max_attempts = int(cfg.get("max_attempts", 40000))
    n_shards = int(cfg.get("shards", 4))
    mix = cfg.get("proposal_mix", {"triangle_n3": 0.45, "square_n4": 0.25,
                                   "rejection_n3": 0.20, "rejection_n4": 0.10})
    lo, hi = _shard_range(shard_idx, n_shards, max_attempts)

    jobs = []
    stats = {"attempts": 0, "accepted": 0, "rejected_shape": 0,
             "rejected_no_target": 0, "duplicate": 0}
    for t in range(lo, hi):
        if len(jobs) >= target_total:
            break
        rng = np.random.default_rng(seed_base + t)
        name, u = _pick_proposal(mix, rng)
        if name.startswith("rejection"):
            prop = random_pinned_proposal(3 if name.endswith("n3") else 4,
                                          float(rng.uniform(0.35, 0.6)), rng)
            draw_seed = int(rng.integers(0, 2**31))
        else:
            prop = TEMPLATES[name](seed_base * 31 + t)
            draw_seed = int(rng.integers(0, 2**31))
        stats["attempts"] += 1
        try:
            inst, patterns, shape = realize(prop, draw_seed)
        except Exception:
            continue
        if shape != "cyclic":
            stats["rejected_shape"] += 1
            continue
        partial = [i for i in range(inst.n_vars)
                   if any(r[i] == 0 for r in patterns)
                   and any(r[i] == 1 for r in patterns)]
        if not partial:
            stats["rejected_no_target"] += 1
            continue
        stats["accepted"] += 1
        jobs.append({
            "iid": f"cyc{inst.n_vars}_t{t:06d}",
            "n_vars": inst.n_vars,
            "structure": {
                "var_parents": {str(k): list(v)
                                for k, v in prop["var_parents"].items()},
                "r_parents": [list(p) for p in prop["r_parents"]]},
            "fixed_cpt": prop["fixed_cpt"],
            "draw_seed": draw_seed,
            "template": prop["template"],
            "tag": "cyclic",
        })
    return jobs, stats


def _shard_range(shard_idx: int, n_shards: int, total: int):
    if shard_idx < 0:
        return 0, total
    step = total // n_shards
    lo = shard_idx * step
    hi = total if shard_idx == n_shards - 1 else (shard_idx + 1) * step
    return lo, hi


def _pick_proposal(mix: dict, rng: np.random.Generator) -> tuple[str, float]:
    names = sorted(mix.keys())
    probs = np.array([mix[n] for n in names], dtype=float)
    probs = probs / probs.sum()
    k = int(rng.choice(len(names), p=probs))
    return names[k], float(probs[k])


def run_cyclic_instance(job: dict, budgets: dict, ci_draws: int) -> list[dict]:
    """Identical Phase-2 pipeline record for one forced-cyclic job."""
    from .engine2 import decide2, sheaf_fiber_verdict
    from .enumerate_structures import (
        classify,
        conflict_flags,
        discover_slice_cis,
        graham_acyclic,
        instantiate as inst_instantiate,
        pick_targets,
        poset_shape as pshape,
    )
    from .gluing import marginal_problem_lp
    from .lp_ground_truth import pack, unpack

    vp = {int(k): tuple(v) for k, v in job["structure"]["var_parents"].items()}
    structure = (vp, tuple(tuple(p) for p in job["structure"]["r_parents"]))
    inst = inst_instantiate(structure, seed=job["draw_seed"],
                            fixed_cpt=job.get("fixed_cpt") or [])
    info = classify(inst)
    jt = inst.joint_table()
    patterns = inst.realized_patterns(jt=jt)
    q = inst.observed_laws(jt)
    pp = {}
    for (v, r), p in jt.items():
        pp[r] = pp.get(r, 0.0) + p
    fam_w = {r: {o: c * pp[r] for o, c in cells.items()} for r, cells in q.items()}
    completability = marginal_problem_lp(inst.n_vars, fam_w)["feasible"]
    sets = [frozenset(i for i in range(inst.n_vars) if r[i] == 1)
            for r in patterns]
    shape = pshape(patterns)
    conflicts = conflict_flags(inst)
    cis = discover_slice_cis(inst, n_draws=ci_draws)
    theta_true = pack(inst)

    records = []
    for tgt in pick_targets(inst):
        eng = decide2(inst, theta_true, tgt, seed=11)
        if eng["gt_verdict"].startswith("UNDETERMINED"):
            mult = int(budgets.get("undecided_round2_multiplier", 2))
            eng = decide2(inst, theta_true, tgt,
                          jump_starts=int(budgets["jump_starts"]) * mult,
                          seed=23)
            eng["gt_evidence"] = "round2:" + eng["gt_evidence"]
        fib = sheaf_fiber_verdict(inst, theta_true, tgt,
                                  n_starts=int(budgets["fiber_starts"]), seed=13)
        records.append({
            "instance_id": job["iid"],
            "tag": "cyclic",
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
            "jacobian_full_rank": bool(fib["jacobian_rank"] == fib["n_free_params"]),
            "observed_family_completable": bool(completability),
            "conflict_mcar_style": conflicts["conflict_mcar_style"],
            "max_cross_pattern_marginal_gap":
                conflicts["max_cross_pattern_marginal_gap"],
            "n_slice_ci_constraints": int(sum(len(v) for v in cis.values())),
            "slice_cis": {"".join(map(str, r)): [list(c) for c in lst]
                          for r, lst in cis.items() if lst},
        })
    return records
