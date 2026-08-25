"""WP2.5.2 adversarial attackers for RECOVERABLE assertions.

Three instruments, all deliberately NON-SHARED with the certificate's own
oracle (no reuse of the certificate's fiber roots, seeds, or budgets):

  A1  deepened witness search: multi-round root-jumping at ~20x Phase-2
      start budgets plus null-space manifold walks with randomized signs,
      horizons, and fresh seeds, hunting a model pair (two factorized m-graph
      completions) that matches the observed fingerprint but differs on the
      target.
  A2  completion enumeration: fresh-seed multistart root enumeration and
      constructive manifold samplers, plus randomized-objective LP vertex
      harvests of the share-pinned completion polytope (exact rational
      vertices); reports the maximal model-valid pair divergence found.
  A3  Frechet-cell certification: classical route. For every admissible
      pair/triple of realized patterns adjacent through the target variable,
      an LP over the union table asks whether the strata laws can coexist in
      one joint and whether they pin P(V_target=1). Infeasible cells are
      recorded as classical obstructions; a globally degenerate corrected
      interval is recorded as classical certification.

A CONFIRMED false RECOVERABLE requires a MODEL-VALID witness: two completions
with max fingerprint distance < dist_tol whose targets differ by > phi_tol.
A3 can corroborate or tension, never confirm by itself. SLSQP is avoided
everywhere (scipy 1.17.1 heap-corruption incident, see engine2.py).

Every attack logs attacker identity, budget consumed, and wall time so
WP2.5.6 can price certificate-vs-search per row.
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np

from .battery import frechet_bounds, instance_from_row
from .engine2 import collect_roots_early
from .lp_ground_truth import (
    observed_vector,
    pack,
    root_jump_search,
    target_value_phi,
    unpack,
    manifold_walk,
)


def deepened_witness_search(inst, theta_ref: np.ndarray, target,
                            cfg: dict, seed: int) -> dict:
    """A1: escalating root-jump rounds + randomized manifold walks."""
    t0 = time.perf_counter()
    rounds = int(cfg.get("a1_jump_rounds", 3))
    starts_per_round = int(cfg.get("a1_starts_per_round", 200))
    walk_seeds = int(cfg.get("a1_walk_n_seeds", 16))
    walk_steps = int(cfg.get("a1_walk_steps", 80))
    phi_tol = float(cfg.get("phi_tol", 1e-4))
    dist_tol = float(cfg.get("dist_tol", 1e-9))

    best = {"delta_phi": 0.0, "dist": np.inf}
    starts_used = 0
    confirmed = None
    rng = np.random.default_rng(seed)
    for rnd in range(rounds):
        res = root_jump_search(inst, theta_ref, target,
                               n_starts=starts_per_round,
                               seed=int(rng.integers(0, 2**31)))
        starts_used += starts_per_round
        if res["delta_phi"] > best["delta_phi"]:
            best = {"delta_phi": res["delta_phi"], "dist": res["dist"]}
        if res["success"]:
            confirmed = {"route": f"A1_rootjump_r{rnd}",
                         "theta_pair": res["theta_pair"],
                         "phi_values": res["phi_values"]}
            break
        walk = manifold_walk(inst, theta_ref, target,
                             n_seeds=walk_seeds, steps=walk_steps,
                             step_size=float(cfg.get("a1_step_size", 0.02)),
                             seed=int(rng.integers(0, 2**31)))
        if walk["delta_phi"] > best["delta_phi"]:
            best = {"delta_phi": walk["delta_phi"], "dist": walk["dist"]}
        if walk["success"]:
            confirmed = {"route": f"A1_manifoldwalk_r{rnd}",
                         "theta_pair": walk["theta_pair"],
                         "phi_values": walk["phi_values"]}
            break
    return {
        "attacker": "A1_deepened_witness",
        "confirmed_false_recoverable": confirmed is not None,
        "best_delta_phi": float(best["delta_phi"]),
        "witness": confirmed,
        "budget_starts": starts_used,
        "wall_s": time.perf_counter() - t0,
    }


def _lp_vertex_harvest(n_vars: int, q: dict, pp: dict, target,
                       n_obj: int, seed: int) -> dict:
    """Randomized-objective harvest of exact vertices of the share-pinned
    completion polytope over joint cells t[v, r] (same object as
    battery.frechet_bounds); returns diverse rational completions and their
    target spread."""
    from scipy.optimize import linprog

    rng = np.random.default_rng(seed)
    patterns = sorted(q.keys())
    cells = list(itertools.product(
        itertools.product((0, 1), repeat=n_vars), patterns))
    cindex = {c: k for k, c in enumerate(cells)}
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

    j = target[1]
    base_c = np.array([float(v[j]) for v, _ in cells])
    phis = []
    for k in range(n_obj):
        pert = rng.uniform(-0.5, 0.5, size=len(cells))
        c = base_c + 1e-3 * pert if k else base_c
        d = 1.0 if k % 2 == 0 else -1.0
        res = linprog(d * c, A_eq=A, b_eq=b,
                      bounds=[(0.0, 1.0)] * len(cells), method="highs")
        if res.status != 0:
            continue
        phis.append(float(np.dot(base_c, res.x)))
    return {"n_vertices": len(phis),
            "phi_min": min(phis) if phis else None,
            "phi_max": max(phis) if phis else None}


def completion_enumeration(inst, theta_ref: np.ndarray, target,
                           cfg: dict, seed: int) -> dict:
    """A2: fresh-seed root enumeration + manifold continuations + LP vertex
    harvest. Model-valid kill requires two enumerated roots differing on the
    target within tolerances."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    root_starts = int(cfg.get("a2_root_starts", 400))
    max_roots = int(cfg.get("a2_max_roots", 32))
    walk_follows = int(cfg.get("a2_walk_follows", 6))
    walk_seeds = int(cfg.get("a2_walk_n_seeds", 12))
    phi_tol = float(cfg.get("phi_tol", 1e-4))
    dist_tol = float(cfg.get("dist_tol", 1e-9))

    m_ref = unpack(inst, theta_ref)
    patterns = m_ref.realized_patterns(jt=m_ref.joint_table())
    f_ref, _ = observed_vector(m_ref, patterns)
    phi_ref = target_value_phi(m_ref, target)
    roots = collect_roots_early(inst, theta_ref, patterns,
                                n_starts=root_starts, max_roots=max_roots,
                                seed=int(rng.integers(0, 2**31)))
    phis = [target_value_phi(unpack(inst, r), target) for r in roots]

    confirmed = None
    spread_model = 0.0
    if len(roots) >= 2:
        hi = int(np.argmax(phis))
        lo = int(np.argmin(phis))
        f_hi, _ = observed_vector(unpack(inst, roots[hi]), patterns)
        f_lo, _ = observed_vector(unpack(inst, roots[lo]), patterns)
        pair_dist = float(np.max(np.abs(f_hi - f_lo)))
        spread_model = abs(phis[hi] - phis[lo])
        if pair_dist < dist_tol and spread_model > phi_tol:
            confirmed = {"route": "A2_root_pair",
                         "theta_pair": (roots[lo], roots[hi]),
                         "phi_values": (phis[lo], phis[hi])}

    walks_used = 0
    if confirmed is None:
        order = np.argsort([-abs(p - phi_ref) for p in phis])
        for rk in [int(k) for k in order[:walk_follows]]:
            w = manifold_walk(inst, roots[rk], target,
                              n_seeds=walk_seeds,
                              steps=int(cfg.get("a2_walk_steps", 60)),
                              seed=int(rng.integers(0, 2**31)))
            walks_used += 1
            if w["success"] and w["theta_pair"] is not None:
                x2 = w["theta_pair"][1]
                f2, _ = observed_vector(unpack(inst, x2), patterns)
                d2 = float(np.max(np.abs(f2 - f_ref)))
                dp2 = abs(target_value_phi(unpack(inst, x2), target) - phi_ref)
                if d2 < dist_tol and dp2 > phi_tol:
                    confirmed = {"route": "A2_manifold_follow",
                                 "theta_pair": (theta_ref.copy(), x2),
                                 "phi_values": (phi_ref,
                                                target_value_phi(unpack(inst, x2), target))}
                    break
                if dp2 > spread_model:
                    spread_model = dp2

    jt = m_ref.joint_table()
    q = m_ref.observed_laws(jt)
    pp = {}
    for (v, r), p in jt.items():
        pp[r] = pp.get(r, 0.0) + p
    verts = _lp_vertex_harvest(
        inst.n_vars, q, pp, target,
        n_obj=int(cfg.get("a2_lp_vertices", 24)),
        seed=int(rng.integers(0, 2**31)))

    return {
        "attacker": "A2_completion_enumeration",
        "confirmed_false_recoverable": confirmed is not None,
        "witness": confirmed,
        "n_roots": len(roots),
        "model_spread": float(spread_model),
        "lp_vertices": verts,
        "budget_starts": root_starts,
        "walks_run": walks_used,
        "wall_s": time.perf_counter() - t0,
    }


def frechet_cell_scan(inst, q: dict, pp: dict, target,
                      cfg: dict) -> dict:
    """A3: classical Frechet-cell certification through the target variable.

    For every admissible pair/triple of realized patterns adjacent through
    the target variable (at least one observer and one non-observer of it;
    union of observed sets bounded), solve share-pinned LPs over the union
    table: feasibility of the strata system and min/max of P(V_j=1).
    """
    from scipy.optimize import linprog

    t0 = time.perf_counter()
    j = target[1]
    n_vars = inst.n_vars
    max_union = int(cfg.get("a3_max_union_vars", 4))
    max_cells = int(cfg.get("a3_max_cells", 120))
    pin_tol = float(cfg.get("a3_pin_tol", 1e-9))
    pats = sorted(q.keys())

    cells = []
    for size in (2, 3):
        for combo in itertools.combinations(pats, size):
            has_obs = any(r[j] == 1 for r in combo)
            has_miss = any(r[j] == 0 for r in combo)
            if not (has_obs and has_miss):
                continue
            U = sorted({i for r in combo for i in range(n_vars) if r[i] == 1})
            if len(U) > max_union:
                continue
            cells.append(combo)
    n_candidate = len(cells)
    if n_candidate > max_cells:
        rng = np.random.default_rng(int(cfg.get("a3_cell_seed", 20250901)))
        picks = rng.choice(n_candidate, size=max_cells, replace=False)
        cells = [cells[int(k)] for k in sorted(picks)]

    n_feasible = n_infeasible = 0
    max_width = 0.0
    witness_cell = None
    for combo in cells:
        U = sorted({i for r in combo for i in range(n_vars) if r[i] == 1})
        ucells = list(itertools.product((0, 1), repeat=len(U)))
        rows = [np.ones(len(ucells))]
        rhs = [sum(float(pp.get(r, 0.0)) for r in combo)]
        ok = True
        for r in combo:
            Oidx = [U.index(i) for i in range(n_vars) if r[i] == 1]
            for o in itertools.product((0, 1), repeat=len(Oidx)):
                row = np.zeros(len(ucells))
                for k, uc in enumerate(ucells):
                    if tuple(uc[a] for a in Oidx) == tuple(o):
                        row[k] = 1.0
                rows.append(row)
                rhs.append(float(pp.get(r, 0.0)) * float(q[r].get(tuple(o), 0.0)))
        A = np.array(rows)
        b = np.array(rhs)
        c = np.array([float(uc[U.index(j)]) for uc in ucells])
        vals = []
        for d in (1.0, -1.0):
            res = linprog(d * c, A_eq=A, b_eq=b,
                          bounds=[(0.0, 1.0)] * len(ucells), method="highs")
            if res.status != 0:
                ok = False
                break
            vals.append(float(res.fun * d))
        if not ok:
            n_infeasible += 1
            if witness_cell is None:
                witness_cell = {"patterns": [list(r) for r in combo],
                                "type": "infeasible_strata_system"}
        else:
            n_feasible += 1
            width = vals[1] - vals[0]
            max_width = max(max_width, width)
    fb = frechet_bounds(n_vars, q, pp, target)
    return {
        "attacker": "A3_frechet_cells",
        "n_cells_tested": len(cells),
        "n_cells_candidate": n_candidate,
        "n_feasible": n_feasible,
        "n_infeasible": n_infeasible,
        "max_feasible_width": float(max_width),
        "classical_witness_cell": witness_cell,
        "global_frechet": fb,
        "classically_certified_unique": bool(
            fb["width"] is not None and fb["width"] <= pin_tol),
        "wall_s": time.perf_counter() - t0,
    }


def attack_row(row: dict, cfg: dict | None = None) -> dict:
    """Full adversarial audit of one undecided x RECOVERABLE assertion.

    `row` is a frozen Phase-2 record (structure + seed + target). Rebuilds
    the instance, runs A1 -> A2 -> A3, and returns a verdict record with
    per-attacker budgets and wall times (WP2.5.6 pricing inputs).
    """
    cfg = cfg or {}
    inst, q, pp = instance_from_row(row)
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
    total_wall = sum(a["wall_s"] for a in (rec["A1"], rec["A2"], rec["A3"])
                     if isinstance(a, dict) and "wall_s" in a)
    rec["total_wall_s"] = total_wall
    return rec


def _stable_seed(row: dict, tag: str) -> int:
    import zlib
    payload = (row["instance_id"] + "|" + json.dumps(row["target"]) + "|" + tag)
    return zlib.crc32(payload.encode()) % 2**31


def _serialize_witness(result: dict):
    wit = result.get("witness")
    if not wit or not wit.get("theta_pair"):
        return None
    return {"route": wit["route"], "phi_values": list(wit["phi_values"]),
            "theta_a": [float(x) for x in wit["theta_pair"][0]],
            "theta_b": [float(x) for x in wit["theta_pair"][1]]}
