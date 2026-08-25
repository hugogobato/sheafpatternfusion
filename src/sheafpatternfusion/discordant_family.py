"""WP2.5.3 discordant-family construction.

Seed: n3_s03759_d0 / target mean(0) (engine UNDETERMINED_RELAXED_FRAGILE,
sheaf UNRECOVERABLE, observed mass-family LP-incompletable, Jacobian rank
20/21). The structure class is frozen; members are fresh mechanism draws on
the same structure. A member is WITNESSED-DISCORDANT iff

  (i)   the engine is UNDETERMINED on the target (discordant character
        preserved), AND
  (ii)  two distinct model-valid completions of the observed fingerprint
        differ on the target by > phi_tol at fingerprint distance < dist_tol
        (independent A2-style samplers, fresh seeds), AND
  (iii) a classical witness exists: the observed MASS family is globally
        LP-incompletable, or an adjacent Frechet cell through the target is
        infeasible.

Success gate: >= cfg.success_threshold witnessed members greenlights the
theory-note path after G2.5b adjudication; collapse to the singleton seed is
recorded as negative and feeds demotion rule D1.
"""
from __future__ import annotations

import time

import numpy as np

from .attackers import frechet_cell_scan
from .engine2 import collect_roots_early, decide2
from .enumerate_structures import classify, instantiate
from .gluing import marginal_problem_lp
from .lp_ground_truth import (
    observed_vector,
    pack,
    target_value_phi,
    unpack,
    manifold_walk,
)


SEED_ROW = {
    "instance_id": "n3_s03759_d0",
    "seed": 396170901,
    "var_parents": {"0": [], "1": [0], "2": [0, 1]},
    "r_parents": [[1], [0, 2], [0, 1, 2]],
    "target": ["mean", 0],
}


def seed_structure() -> tuple:
    vp = {int(k): tuple(v) for k, v in SEED_ROW["var_parents"].items()}
    return vp, tuple(tuple(p) for p in SEED_ROW["r_parents"])


def member_draw_seed(k: int, cfg: dict) -> int:
    base = int(cfg.get("draw_seed_base", 20270901))
    if k == 0:
        return SEED_ROW["seed"]
    return base + 7919 * k


def _model_pair(inst, theta_ref, target, patterns, cfg: dict, rng_seed: int):
    rng = np.random.default_rng(rng_seed)
    from .lp_ground_truth import param_bounds

    lo, hi = param_bounds(inst)
    free = np.where(hi - lo > 0)[0]
    dist_tol = float(cfg.get("dist_tol", 1e-9))
    phi_tol = float(cfg.get("phi_tol", 1e-4))
    roots = collect_roots_early(
        inst, theta_ref, patterns,
        n_starts=int(cfg.get("member_root_starts", 200)),
        max_roots=int(cfg.get("member_max_roots", 16)),
        seed=rng_seed)
    phis = [target_value_phi(unpack(inst, r), target) for r in roots]

    def check_pair(a_th, b_th, route):
        fa, _ = observed_vector(unpack(inst, a_th), patterns)
        fb, _ = observed_vector(unpack(inst, b_th), patterns)
        d = float(np.max(np.abs(fa - fb)))
        dp = abs(target_value_phi(unpack(inst, a_th), target)
                 - target_value_phi(unpack(inst, b_th), target))
        return (d < dist_tol and dp > phi_tol), d, dp, route

    best = {"found": False, "spread": 0.0, "pair": None}
    if len(roots) >= 2:
        hi_i = int(np.argmax(phis))
        lo_i = int(np.argmin(phis))
        ok, d, dp, _ = check_pair(roots[lo_i], roots[hi_i], "A2_root_pair")
        best["spread"] = max(best["spread"], dp)
        if ok:
            best.update(found=True,
                        pair={"theta_a": [float(x) for x in roots[lo_i]],
                              "theta_b": [float(x) for x in roots[hi_i]],
                              "dist": d, "delta_phi": dp,
                              "route": "A2_root_pair"})
            return best
    for rk in range(min(int(cfg.get("member_walk_follows", 4)), len(roots))):
        w = manifold_walk(inst, roots[rk], target,
                          n_seeds=int(cfg.get("member_walk_n_seeds", 10)),
                          steps=int(cfg.get("member_walk_steps", 60)),
                          seed=int(rng.integers(0, 2**31)))
        if w["success"] and w["theta_pair"] is not None:
            x2 = w["theta_pair"][1]
            ok, d, dp, _ = check_pair(theta_ref, x2, "A2_manifold_follow")
            best["spread"] = max(best["spread"], dp)
            if ok:
                best.update(found=True,
                            pair={"theta_a": [float(x) for x in theta_ref],
                                  "theta_b": [float(x) for x in x2],
                                  "dist": d, "delta_phi": dp,
                                  "route": "A2_manifold_follow"})
                break
    return best


def classical_witness(inst, q: dict, pp: dict, target, cfg: dict) -> dict | None:
    jt_w = {r: {o: pp[r] * c for o, c in cells.items()} for r, cells in q.items()}
    fam = marginal_problem_lp(inst.n_vars, jt_w)
    if not fam["feasible"]:
        return {"type": "mass_family_lp_infeasible"}
    scan = frechet_cell_scan(inst, q, pp, target, cfg)
    if scan["n_infeasible"] > 0:
        return {"type": "frechet_cell_infeasible",
                "cell": scan["classical_witness_cell"]}
    return None


def evaluate_member(k: int, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    t0 = time.perf_counter()
    structure = seed_structure()
    draw_seed = member_draw_seed(k, cfg)
    inst = instantiate(structure, seed=draw_seed)

    theta = pack(inst)
    target = ("mean", 0)
    eng = decide2(inst, theta, target,
                  jump_starts=int(cfg.get("engine_jump_starts", 40)), seed=11)
    engine_round2 = False
    if eng["gt_verdict"].startswith("UNDETERMINED"):
        eng = decide2(inst, theta, target,
                      jump_starts=int(cfg.get("engine_jump_starts_r2", 90)),
                      seed=23)
        engine_round2 = True

    m = unpack(inst, theta)
    jt = m.joint_table()
    patterns = m.realized_patterns(jt=jt)
    q = m.observed_laws(jt)
    pp = {}
    for (v, r), p in jt.items():
        pp[r] = pp.get(r, 0.0) + p

    rec = {
        "member_id": k,
        "draw_seed": draw_seed,
        "is_origin_seed": k == 0,
        "structure": {
            "var_parents": {str(i): list(p) for i, p in structure[0].items()},
            "r_parents": [list(p) for p in structure[1]]},
        "target": list(target),
        "engine_verdict": eng["gt_verdict"],
        "engine_evidence": eng["gt_evidence"],
        "engine_round2": engine_round2,
        "mechanism_class": None,
        "n_patterns": len(patterns),
    }
    try:
        rec["mechanism_class"] = classify(inst)["mechanism_class"]
    except Exception:
        pass

    if not eng["gt_verdict"].startswith("UNDETERMINED"):
        rec.update({"witnessed_discordant": False,
                    "fail_reason": "engine_decided",
                    "wall_s": time.perf_counter() - t0})
        return rec

    pair = _model_pair(inst, theta, target, patterns, cfg,
                       int(cfg.get("sampler_seed", 5150)) + k)
    rec["model_pair_found"] = pair["found"]
    rec["model_spread"] = pair["spread"]
    rec["model_pair_route"] = pair["pair"]["route"] if pair["pair"] else None
    wit = None
    if pair["found"]:
        wit = classical_witness(inst, q, pp, target, cfg)
    rec["classical_witness"] = wit
    rec["witnessed_discordant"] = bool(pair["found"] and wit)
    if not rec["witnessed_discordant"]:
        rec["fail_reason"] = ("no_model_pair" if not pair["found"]
                              else "no_classical_witness")
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def run_family(cfg: dict | None = None, limit: int = 0) -> dict:
    cfg = cfg or {}
    n_members = int(cfg.get("n_members", 120))
    if limit:
        n_members = min(n_members, limit)
    records = [evaluate_member(k, cfg) for k in range(n_members)]
    witnessed = [r for r in records if r["witnessed_discordant"]]
    summary = {
        "n_members": len(records),
        "n_witnessed_discordant": len(witnessed),
        "success_threshold": int(cfg.get("success_threshold", 10)),
        "gate": "PASS" if len(witnessed) >= int(cfg.get("success_threshold", 10))
        else "COLLAPSE",
        "origin_seed_witnessed": bool(records and records[0].get("witnessed_discordant")),
    }
    return {"summary": summary, "records": records}
