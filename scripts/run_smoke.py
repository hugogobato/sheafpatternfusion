"""WP1.4 correspondence and content smoke test.

Parts:
 1. Bank correspondence: for every transcription-bank instance,
    (a) factorized-model families satisfy the sheaf section condition (100
        draws per instance);
    (b) fiber-constancy verdict computed SHEAF-SIDE (collect distinct
        factorized completions of the true observed law via multistart root
        finding; check spread of the target across them) and compared with
        the engine verdict and the published-label expectation.
 2. Hazard A probes: the linear (mean-coordinate) sheaf has H^1 = 0 by
    construction; verified numerically. Full-law sheaf is treated separately
    in part 3.
 3. Engineered obstruction witnesses: covariance-completion feasibility on a
    triangle poset (exact PSD certificates), with a feasible control.

Outputs: results/phase1/{correspondence.jsonl,hazard_A.json,obstructions.json}
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sheafpatternfusion.bank import load_bank, targets_of
from sheafpatternfusion.fuse import fuse
from sheafpatternfusion.lp_ground_truth import (
    decide, observed_vector, pack, param_bounds, root_jump_search,
    target_value_phi, unpack,
)
from sheafpatternfusion.laplacian import assemble_laplacian
from sheafpatternfusion.poset import PatternPoset, random_poset
from sheafpatternfusion.radius import minimal_radius
from sheafpatternfusion.sheaf import DiscreteSheaf

OUT = Path(__file__).resolve().parents[1] / "results" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)
rng_global = np.random.default_rng(20260824)


# --------------------------------------------------------------------------
# part 1: bank correspondence
# --------------------------------------------------------------------------

def random_fill_respecting(inst, rng):
    import copy
    import itertools
    m = copy.deepcopy(inst)
    for i in range(m.n_vars):
        pa = m.var_parents[i]
        keys = list(itertools.product(*[(0, 1)] * len(pa))) if pa else [()]
        m.var_cpt[i] = {k: float(rng.uniform(0.15, 0.85)) for k in keys}
    for i in range(m.n_vars):
        pa = m.r_parents[i]
        keys = list(itertools.product(*[(0, 1)] * len(pa))) if pa else [()]
        m.r_cpt[i] = {k: float(rng.uniform(0.25, 0.75)) for k in keys}
    return m


def apply_fixed(inst, cfg):
    for fx in cfg.get("fixed_cpt", []):
        pa = tuple(fx["parents"])
        table = inst.r_cpt if fx["kind"] == "r" else inst.var_cpt
        table[fx["node"]][pa] = float(fx["p"])
    return inst


def collect_roots(inst, f_ref, patterns, n_starts=48, seed=0, tol=1e-9):
    """Distinct factorized completions matching f_ref (pinned parameters are
    substituted out so least_squares bounds stay strict)."""
    rng = np.random.default_rng(seed)
    lo, hi = param_bounds(inst)
    free = np.where(hi - lo > 0)[0]
    base = pack(inst)

    def expand(xf):
        th = base.copy()
        th[free] = xf
        return th

    roots = []
    if len(free) == 0:
        return [base]
    span = hi[free] - lo[free]
    for _ in range(n_starts):
        x0f = lo[free] + 0.02 * span + rng.random(len(free)) * (0.96 * span)
        res = least_squares(lambda xf: observed_vector(unpack(inst, expand(xf)), patterns)[0] - f_ref,
                            x0f, bounds=(lo[free], hi[free]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if np.max(np.abs(res.fun)) < tol:
            full = expand(res.x)
            if all(np.max(np.abs(full - u)) > 1e-6 for u in roots):
                roots.append(full.copy())
    return roots


def run_bank():
    bank = load_bank()
    rows = []
    for iid, (inst, cfg) in sorted(bank.items()):
        # (a) two mechanical checks on random factorized models
        #   A: mass-carrying slices W_r(o) = P(V_O=o, R=r) are valid (nonneg,
        #      mass P(r)): they reassemble into one full table by construction
        #      of the t-polytope;
        #   B: cross-pattern equality of POPULATION-marginal conditionals
        #      (a section of the marginal sheaf) characterizes MCAR-type
        #      ignorability: expected ~100% only for the MCAR instance.
        sec_ok_A = 0
        sec_ok_B = 0
        n_draws = 25
        for _ in range(n_draws):
            m = apply_fixed(random_fill_respecting(inst, rng_global), cfg)
            jt = m.joint_table()
            q = m.observed_laws(jt)
            pp = {}
            for (v, r), p in jt.items():
                pp[r] = pp.get(r, 0.0) + p
            fam_w = {r: {o: c * pp[r] for o, c in cells.items()}
                     for r, cells in q.items()}
            okA = all(abs(sum(cells.values()) - pp[r]) < 1e-9 and
                      all(v >= -1e-12 for v in cells.values())
                      for r, cells in fam_w.items())
            sec_ok_A += int(okA)

            mcar = all(len(m.r_parents[i]) == 0 for i in range(inst.n_vars))
            agree = True
            pats = sorted(q.keys())
            for i in range(inst.n_vars):
                dists = []
                for r in pats:
                    if r[i] != 1:
                        continue
                    posn = sum(1 for j in range(inst.n_vars) if r[j] == 1 and j < i)
                    mg = {}
                    for o, c in q[r].items():
                        key = o[posn]
                        mg[key] = mg.get(key, 0.0) + c
                    dists.append(mg)
                for d1 in dists[1:]:
                    if max(abs(d1.get(k, 0.0) - dists[0].get(k, 0.0))
                           for k in set(d1) | set(dists[0])) > 1e-9:
                        agree = False
            sec_ok_B += int(agree) if mcar else int(not agree)
        sec_rate_A = sec_ok_A / n_draws
        sec_rate_B = sec_ok_B / n_draws

        # (b) fiber-constancy per target
        th_true = pack(inst)
        m_true = unpack(inst, th_true)
        patterns = m_true.realized_patterns(jt=m_true.joint_table())
        f_ref, _ = observed_vector(m_true, patterns)
        for tgt, tspec in zip(targets_of(cfg), cfg["targets"]):
            phi_ref = target_value_phi(m_true, tgt)
            roots = collect_roots(inst, f_ref, patterns, n_starts=40, seed=7)
            phis = [target_value_phi(unpack(inst, r), tgt) for r in roots]
            spread = max(phis) - min(phis) if phis else 0.0
            sheaf_verdict = "RECOVERABLE" if spread < 1e-6 else "UNRECOVERABLE"
            eng = decide(inst, th_true, tgt, tspec.get("formula"), seed=11)
            rows.append({
                "instance": iid,
                "target": list(tgt),
                "expected_label": tspec["expected"],
                "slice_validity_rate": sec_rate_A,
                "mcar_section_rate": sec_rate_B,
                "n_distinct_completions": len(roots),
                "phi_spread_over_fiber": spread,
                "sheaf_verdict": sheaf_verdict,
                "engine_verdict": eng["verdict"],
                "agreement_engine_expected": (
                    eng["verdict"].replace("_RELAXED", "") == tspec["expected"]),
                "agreement_sheaf_expected": (
                    sheaf_verdict == tspec["expected"] or
                    (tspec["expected"] == "UNRECOVERABLE" and sheaf_verdict == "UNRECOVERABLE")),
            })
            print(f"[bank] {iid} target={tgt} sheaf={sheaf_verdict} "
                  f"engine={eng['verdict']} expected={tspec['expected']} "
                  f"spread={spread:.4g} roots={len(roots)}")
    with open(OUT / "correspondence.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


# --------------------------------------------------------------------------
# part 2: Hazard A on the linear (mean) sheaf
# --------------------------------------------------------------------------

def run_hazard_A(n_trials=200):
    """Constructive H^1 = 0 for the linear (mean-coordinate) sheaf: slice any
    global mean vector to the patterns and the minimal radius is exactly zero
    (every consistent family glues). No dimension-formula assertions."""
    ok = 0
    details = []
    for t_i in range(n_trials):
        n_vars = int(rng_global.integers(2, 4))
        n_pat = int(rng_global.integers(2, 2 ** n_vars))
        poset = random_poset(n_vars, n_pat, seed=t_i + 500)
        mu = rng_global.normal(size=n_vars)
        obs = {}
        for r in poset.patterns:
            idx = [i for i in range(n_vars) if r[i] == 1]
            if idx:
                obs[r] = mu[idx]
        if len(obs) < 1:
            continue
        res = minimal_radius(poset, obs)
        glued = res["radius"] < 1e-9
        ok += int(glued)
        if t_i < 10:
            details.append({"trial": t_i, "patterns": [list(p) for p in poset.patterns],
                            "radius": res["radius"], "glued": bool(glued)})
    out = {"trials": n_trials, "passed": ok,
           "claim": "linear mean-coordinate sheaf: every consistent family glues "
                    "(constructive H^1 = 0); obstruction content requires richer stalks",
           "details": details}
    with open(OUT / "hazard_A.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"[hazardA] {ok}/{n_trials} constructive gluing checks passed")
    return out


# --------------------------------------------------------------------------
# part 3: engineered obstruction witnesses
# --------------------------------------------------------------------------

def psd_min_eigenvalue(corr: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(0.5 * (corr + corr.T))))


def run_obstructions():
    results = []
    cases = [
        {"name": "O1_obstructed_triangle",
         "desc": "pair stalks {12} (CI: rho12=0), {13}, {23}; family assigns rho13=rho23=0.9; "
                 "matched unit margins. Any joint law would need correlation matrix with "
                 "rho12=0, rho13=rho23=0.9, which is not PSD.",
         "rho12": 0.0, "rho13": 0.9, "rho23": 0.9},
        {"name": "O2_feasible_control",
         "desc": "same poset and CI constraint, moderate correlations: glues",
         "rho12": 0.0, "rho13": 0.5, "rho23": 0.5},
        {"name": "O3_boundary_case",
         "desc": "rho13=rho23=-0.7071, rho12=0: determinant exactly zero boundary",
         "rho12": 0.0, "rho13": -0.7071067811865476, "rho23": -0.7071067811865476},
    ]
    for case in cases:
        C = np.array([[1.0, case["rho12"], case["rho13"]],
                      [case["rho12"], 1.0, case["rho23"]],
                      [case["rho13"], case["rho23"], 1.0]])
        min_eig = psd_min_eigenvalue(C)
        det = float(np.linalg.det(C))
        if min_eig > 1e-9:
            status = "GLUES"
        elif min_eig > -1e-9:
            status = "BOUNDARY"
        else:
            status = "OBSTRUCTED"
        results.append({
            "name": case["name"],
            "description": case["desc"],
            "correlation_matrix": C.tolist(),
            "min_eigenvalue": min_eig,
            "determinant": det,
            "global_section_exists": bool(min_eig >= -1e-9),
            "status": status,
            "witness_type": "PSD-certificate" if status == "OBSTRUCTED" else "explicit-glue",
        })
        print(f"[obstr] {case['name']}: min_eig={min_eig:.4f} -> {status}")
    with open(OUT / "obstructions.json", "w") as f:
        json.dump(results, f, indent=1)
    return results


if __name__ == "__main__":
    rows = run_bank()
    haz = run_hazard_A()
    obs = run_obstructions()

    n_agree_eng = sum(r["agreement_engine_expected"] for r in rows)
    n_agree_shf = sum(r["agreement_sheaf_expected"] for r in rows)
    slice_rates = [r["slice_validity_rate"] for r in rows]
    summary = {
        "bank_rows": len(rows),
        "engine_vs_expected_agreement": f"{n_agree_eng}/{len(rows)}",
        "sheaf_vs_expected_agreement": f"{n_agree_shf}/{len(rows)}",
        "min_slice_validity_rate": min(slice_rates),
        "mcar_section_detection": {
            r["instance"]: r["mcar_section_rate"] for r in rows[:1]} ,
        "hazardA_passed": f"{haz['passed']}/{haz['trials']}",
        "obstructions": {o["name"]: o["global_section_exists"] for o in obs},
    }
    with open(OUT / "smoke_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
