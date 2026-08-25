"""Phase 2 orchestrator (WP2.1/WP2.2/WP2.3): enumeration falsification.

Stages:
  pilot      -- small stratified grid, single process, prints per-stage timing
  enumeration-- full grid; parallel workers; resumable JSONL checkpoint
  content    -- obstruction scans (discrete gluing LPs + Gaussian PSD) -> content.csv
  analyze    -- confusion matrices, habitat table, gate metrics -> CSVs + summary.json

Usage:
  python3 scripts/run_phase2.py --stage pilot
  python3 scripts/run_phase2.py --stage enumeration [--workers 6]
  python3 scripts/run_phase2.py --stage content
  python3 scripts/run_phase2.py --stage analyze
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)
INSTANCES_JSONL = OUT / "instances.jsonl"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# --------------------------------------------------------------------------
# grid construction
# --------------------------------------------------------------------------

def build_grid(pilot: bool = False):
    from sheafpatternfusion.enumerate_structures import (
        all_structures,
        named_structures,
        r_mechanisms,
        var_dags,
    )

    cfg = json.loads((ROOT / "configs" / "phase2" / "grid.json").read_text())
    if pilot:
        cfg = json.loads(json.dumps(cfg))
        cfg["n2"]["draws_per_structure"] = 1
        cfg["n3"]["sampled_structures"] = 10
        cfg["n4"]["sampled_structures"] = 4
        cfg["named_classes"]["draws_per_structure"] = 1
    grid = []  # list of dicts {iid, n_vars, structure, draw_seed, tag}

    def add(iid, structure, draw_seed, tag):
        grid.append({"iid": iid, "n_vars": len(structure[0]),
                     "structure": {"var_parents": {str(k): v for k, v in structure[0].items()},
                                   "r_parents": [list(p) for p in structure[1]]},
                     "draw_seed": draw_seed, "tag": tag})

    # exhaustive n=2
    for si, st in enumerate(all_structures(2)):
        for d in range(cfg["n2"]["draws_per_structure"]):
            add(f"n2_s{si:03d}_d{d}", st, cfg["seeds"]["draw_seed_base"] + 1000 * si + d, "n2")
    # sampled n=3
    rng3 = __import__("numpy").random.default_rng(cfg["seeds"]["structure_seed_base_n3"])
    structs3 = all_structures(3)
    picks = rng3.choice(len(structs3), size=min(cfg["n3"]["sampled_structures"], len(structs3)),
                        replace=False)
    for k in sorted(int(x) for x in picks):
        for d in range(cfg["n3"]["draws_per_structure"]):
            add(f"n3_s{k:05d}_d{d}", structs3[k],
                cfg["seeds"]["draw_seed_base"] + 100000 * k + d, "n3")
    # named classes
    named = named_structures()

    def parse_structure(spec):
        vp = {i: tuple(p) for i, p in spec[0].items()}
        rp = tuple(tuple(p) for p in spec[1])
        return (vp, rp)

    for name, spec in sorted(named.items()):
        st = parse_structure(spec)
        for d in range(cfg["named_classes"]["draws_per_structure"]):
            add(f"named_{name}_d{d}", st,
                cfg["seeds"]["draw_seed_base"] + 7777 * (list(sorted(named)).index(name)) + d,
                "named")
    # sampled n=4
    rng4 = __import__("numpy").random.default_rng(cfg["seeds"]["structure_seed_base_n4"])
    vd4 = var_dags(4)
    rm4 = r_mechanisms(4)
    for k in range(cfg["n4"]["sampled_structures"]):
        vp = vd4[int(rng4.integers(0, len(vd4)))]
        rp = rm4[int(rng4.integers(0, len(rm4)))]
        for d in range(cfg["n4"]["draws_per_structure"]):
            add(f"n4_r{k:04d}_d{d}", (vp, rp),
                cfg["seeds"]["draw_seed_base"] + 31 * k + d + 5_000_000, "n4")
    return grid


# --------------------------------------------------------------------------
# per-instance worker
# --------------------------------------------------------------------------

def run_instance(job: dict) -> dict | None:
    """Engine verdict + sheaf-fiber verdict + structural annotations for one
    instance (all targets). Returns a single flat record per target."""
    import numpy as np

    from sheafpatternfusion.engine2 import decide2, sheaf_fiber_verdict
    from sheafpatternfusion.enumerate_structures import (
        classify,
        conflict_flags,
        discover_slice_cis,
        graham_acyclic,
        pick_targets,
        poset_shape,
    )
    from sheafpatternfusion.enumerate_structures import instantiate
    from sheafpatternfusion.gluing import marginal_problem_lp, slice_marginal
    from sheafpatternfusion.lp_ground_truth import pack, unpack

    t0 = time.perf_counter()
    vp = {int(k): tuple(v) for k, v in job["structure"]["var_parents"].items()}
    structure = (vp, tuple(tuple(p) for p in job["structure"]["r_parents"]))
    inst = instantiate(structure, seed=job["draw_seed"])
    info = classify(inst)
    jt = inst.joint_table()
    patterns = inst.realized_patterns(jt=jt)

    # table-layer sanity: the true model's own W_r family must be completable
    q = inst.observed_laws(jt)
    pp: dict = {}
    for (v, r), p in jt.items():
        pp[r] = pp.get(r, 0.0) + p
    fam_w = {r: {o: c * pp[r] for o, c in cells.items()} for r, cells in q.items()}
    completability = marginal_problem_lp(inst.n_vars, fam_w)["feasible"]

    sets = [frozenset(i for i in range(inst.n_vars) if r[i] == 1) for r in patterns]
    shape = poset_shape(patterns)
    conflicts = conflict_flags(inst)
    cis = discover_slice_cis(inst, n_draws=16)
    theta_true = pack(inst)
    m_true = unpack(inst, theta_true)

    records = []
    for tgt in pick_targets(inst):
        eng = decide2(inst, theta_true, tgt, seed=11)
        if eng["gt_verdict"].startswith("UNDETERMINED"):
            eng = decide2(inst, theta_true, tgt, jump_starts=90, seed=23)
            eng["gt_evidence"] = "round2:" + eng["gt_evidence"]
        fib = sheaf_fiber_verdict(inst, theta_true, tgt, seed=13)

        records.append({
            "instance_id": job["iid"],
            "tag": job["tag"],
            "seed": job["draw_seed"],
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
            "max_cross_pattern_marginal_gap": conflicts["max_cross_pattern_marginal_gap"],
            "n_slice_ci_constraints": int(sum(len(v) for v in cis.values())),
            "slice_cis": {"".join(map(str, r)): [list(c) for c in lst]
                          for r, lst in cis.items() if lst},
            "wall_s": None,
        })
    dt = time.perf_counter() - t0
    for rec in records:
        rec["wall_s"] = round(dt / max(len(records), 1), 3)
    return records


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def expected_target_keys(job: dict) -> set[str]:
    vp = {int(k): tuple(v) for k, v in job["structure"]["var_parents"].items()}
    structure = (vp, tuple(tuple(pp) for pp in job["structure"]["r_parents"]))
    from sheafpatternfusion.enumerate_structures import instantiate, pick_targets
    inst = instantiate(structure, seed=job["draw_seed"])
    return {job["iid"] + "|" + json.dumps(list(t)) for t in pick_targets(inst)}


def stage_enumeration(workers: int, limit: int = 0):
    import multiprocessing as mp

    grid = build_grid(pilot=False)
    if limit:
        grid = grid[:limit]
    done = set()
    if INSTANCES_JSONL.exists():
        with open(INSTANCES_JSONL) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["instance_id"] + "|" + json.dumps(rec["target"]))
                except Exception:
                    pass
    print(f"[enum] grid size {len(grid)} instances; {len(done)} target-rows on file")

    pending = []
    for j in grid:
        try:
            keys = expected_target_keys(j)
        except Exception as e:
            print(f"[enum] cannot pre-instantiate {j['iid']}: {e}; keeping")
            keys = None
        if keys is not None and keys <= done:
            continue
        pending.append(j)
    print(f"[enum] {len(pending)} instances still to run")

    # crash quarantine: jobs present when a pool breaks gain an attempt;
    # after 3 attempts they are skipped permanently and reported
    qfile = OUT / "quarantine.jsonl"
    attempts: dict[str, int] = {}
    if qfile.exists():
        for line in qfile.read_text().splitlines():
            try:
                rec = json.loads(line)
                attempts[rec["iid"]] = max(attempts.get(rec["iid"], 0), rec["attempts"])
            except Exception:
                pass
    skip = {j["iid"] for j in pending if attempts.get(j["iid"], 0) >= 3}
    if skip:
        print(f"[enum] quarantined (>=3 crash attempts): {sorted(skip)}")
        pending = [j for j in pending if j["iid"] not in skip]

    ctx = mp.get_context("spawn")
    t_start = time.perf_counter()
    n_written = 0
    restarts = 0
    wave_size = max(2 * workers, 6)
    idx = 0
    while idx < len(pending):
        wave = pending[idx:idx + wave_size]
        idx += len(wave)
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = {ex.submit(run_instance, j): j for j in wave}
                for fut in as_completed_safe(futs):
                    j = futs[fut]
                    try:
                        recs = fut.result()
                    except Exception as e:
                        # NEVER re-run jobs inside the main process: a corrupting
                        # instance would kill the supervisor loop's only parent.
                        attempts[j["iid"]] = attempts.get(j["iid"], 0) + 1
                        with open(qfile, "a") as fq:
                            fq.write(json.dumps({"iid": j["iid"],
                                                 "attempts": attempts[j["iid"]]}) + "\n")
                        print(f"[enum] job failed ({type(e).__name__}): "
                              f"{j['iid']} attempt={attempts[j['iid']]}")
                        sys.stdout.flush()
                        continue
                    with open(INSTANCES_JSONL, "a") as fout:
                        for r in recs:
                            fout.write(json.dumps(r) + "\n")
                    n_written += 1
                    if n_written % 25 == 0:
                        el = time.perf_counter() - t_start
                        print(f"[enum] {n_written} instances done "
                              f"({el/60:.1f} min, {el/max(n_written,1):.2f} s/inst)")
                        sys.stdout.flush()
        except (BrokenProcessPool, EOFError, OSError) as e:
            restarts += 1
            with open(qfile, "a") as fq:
                for j2 in wave:
                    fq.write(json.dumps({"iid": j2["iid"],
                                         "attempts": attempts.get(j2["iid"], 0) + 1}) + "\n")
                for j2 in wave:
                    attempts[j2["iid"]] = attempts.get(j2["iid"], 0) + 1
            pending_left = [j2 for j2 in pending[idx - len(wave):]
                            if attempts.get(j2["iid"], 0) < 3]
            idx = len(pending) - len(pending_left)
            if restarts > 60:
                print(f"[enum] giving up after {restarts} pool breaks: {e}")
                raise
            print(f"[enum] pool broke ({type(e).__name__}); restarting wave "
                  f"(restart {restarts}, {len(pending_left)} left)")
            sys.stdout.flush()
            time.sleep(2)
    print(f"[enum] complete: {n_written} new instance records -> {INSTANCES_JSONL}")
    (OUT / "ENUM_COMPLETE").write_text("done\n")


def as_completed_safe(futures):
    from concurrent.futures import wait, FIRST_COMPLETED
    pending = set(futures)
    while pending:
        done_set, pending = wait(pending, return_when=FIRST_COMPLETED)
        for f in done_set:
            yield f


def stage_pilot():
    grid = build_grid(pilot=True)
    print(f"[pilot] {len(grid)} instances")
    tot = {"decide": 0.0, "fiber": 0.0, "other": 0.0}
    import numpy as np

    from sheafpatternfusion.engine2 import decide2, sheaf_fiber_verdict
    from sheafpatternfusion.enumerate_structures import (
        classify, discover_slice_cis, pick_targets, instantiate)
    from sheafpatternfusion.lp_ground_truth import pack

    for k, job in enumerate(grid):
        vp = {int(a): tuple(b) for a, b in job["structure"]["var_parents"].items()}
        structure = (vp, tuple(tuple(p) for p in job["structure"]["r_parents"]))
        t0 = time.perf_counter()
        inst = instantiate(structure, seed=job["draw_seed"])
        info = classify(inst)
        cis = discover_slice_cis(inst, n_draws=16)
        t_struct = time.perf_counter() - t0
        theta = pack(inst)
        for tgt in pick_targets(inst):
            t0 = time.perf_counter()
            eng = decide2(inst, theta, tgt, seed=11)
            t_eng = time.perf_counter() - t0
            t0 = time.perf_counter()
            fib = sheaf_fiber_verdict(inst, theta, tgt, seed=13)
            t_fib = time.perf_counter() - t0
            print(f"[pilot] {job['iid']} {tgt} class={info['mechanism_class']:9s} "
                  f"struct={t_struct:5.2f}s decide={t_eng:6.2f}s ({eng['gt_verdict'][:12]:12s}) "
                  f"fiber={t_fib:5.2f}s ({fib['sheaf_verdict'][:8]:8s}, "
                  f"spread={fib['phi_spread_over_fiber']:.3f})")
            tot["decide"] += t_eng
            tot["fiber"] += t_fib
            tot["other"] += t_struct
    print(f"[pilot] totals over grid: {tot}")


def stage_content():
    import numpy as np
    import csv

    from sheafpatternfusion.enumerate_structures import graham_acyclic
    from sheafpatternfusion.gluing import (
        canonical_cycle_cases,
        psd_completion_min_eig,
        scan_poset_discrete,
    )

    cfg = json.loads((ROOT / "configs" / "phase2" / "grid.json").read_text())
    seed = cfg["seeds"]["poset_scan_seed"]
    rows = []

    shapes = {
        "K4_all_pairs": [tuple(1 if k in (i, j) else 0 for k in range(4))
                         for i, j in itertools.combinations(range(4), 2)],
        "cycle4_pairs": [(1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1), (1, 0, 0, 1)],
        "star3_pairs_acyclic": [(1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1)],
        "triangle_K3": [(1, 1, 0), (1, 0, 1), (0, 1, 1)],
    }
    import zlib
    for name, pats in sorted(shapes.items()):
        assert all(sum(p) == 2 for p in pats), f"{name} must be pure pair-stalks"
        sets = [frozenset(i for i in range(len(p)) if p[i] == 1) for p in pats]
        res = scan_poset_discrete(pats, n_families=cfg["poset_scan_families"] * 3,
                                  seed=seed + zlib.crc32(name.encode()) % 1000)
        rows.append({
            "kind": "discrete_gluing", "name": name,
            "graham_acyclic": graham_acyclic(sets),
            "n_families_tested": res["n_families_tested"],
            "n_globally_feasible": res["n_globally_feasible"],
            "obstruction_fraction": (res["n_obstructed"] /
                                     max(res["n_families_tested"], 1)),
            "witness_json": json.dumps(res["witness"]) if res["witness"] else "",
        })
        print(f"[content] discrete {name}: obstructed "
              f"{res['n_obstructed']}/{res['n_families_tested']}")

    # constraint variant: mutual independence on all triangle pair-stalks
    rng = np.random.default_rng(seed + 55)
    feas = 0
    n_t = 60
    tri = [(1, 1, 0), (1, 0, 1), (0, 1, 1)]
    for _ in range(n_t):
        m = rng.dirichlet(np.ones(2), size=3)
        fam = {}
        for r, (i, j) in zip(tri, [(0, 1), (0, 2), (1, 2)]):
            pi, pj = m[i][1], m[j][1]
            fam[r] = {(1, 1): pi * pj, (1, 0): pi * (1 - pj),
                      (0, 1): (1 - pi) * pj, (0, 0): (1 - pi) * (1 - pj)}
        from sheafpatternfusion.gluing import marginal_problem_lp
        feas += int(marginal_problem_lp(3, fam)["feasible"])
    rows.append({
        "kind": "discrete_gluing_independence_constrained", "name": "triangle_K3",
        "graham_acyclic": False, "n_families_tested": n_t,
        "n_globally_feasible": feas, "obstruction_fraction": 1 - feas / n_t,
        "witness_json": "",
    })
    print(f"[content] independence-constrained triangle: feasible {feas}/{n_t}")

    # Gaussian randomized assignments on the 4-cycle
    rng_g = np.random.default_rng(cfg["seeds"]["content_scan_seed"])
    cyc = [(0, 1), (1, 2), (2, 3), (0, 3)]
    n_obs = 0
    n_rand = 300
    witness_g = None
    for _ in range(n_rand):
        assigned = {(i, j): float(rng_g.uniform(-0.97, 0.97)) for i, j in cyc}
        res = psd_completion_min_eig(4, assigned, n_starts=6, seed=int(rng_g.integers(1e6)))
        if not res["completable"]:
            n_obs += 1
            if witness_g is None:
                witness_g = res
    rows.append({
        "kind": "gaussian_cov_completion", "name": "cycle4_random_assignments",
        "graham_acyclic": False, "n_families_tested": n_rand,
        "n_globally_feasible": n_rand - n_obs,
        "obstruction_fraction": n_obs / n_rand,
        "witness_json": json.dumps(witness_g) if witness_g else "",
    })
    print(f"[content] gaussian cycle4 random: obstructed {n_obs}/{n_rand}")

    for case in canonical_cycle_cases():
        rows.append({
            "kind": "gaussian_canonical", "name": case["name"],
            "graham_acyclic": False,
            "n_families_tested": 1,
            "n_globally_feasible": int(case["completable"]),
            "obstruction_fraction": float(not case["completable"]),
            "witness_json": json.dumps(
                {"assigned": case["assigned"],
                 "min_eig": case["optimal_min_eigenvalue"]}),
        })
    with open(OUT / "content.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[content] wrote {len(rows)} rows -> {OUT/'content.csv'}")


def stage_analyze():
    import csv

    merged = OUT / "instances_merged.jsonl"
    src_file = merged if merged.exists() else INSTANCES_JSONL
    print(f"[analyze] source: {src_file.name}")
    rows = [json.loads(l) for l in open(src_file)]
    print(f"[analyze] {len(rows)} rows")

    decidable = [r for r in rows if not r["gt_recoverable"].startswith("UNDETERMINED")]
    undecided = [r for r in rows if r["gt_recoverable"].startswith("UNDETERMINED")]

    def agree(r):
        gt_rec = r["gt_recoverable"] == "RECOVERABLE"
        sh_rec = r["sheaf_recoverable"] == "RECOVERABLE"
        return gt_rec == sh_rec

    mismatches = [r for r in decidable if not agree(r)]

    def conf(rows_sub):
        tp = sum(1 for r in rows_sub if r["gt_recoverable"] == "RECOVERABLE"
                 and r["sheaf_recoverable"] == "RECOVERABLE")
        tn = sum(1 for r in rows_sub if r["gt_recoverable"] != "RECOVERABLE"
                 and r["sheaf_recoverable"] != "RECOVERABLE")
        fp = sum(1 for r in rows_sub if r["gt_recoverable"] != "RECOVERABLE"
                 and r["sheaf_recoverable"] == "RECOVERABLE")
        fn = sum(1 for r in rows_sub if r["gt_recoverable"] == "RECOVERABLE"
                 and r["sheaf_recoverable"] != "RECOVERABLE")
        return tp, tn, fp, fn

    groups = {
        "ALL": rows,
        "MCAR": [r for r in rows if r["mechanism_class"] == "MCAR"],
        "MAR": [r for r in rows if r["mechanism_class"] == "MAR"],
        "MNAR_self": [r for r in rows if r["mechanism_class"] == "MNAR_self"],
        "MNAR_other": [r for r in rows if r["mechanism_class"] == "MNAR_other"],
        "acyclic_poset": [r for r in rows if r["poset_shape"] in ("chain", "acyclic")],
        "cyclic_poset": [r for r in rows if r["poset_shape"] == "cyclic"],
        "n_vars=2": [r for r in rows if r["n_vars"] == 2],
        "n_vars=3": [r for r in rows if r["n_vars"] == 3],
        "n_vars=4": [r for r in rows if r["n_vars"] == 4],
    }

    with open(OUT / "enumeration.csv", "w", newline="") as f:
        cols = ["instance_id", "tag", "seed", "n_vars", "mechanism_class",
                "has_self_edge", "poset_shape", "graham_acyclic",
                "n_realized_patterns", "target", "true_value",
                "gt_recoverable", "gt_evidence", "lp_width",
                "witness_delta_phi", "sheaf_recoverable",
                "phi_spread_over_fiber", "n_distinct_completions",
                "observed_family_completable", "conflict_mcar_style",
                "agreement"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {c: r.get(c) for c in cols}
            out["agreement"] = (agree(r) if not r["gt_recoverable"].startswith("UNDETERMINED")
                                else "UNDECIDED_GT")
            w.writerow(out)

    summary = {"n_rows": len(rows), "n_decidable_gt": len(decidable),
               "n_undecided_gt": len(undecided),
               "undecided_rate": len(undecided) / max(len(rows), 1)}
    for g, rs in groups.items():
        rd = [r for r in rs if not r["gt_recoverable"].startswith("UNDETERMINED")]
        tp, tn, fp, fn = conf(rd)
        mm = [r for r in rd if not agree(r)]
        summary[g] = {
            "rows": len(rs), "decidable": len(rd),
            "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "agreement_rate": (tp + tn) / max(len(rd), 1),
            "unexplained_mismatch_rate": len(mm) / max(len(rd), 1),
        }
        if mm:
            summary[g]["mismatch_ids"] = [
                {"id": r["instance_id"], "target": r["target"],
                 "gt": r["gt_recoverable"], "sheaf": r["sheaf_recoverable"],
                 "evidence": r["gt_evidence"]} for r in mm[:20]]

    # habitat crossing (Hazard N)
    habitat = {}
    hab_rows = []
    for conflict in (True, False):
        for rec in ("RECOVERABLE", "UNRECOVERABLE"):
            cell = [r for r in decidable
                    if r["conflict_mcar_style"] == conflict
                    and ((r["gt_recoverable"] == "RECOVERABLE")
                         == (rec == "RECOVERABLE"))]
            habitat[f"conflict={int(conflict)}|{rec}"] = len(cell)

    # engine-undecided rows: what does the sheaf side say, and are the
    # fingerprints locally overidentified?
    und = [r for r in rows if r["gt_recoverable"].startswith("UNDETERMINED")]
    summary["engine_undecided_x_sheaf"] = dict(
        __import__("collections").Counter(r["sheaf_recoverable"] for r in und))
    summary["engine_undecided_jacobian_full_rank"] = float(
        sum(1 for r in und if r.get("jacobian_full_rank")) / max(len(und), 1))
    summary["decidable_jacobian_full_rank_rate"] = float(
        sum(1 for r in decidable if r.get("jacobian_full_rank"))
        / max(len(decidable), 1))
    for r in decidable:
        hab_rows.append({
            "instance_id": r["instance_id"], "mechanism_class": r["mechanism_class"],
            "conflict": int(r["conflict_mcar_style"]),
            "recoverable": int(r["gt_recoverable"] == "RECOVERABLE"),
            "habitat_cell": int(r["conflict_mcar_style"] and
                                r["gt_recoverable"] == "RECOVERABLE"),
        })
    with open(OUT / "habitat.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance_id", "mechanism_class",
                                          "conflict", "recoverable", "habitat_cell"])
        w.writeheader()
        w.writerows(hab_rows)
    n_habitat = sum(h["habitat_cell"] for h in hab_rows)
    summary["habitat_nonempty_cells"] = n_habitat
    summary["habitat_crossing"] = habitat
    summary["gate_metrics"] = {
        "G3a_agreement_overall": summary["ALL"]["agreement_rate"],
        "G3a_unexplained_overall": summary["ALL"]["unexplained_mismatch_rate"],
        "threshold_agreement_min": 0.98,
        "threshold_unexplained_max": 0.02,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("n_rows", "n_decidable_gt", "n_undecided_gt",
                               "undecided_rate", "ALL", "habitat_nonempty_cells",
                               "gate_metrics")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["pilot", "enumeration", "content", "analyze"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="truncate the enumeration grid (smoke tests only)")
    args = ap.parse_args()
    try:
        if args.stage == "pilot":
            stage_pilot()
        elif args.stage == "enumeration":
            stage_enumeration(args.workers, limit=args.limit)
        elif args.stage == "content":
            stage_content()
        else:
            stage_analyze()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    # known glibc teardown fault in this scipy/HiGHS combination aborts
    # AFTER all work completes; skip interpreter shutdown entirely
    os._exit(0)
