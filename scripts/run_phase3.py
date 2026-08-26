"""Phase 3 aggregator / analyzer / gate writer (WP3.0a/b/c -> G2.6).

Stages (run locally after collecting the Colab fleet outputs):

  collect         scan notebooks_colab/phase3/incoming (or results/phase3/shards)
                  for every expected output, validate JSON/JSONL, de-duplicate
                  by instance_id|target, checksums, and copy into
                  results/phase3/; writes COLLECT_REPORT_phase3.json.
  scaling         merge the scaling-probe shards (engine rows + attacks +
                  per-shard summaries) -> results/phase3/scaling_probe.jsonl,
                  scaling_readings.json (decidability by n/tag, cert-pipeline
                  vs attack cost ratios, GO arms for WP3.0b).
  signal          definitive local recomputation of WP3.0c (features,
                  stratified permutation AUCs, random-m-graph match null,
                  downstream spread-vs-naive-pooling correlation) ->
                  results/phase3/signal_validity.{json,csv}.
  gate            assemble results/phase3/gate_G26_memo.md from whatever
                  readings exist, with PENDING markers for missing inputs and
                  the pre-registered >=2/3 rule applied when complete.

Usage:
  python3 scripts/run_phase3.py --stage collect
  python3 scripts/run_phase3.py --stage scaling
  python3 scripts/run_phase3.py --stage signal
  python3 scripts/run_phase3.py --stage gate
  python3 scripts/run_phase3.py --stage all

Every gate comparison uses the frozen thresholds in configs/phase3/*.json.
No retro-fitted thresholds are tolerated: the gate writer refuses to run
without them.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results" / "phase3"
INCOMING = ROOT / "notebooks_colab" / "phase3" / "incoming"
SHARDS = RESULTS / "shards"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------

def stage_collect():
    RESULTS.mkdir(parents=True, exist_ok=True)
    INCOMING.mkdir(parents=True, exist_ok=True)
    SHARDS.mkdir(parents=True, exist_ok=True)

    expected = [
        ("prevalence_scan.json", ["prevalence_scan.json"]),
        ("prevalence_scan.csv", ["prevalence_scan.csv"]),
        ("scaling_probe", [f"scaling_probe_shard{i:02d}.jsonl" for i in range(12)]),
        ("scaling_attacks", [f"scaling_attacks_shard{i:02d}.jsonl" for i in range(12)]),
        ("scaling_summary", [f"scaling_summary_shard{i:02d}.json" for i in range(12)]),
        ("cycattack_verdicts", [f"cycattack_verdicts_shard{i:02d}.jsonl" for i in range(4)]),
        ("cycattack_summary", [f"cycattack_summary_shard{i:02d}.json" for i in range(4)]),
        ("signal_validity.json", ["signal_validity.json"]),
        ("signal_validity.csv", ["signal_validity.csv"]),
    ]
    search_roots = [INCOMING, RESULTS, SHARDS]
    report = {"collected_at": time.strftime("%Y-%m-%d %H:%M %Z"),
              "search_roots": [str(p) for p in search_roots],
              "files": {}}
    for label, names in expected:
        hits = []
        for root in search_roots:
            for nm in names:
                for cand in root.glob(nm):
                    hits.append(cand)
        # also accept glob hits directly in incoming (downloaded flat)
        for root in search_roots:
            for pat in names:
                if "*" in pat:
                    continue
                for cand in root.glob(pat):
                    if cand not in hits:
                        hits.append(cand)
        report["files"][label] = []
        for cand in sorted(set(hits)):
            try:
                sz = cand.stat().st_size
                if cand.suffix == ".jsonl":
                    n = sum(1 for _ in open(cand) if _.strip())
                elif cand.suffix == ".json":
                    _ = json.loads(cand.read_text())
                    n = 1
                elif cand.suffix == ".csv":
                    n = sum(1 for _ in open(cand)) - 1
                else:
                    n = None
                dest = RESULTS / cand.name
                if cand.resolve() != dest.resolve():
                    shutil.copy2(cand, dest)
                report["files"][label].append(
                    {"name": cand.name, "bytes": sz, "rows": n,
                     "source": str(cand.parent)})
            except Exception as e:
                report["files"][label].append(
                    {"name": cand.name, "error": f"{type(e).__name__}: {e}"})

    (RESULTS / "COLLECT_REPORT_phase3.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    missing = [k for k, v in report["files"].items() if not v]
    if missing:
        print(f"[collect] MISSING families: {missing} (gate will show PENDING)", flush=True)
    else:
        print("[collect] all expected families present", flush=True)


# --------------------------------------------------------------------------
# scaling
# --------------------------------------------------------------------------

def _med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def stage_scaling():
    probe_files = sorted(RESULTS.glob("scaling_probe_shard*.jsonl"))
    att_files = sorted(RESULTS.glob("scaling_attacks_shard*.jsonl"))
    summ_files = sorted(RESULTS.glob("scaling_summary_shard*.json"))
    if not probe_files:
        print("[scaling] no scaling_probe_shard*.jsonl found -- run collect first", flush=True)
        return

    rows = []
    seen = set()
    for p in probe_files:
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = r["instance_id"] + "|" + json.dumps(r["target"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    merged = RESULTS / "scaling_probe.jsonl"
    with open(merged, "w") as f:
        for r in sorted(rows, key=lambda x: x["instance_id"]):
            f.write(json.dumps(r) + "\n")

    attacks = []
    for p in att_files:
        if not p.exists():
            continue
        for line in open(p):
            try:
                a = json.loads(line)
            except Exception:
                continue
            if a.get("verdict") not in ("SKIPPED_QUOTA", None):
                attacks.append(a)
    a_merged = RESULTS / "scaling_attacks.jsonl"
    with open(a_merged, "w") as f:
        for a in sorted(attacks, key=lambda x: x.get("instance_id", "")):
            f.write(json.dumps(a) + "\n")

    def cert_median(sub):
        walls = [r.get("wall_struct_s", 0) + r.get("wall_formula_s", 0)
                 + r.get("wall_lp_s", 0) + r.get("wall_engine_r1_s", 0)
                 + r.get("wall_engine_r2_s", 0) + r.get("wall_fiber_s", 0)
                 for r in sub]
        return _med(walls)

    def attack_median(sub, key="total_wall_s"):
        return _med([a.get(key) for a in attacks
                     if a.get("n_vars") == sub[0]["n_vars"]] if sub else [])

    by_tag = {}
    for tag in ("n4t", "n5", "n6"):
        sub = [r for r in rows if r.get("tag") == tag]
        if not sub:
            continue
        dec = [r for r in sub if not r["gt_recoverable"].startswith("UNDETERMINED")]
        und = [r for r in sub if r["gt_recoverable"].startswith("UNDETERMINED")]
        by_tag[tag] = {
            "rows": len(sub), "decidable": len(dec),
            "decidability_rate": len(dec) / max(len(sub), 1),
            "median_cert_pipeline_s": cert_median(sub),
            "undecided_x_sheaf_REC": sum(
                1 for r in und if r["sheaf_recoverable"] == "RECOVERABLE"),
        }

    # historical baselines from the frozen merge
    merge = ROOT / "data" / "frozen" / "instances_merged.jsonl"
    hist = {}
    if merge.exists():
        mrows = [json.loads(l) for l in open(merge)]
        for n in (2, 3, 4):
            sub = [r for r in mrows if r["n_vars"] == n]
            dec = [r for r in sub if not r["gt_recoverable"].startswith("UNDETERMINED")]
            hist[f"n{n}"] = {
                "rows": len(sub),
                "decidability_rate": len(dec) / max(len(sub), 1),
                "median_wall_s": _med([r.get("wall_s") for r in sub if r.get("wall_s")]),
                "median_wall_undecided_s": _med([
                    r.get("wall_s") for r in sub
                    if r["gt_recoverable"].startswith("UNDETERMINED")
                    and r.get("wall_s")]),
            }
    # attack medians by n from the already-collected audit verdicts
    # (legacy pricing table) plus the new n=5 attacks
    audit = ROOT / "results" / "phase25" / "audit_verdicts.jsonl"
    if audit.exists():
        arows = [json.loads(l) for l in open(audit) if l.strip()]
        for n in (2, 3, 4):
            hist[f"n{n}"]["audit_median_attack_s"] = _med(
                [r.get("total_wall_s") for r in arows if r.get("n_vars") == n])
            if hist[f"n{n}"].get("audit_median_attack_s") and hist[f"n{n}"].get("median_wall_undecided_s"):
                hist[f"n{n}"]["attack_over_cert_ratio"] = (
                    hist[f"n{n}"]["audit_median_attack_s"] /
                    hist[f"n{n}"]["median_wall_undecided_s"])
    for tag in ("n5", "n4t", "n6"):
        if tag in by_tag and attacks:
            # per-n alias
            mm = {"n5": 5, "n4t": 4, "n6": 6}[tag]
            acm = _med([a.get("total_wall_s") for a in attacks
                        if a.get("n_vars") == mm])
            by_tag[tag]["median_attack_s"] = acm
            if acm is not None and by_tag[tag].get("median_cert_pipeline_s"):
                by_tag[tag]["attack_over_cert_ratio"] = (
                    acm / by_tag[tag]["median_cert_pipeline_s"])

    cfg = json.loads((ROOT / "configs" / "phase3" / "scaling.json").read_text())
    go = cfg["pre_registered_readings"]
    n5_dec = by_tag.get("n5", {}).get("decidability_rate")
    go_feas = (n5_dec is not None and n5_dec >= float(go["GO_feasibility_decidability_min_n5"]))
    r_n5 = by_tag.get("n5", {}).get("attack_over_cert_ratio")
    # trend comparison uses the legacy n=4 ratio if available, else n=3
    r_prev = hist.get("n4", {}).get("attack_over_cert_ratio") \
        or hist.get("n3", {}).get("attack_over_cert_ratio")
    go_econ = (r_n5 is not None and r_n5 >= float(go["GO_economics_ratio_min"])
               and r_prev is not None and r_n5 > r_prev)
    shard_summaries = []
    for p in summ_files:
        try:
            shard_summaries.append(json.loads(p.read_text()))
        except Exception:
            pass
    n5_planned = int(cfg["design"]["n5_structures_total"])
    n5_achieved = sum(s.get("by_tag", {}).get("n5", {}).get("rows", 0)
                      for s in shard_summaries) or by_tag.get("n5", {}).get("rows", 0)
    readings = {
        "config": "configs/phase3/scaling.json (frozen 2026-08-26; revised)",
        "by_tag": by_tag,
        "historical": hist,
        "n5_planned": n5_planned,
        "n5_achieved_target_rows": n5_achieved,
        "coverage_note": ("deadline-guarded; un-run job ids are logged in "
                          "each shard's scaling_summary and retained in COLLECT_REPORT"),
        "GO_feasibility_decidability_min_n5": go["GO_feasibility_decidability_min_n5"],
        "n5_decidability_rate": n5_dec,
        "arm_feasibility_GO": go_feas,
        "GO_economics_ratio_min": go["GO_economics_ratio_min"],
        "r_n5": r_n5, "r_prev": r_prev,
        "arm_economics_GO": go_econ,
        "WP3_0b_overall_GO_arms": int(go_feas) + int(go_econ),
        "WP3_0b_verdict_GO_exists": bool(go_feas or go_econ),
    }
    (RESULTS / "scaling_readings.json").write_text(json.dumps(readings, indent=1))
    print(json.dumps(readings, indent=1))


# --------------------------------------------------------------------------
# signal (definitive local recomputation, piggybacks on WP3.0b outputs when present)
# --------------------------------------------------------------------------

def stage_signal():
    from sheafpatternfusion.battery import frechet_bounds, fraction_observed, overlap_density
    from sheafpatternfusion.enumerate_structures import instantiate, pick_targets
    from sheafpatternfusion.engine2 import sheaf_fiber_verdict
    from sheafpatternfusion.lp_ground_truth import pack
    from sheafpatternfusion.phase3_probe import (
        decompress_payload, instance_from_row_fixed, permutation_auc_p,
        permutation_corr_p, rank_auc, spread_naive_table)

    merge = ROOT / "data" / "frozen" / "instances_merged.jsonl"
    cyc_path = ROOT / "results" / "phase25" / "cyclic_instances.jsonl"
    audit_path = ROOT / "results" / "phase25" / "audit_verdicts.jsonl"
    cfg_sig = json.loads((ROOT / "configs" / "phase3" / "signal.json").read_text())

    if not merge.exists():
        print("[signal] missing data/frozen/instances_merged.jsonl", flush=True)
        return
    mrows = [json.loads(l) for l in open(merge)]
    crows = [json.loads(l) for l in open(cyc_path)] if cyc_path.exists() else []
    if not crows and (RESULTS / "cycattack_verdicts.jsonl").exists():
        # fallback: compact payload was embedded, not materialized
        pass

    # local recompute population: merge + cyclic rows (up to ~4500)
    need = list(mrows) + crows
    need = [dict(r, fixed_cpt=r.get("fixed_cpt")) for r in need]

    def feature_row(row):
        inst, q, pp = instance_from_row_fixed(row)
        fb = frechet_bounds(inst.n_vars, q, pp, tuple(row["target"]))
        return {
            "key": row["instance_id"] + "|" + json.dumps(row["target"]),
            "frechet_width": fb["width"],
            "frac_observed": round(fraction_observed(pp, inst.n_vars), 6),
            "overlap_density": round(overlap_density(
                [tuple(p) for p in row["patterns"]]), 6),
        }

    # 2-worker pool (fork) with checkpoint, same helper as the notebooks
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    feat_path = RESULTS / "_signal_feature_cache.jsonl"
    done = set()
    if feat_path.exists():
        for line in open(feat_path):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    pending = [r for r in need
               if r["instance_id"] + "|" + json.dumps(r["target"]) not in done]
    print(f"[signal] feature cache: {len(done)} on file, {len(pending)} to compute",
          flush=True)
    if pending:
        def _f(r):
            return feature_row(r)
        ctx = mp.get_context("spawn")
        t0 = time.time()
        fout = open(feat_path, "a")
        ex = ProcessPoolExecutor(max_workers=2, mp_context=ctx)
        futs = {ex.submit(_f, r): r for r in pending}
        pend = set(futs)
        done_n = 0
        while pend:
            ds, pend = wait(pend, timeout=3600, return_when=FIRST_COMPLETED)
            if not ds:
                raise RuntimeError("feature pool stalled")
            for f in ds:
                fout.write(json.dumps(f.result()) + "\n")
                done_n += 1
                if done_n % 400 == 0:
                    print(f"  features {done_n}/{len(pending)}", flush=True)
        fout.close()
        ex.shutdown(wait=False)
        print(f"[signal] features computed in {time.time() - t0:.0f}s", flush=True)

    FEATS = {}
    for line in open(feat_path):
        try:
            r = json.loads(line)
            FEATS[r["key"]] = r
        except Exception:
            pass
    MERGE_KEY = {r["instance_id"] + "|" + json.dumps(r["target"]): r for r in mrows}
    CYC_KEY = {r["instance_id"] + "|" + json.dumps(r["target"]): r for r in crows}

    def features_for(row):
        k = row["instance_id"] + "|" + json.dumps(row["target"])
        f = FEATS.get(k, {})
        return {
            "frechet_width": f.get("frechet_width", row.get("frechet_width")),
            "frac_observed": f.get("frac_observed", row.get("frac_observed")),
            "overlap_density": f.get("overlap_density", row.get("overlap_density")),
            "jacobian_rank_deficiency": (
                row.get("n_free_params") - row.get("jacobian_rank")
                if row.get("n_free_params") is not None and row.get("jacobian_rank") is not None
                else row.get("jacobian_rank_deficiency")),
            "max_cross_pattern_marginal_gap": row.get("max_cross_pattern_marginal_gap"),
        }

    FEATURE_NAMES = ["frechet_width", "jacobian_rank_deficiency",
                     "max_cross_pattern_marginal_gap", "frac_observed",
                     "overlap_density"]

    # attacker-labeled pools: audit + collected fleet (scaling + cycattack)
    p2 = {}

    def add_p2(iid, tgt, verdict, mech, pos, nv, src):
        k = iid + "|" + json.dumps(tgt)
        if k in p2:
            p2[k]["sources"].append(src)
            return
        p2[k] = {"instance_id": iid, "target": tgt,
                 "label": 1 if verdict == "CONFIRMED_FALSE_RECOVERABLE" else 0,
                 "mechanism_class": mech, "poset_shape": pos, "n_vars": nv,
                 "sources": [src], "verdict": verdict}

    if audit_path.exists():
        for line in open(audit_path):
            if not line.strip():
                continue
            r = json.loads(line)
            add_p2(r["instance_id"], r["target"], r.get("verdict"),
                   r.get("mechanism_class"), r.get("poset_shape"),
                   r.get("n_vars"), "audit25")
    for pth in sorted(RESULTS.glob("cycattack_verdicts*.jsonl")):
        for line in open(pth):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("verdict"):
                continue
            add_p2(r["instance_id"], r["target"], r.get("verdict"),
                   r.get("mechanism_class"), r.get("poset_shape"),
                   r.get("n_vars"), "cycattack30")
    for pth in sorted(RESULTS.glob("scaling_attacks*.jsonl")):
        for line in open(pth):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("verdict") in ("SKIPPED_QUOTA", None):
                continue
            add_p2(r["instance_id"], r["target"], r.get("verdict"),
                   r.get("mechanism_class"), r.get("poset_shape"),
                   r.get("n_vars"), "scaling30")
    for pth in sorted(RESULTS.glob("scaling_probe*.jsonl")):
        for line in open(pth):
            if not line.strip():
                continue
            r = json.loads(line)
            a = r.get("attack")
            if isinstance(a, dict) and a.get("verdict") in (
                    "CONFIRMED_FALSE_RECOVERABLE", "NO_FALSE_RECOVERABLE_FOUND"):
                add_p2(r["instance_id"], r["target"], a["verdict"],
                       r.get("mechanism_class"), r.get("poset_shape"),
                       r.get("n_vars"), "scaling_inline")

    feat_from_scaling = {}
    for pth in sorted(RESULTS.glob("scaling_probe*.jsonl")):
        for line in open(pth):
            if not line.strip():
                continue
            r = json.loads(line)
            feat_from_scaling[r["instance_id"] + "|" + json.dumps(r["target"])] = r

    rows_p2 = []
    for k, e in p2.items():
        src = feat_from_scaling.get(k)
        f = features_for(src) if src else features_for(
            MERGE_KEY.get(k) or CYC_KEY.get(k) or {})
        if not f:
            continue
        rows_p2.append({**e, **f})

    rows_p1 = []
    for r in list(mrows) + crows:
        if r["gt_recoverable"].startswith("UNDETERMINED"):
            continue
        f = features_for(r)
        rows_p1.append({"label": 1 if r["gt_recoverable"] == "UNRECOVERABLE" else 0,
                        "n_vars": r["n_vars"],
                        "mechanism_class": r.get("mechanism_class"), **f})

    print(f"[signal] P2 attacker-labeled undecided: {len(rows_p2)} "
          f"(positives={sum(e['label'] for e in rows_p2)})")
    print(f"[signal] P1 engine-decided context: {len(rows_p1)} "
          f"(positives={sum(e['label'] for e in rows_p1)})")

    def auc_table(rows, tag, B_perm):
        out = {"pool": tag, "n": len(rows),
               "positives": sum(r["label"] for r in rows)}
        if not rows or len({r["label"] for r in rows}) < 2:
            out["note"] = "single-class pool: AUC undefined (recorded honestly)"
            for fn in FEATURE_NAMES:
                out[fn] = {"auc": None, "p_value": None}
            return out
        ys = [r["label"] for r in rows]
        strata = [f"{r['n_vars']}|{r.get('mechanism_class')}" for r in rows]
        for fn in FEATURE_NAMES:
            xs = [r.get(fn) for r in rows]
            ok = [i for i, x in enumerate(xs) if x is not None]
            res = permutation_auc_p([xs[i] for i in ok], [ys[i] for i in ok],
                                    [strata[i] for i in ok], B=B_perm,
                                    seed=cfg_sig["metrics"]["null_baselines"]
                                         ["label_permutation_within_strata"]["seed_base"])
            out[fn] = res
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            X = np.array([[float(r.get(fn)) if r.get(fn) is not None else 0.0
                           for fn in FEATURE_NAMES] for r in rows])
            pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000))
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            aucs = cross_val_score(pipe, X, np.array(ys), cv=cv, scoring="roc_auc")
            out["logistic_combo_cv_auc_mean"] = float(aucs.mean())
            out["logistic_combo_cv_auc_sd"] = float(aucs.std())
        except Exception as e:
            out["logistic_combo_error"] = str(e)
        return out

    perm_cfg = cfg_sig["metrics"]["null_baselines"]["label_permutation_within_strata"]
    res_primary = auc_table(rows_p2, "P2_attacker_labeled_undecided_PRIMARY",
                            int(perm_cfg["B"]))
    res_context = auc_table(rows_p1, "P1_engine_decided_CONTEXT_CIRCULAR",
                            min(int(perm_cfg["B"]), 500))

    # random-m-graph match null
    K = int(cfg_sig["metrics"]["null_baselines"]["random_m_graph_matches"]["K_per_bucket"])
    from sheafpatternfusion.enumerate_structures import instantiate
    from sheafpatternfusion.phase3_probe import sample_structures
    match_stats = {}
    for n in (3, 4, 5):
        jobs = sample_structures(n, min(K, 400),
                                 seed=20460977 + n, prefix=f"match_n{n}")

        def mf(job):
            vp = {int(k): tuple(v) for k, v in job["structure"]["var_parents"].items()}
            rp = tuple(tuple(p) for p in job["structure"]["r_parents"])
            inst = instantiate((vp, rp), seed=job["draw_seed"])
            jt = inst.joint_table()
            pats = inst.realized_patterns(jt=jt)
            q = inst.observed_laws(jt)
            pp = {}
            for (v, r), pr in jt.items():
                pp[r] = pp.get(r, 0.0) + pr
            tgts = __import__("sheafpatternfusion.enumerate_structures",
                              fromlist=["pick_targets"]).pick_targets(inst)
            if not tgts:
                return None
            fb = frechet_bounds(inst.n_vars, q, pp, tuple(tgts[0]))
            return {"frechet_width": fb["width"],
                    "frac_observed": round(fraction_observed(pp, inst.n_vars), 6),
                    "overlap_density": round(overlap_density(
                        [tuple(p) for p in pats]), 6)}
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        feats = []
        with ProcessPoolExecutor(max_workers=2,
                                  mp_context=mp.get_context("spawn")) as ex:
            for f in ex.map(mf, jobs, chunksize=8):
                if f is not None:
                    feats.append(f)
        real = [r for r in rows_p1 + rows_p2 if r["n_vars"] == n]
        ks = {}
        from scipy.stats import ks_2samp
        for fn in ("frechet_width", "frac_observed", "overlap_density"):
            a = [f[fn] for f in feats if f.get(fn) is not None]
            b = [r.get(fn) for r in real if r.get(fn) is not None]
            if len(a) > 20 and len(b) > 20:
                res = ks_2samp(a, b)
                ks[fn] = {"ks_stat": float(res.statistic),
                          "p_value": float(res.pvalue),
                          "n_match": len(a), "n_real": len(b)}
        match_stats[f"n{n}"] = ks

    # downstream
    corr_inputs = []
    for r in list(mrows) + crows:
        if r["gt_recoverable"] == "UNRECOVERABLE":
            rr = dict(r)
            rr["source_tag"] = "cyclic" if r.get("tag") == "cyclic" else "merge"
            corr_inputs.append(rr)
    # family rows with recomputed spreads (reuse the cache path to avoid 120
    # full fiber runs if the signal notebook already did them)
    fam_extra = []
    fam_path = ROOT / "results" / "phase25" / "discordant_family.jsonl"
    if fam_path.exists():
        for line in open(fam_path):
            try:
                fr = json.loads(line)
            except Exception:
                continue
            try:
                vp = {int(k): tuple(v) for k, v in fr["structure"]["var_parents"].items()}
                rp = tuple(tuple(p) for p in fr["structure"]["r_parents"])
                inst = instantiate((vp, rp), seed=fr["draw_seed"])
                theta = pack(inst)
                tgt = tuple(fr["target"])
                fib = sheaf_fiber_verdict(inst, theta, tgt,
                                          n_starts=48, max_roots=12, seed=13)
                fam_extra.append({
                    "instance_id": f"family_{fr['member_id']}",
                    "seed": fr["draw_seed"],
                    "var_parents": {str(k): list(v) for k, v in vp.items()},
                    "r_parents": [list(p) for p in rp],
                    "fixed_cpt": None,
                    "target": fr["target"],
                    "phi_spread_over_fiber": fib["phi_spread_over_fiber"],
                    "source_tag": "family",
                })
            except Exception as e:
                print("[signal] family row skipped:", fr.get("member_id"), e)
    corr_rows = corr_inputs + fam_extra
    table = spread_naive_table(corr_rows)
    corr_out = {}
    sp = [t["spread"] for t in table]
    ne = [t["naive_abs_err"] for t in table]
    for method in cfg_sig["metrics"]["downstream"]["methods"]:
        corr_out[method] = permutation_corr_p(
            sp, ne, B=int(cfg_sig["metrics"]["downstream"]["corr_B"]),
            seed=20460901, method=method)
    fw = [t.get("frechet_width") for t in table]
    corr_out["frechet_vs_naive_err_spearman"] = permutation_corr_p(
        fw, ne, B=int(cfg_sig["metrics"]["downstream"]["corr_B"]),
        seed=20460902, method="spearman")

    g = cfg_sig["pre_registered_gates"]
    best = 0.0
    best_p = 1.0
    if res_primary.get("positives", 0) > 0:
        best = max((v.get("auc") or 0.0 for k, v in res_primary.items()
                    if k in FEATURE_NAMES), default=0.0)
        best_p = min((v.get("p_value") if v.get("p_value") is not None else 1.0
                      for k, v in res_primary.items() if k in FEATURE_NAMES),
                     default=1.0)
        combo = res_primary.get("logistic_combo_cv_auc_mean")
        auc_arm = (best >= float(g["GO_auc_min"]) and best_p < float(g["GO_p_max"])) \
            or (combo is not None and combo >= float(g["GO_auc_min"]))
    else:
        auc_arm = None
    sp_res = corr_out.get("spearman", {})
    rho_arm = None
    if sp_res:
        rho_arm = abs(sp_res.get("rho") or 0.0) >= float(g["GO_rho_abs_min"]) \
            and (sp_res.get("p_two_sided") or 1.0) < float(g["GO_p_max"])

    signal = {
        "mode": "FINAL" if (RESULTS / "scaling_probe.jsonl").exists()
        else "PARTIAL_existing_assets_only",
        "primary_auc": res_primary,
        "context_circular": res_context,
        "random_m_graph_match_null": match_stats,
        "downstream_correlation": corr_out,
        "arm_auc_GO": auc_arm,
        "arm_rho_GO": rho_arm,
        "WP3_0c_verdict": ("GO" if (auc_arm or rho_arm) else
                           ("NO-GO" if res_primary.get("positives", 0) > 0
                            else "INDETERMINATE_no_positive_labels_yet")),
        "gates": g,
    }
    (RESULTS / "signal_validity.json").write_text(json.dumps(signal, indent=1))
    with open(RESULTS / "signal_validity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "feature", "metric", "value"])
        for k, v in res_primary.items():
            if isinstance(v, dict):
                for mk, mv in v.items():
                    w.writerow(["primary_auc", k, mk, mv])
            else:
                w.writerow(["primary_auc", "", k, v])
        for k, v in res_context.items():
            if isinstance(v, dict):
                for mk, mv in v.items():
                    w.writerow(["context_circular", k, mk, mv])
            else:
                w.writerow(["context_circular", "", k, v])
        for method, v in corr_out.items():
            for mk, mv in v.items():
                w.writerow(["downstream_corr", "", f"{method}.{mk}", mv])
    print(json.dumps({k: v for k, v in signal.items()
                      if k in ("mode", "arm_auc_GO", "arm_rho_GO",
                               "WP3_0c_verdict")}, indent=1))


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------

def stage_gate():
    cfg_prev = json.loads((ROOT / "configs" / "phase3" / "prevalence.json").read_text())
    cfg_scale = json.loads((ROOT / "configs" / "phase3" / "scaling.json").read_text())
    cfg_sig = json.loads((ROOT / "configs" / "phase3" / "signal.json").read_text())

    prev_path = RESULTS / "prevalence_scan.json"
    scale_path = RESULTS / "scaling_readings.json"
    sig_path = RESULTS / "signal_validity.json"

    lines = [
        "# Gate G2.6 (WP3.0 pivot) adjudication memo",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "Source configs: `configs/phase3/{{prevalence,scaling,signal}}.json` (frozen 2026-08-26; scaling revised same day after the mandatory pilot).",
        "Rule (pre-registered, Section 7): PROCEED to WP3.1'/WP3.2' iff AT LEAST 2 of "
        "{WP3.0a, WP3.0b(feasibility OR economics arm), WP3.0c} return GO; "
        "ties toward proceeding ONLY when WP3.0c is one of the GOs. "
        "Otherwise TERMINATE and ship the boundary paper per Appendix A.",
        "",
        "## Probe readouts",
        "",
    ]

    gos = {}
    pend = set()

    if prev_path.exists():
        pj = json.loads(prev_path.read_text())
        ver = pj.get("WP3_0a_verdict", "MISSING")
        gos["WP3.0a"] = ver == "GO"
        lines += [
            "### WP3.0a natural-prevalence scan (plan Section 7)",
            f"Verbatim threshold (from {cfg_prev['pre_registered_gates']}):",
            f"GO iff `n_datasets_with_cycles >= {cfg_prev['pre_registered_gates']['GO_datasets_with_cycles_min']} "
            f"OR pooled_cyclic_fraction >= {cfg_prev['pre_registered_gates']['GO_pooled_cyclic_fraction_min']}`.",
            f"Censused datasets: {pj.get('datasets_loaded', '?')} loaded, "
            f"{len(pj.get('datasets_with_cycles', []))} with Berge-cyclic realized structure "
            f"({', '.join(pj.get('datasets_with_cycles', [])[:5])}"
            f"{' ...' if len(pj.get('datasets_with_cycles', [])) > 5 else ''}), "
            f"pooled eligible subsets {pj.get('pooled_eligible', '?')}, "
            f"pooled cyclic {pj.get('pooled_cyclic', '?')} "
            f"=> `pooled_cyclic_fraction = {pj.get('pooled_cyclic_fraction')}`.",
            f"**WP3.0a verdict: {ver}**.",
            "",
        ]
    else:
        pend.add("WP3.0a")
        lines += ["### WP3.0a ... PENDING (run the prevalence notebook)\n"]

    if scale_path.exists():
        sj = json.loads(scale_path.read_text())
        by = sj.get("by_tag", {})
        fea = sj.get("arm_feasibility_GO")
        eco = sj.get("arm_economics_GO")
        lines += [
            "### WP3.0b scaling probe at n=5 (+n=6 descriptive arm)",
            f"Feasibility arm (threshold from {cfg_scale['pre_registered_readings']}): "
            f"`decidability_rate_n5 >= {cfg_scale['pre_registered_readings']['GO_feasibility_decidability_min_n5']}`. "
            f"Observed n5 rate `{sj.get('n5_decidability_rate')}` over roughly "
            f"`{sj.get('n5_achieved_target_rows', '?')}` target rows "
            f"vs planned `{sj.get('n5_planned')}` -> **{'GO' if fea else 'NO-GO'}**.",
            f"Economics arm: `ratio = median(attack wall)/median(cert-pipeline wall) >= "
            f"{cfg_scale['pre_registered_readings']['GO_economics_ratio_min']} AND strictly "
            f"increasing from n=4 to n=5`. Observed `r_n5 = {sj.get('r_n5')}`, "
            f"`r_prev = {sj.get('r_prev')}` -> **{'GO' if eco else 'NO-GO'}** "
            f"(cert wall = struct+formula+lp+r1+r2+fiber; historical ratios "
            f"n2 {cfg_scale['pre_registered_readings']['baseline_ratios_phase25'].get('n2')}x, "
            f"n3 {cfg_scale['pre_registered_readings']['baseline_ratios_phase25'].get('n3')}x).",
            f"Coverage note: {sj.get('coverage_note', '')}.",
            f"**WP3.0b verdict (exists GO): {bool(fea or eco)}** "
            f"[feasibility={fea}, economics={eco}].",
            "",
        ]
        gos["WP3.0b"] = bool(fea or eco)
    else:
        pend.add("WP3.0b")
        lines += ["### WP3.0b ... PENDING (run the scaling fleet)\n"]

    if sig_path.exists():
        gj = json.loads(sig_path.read_text())
        ver = gj.get("WP3_0c_verdict", "MISSING")
        g = cfg_sig["pre_registered_gates"]
        lines += [
            "### WP3.0c signal-validity probe",
            f"Threshold (from {g}): `AUC >= {g['GO_auc_min']} (perm p < {g['GO_p_max']}) "
            f"OR |rho(spread, naive-pooling error)| >= {g['GO_rho_abs_min']} (p < {g['GO_p_max']})`; "
            "fiber spread is NEVER a predictor (circular).",
            f"Mode: `{gj.get('mode')}`. Primary undecided pool: "
            f"`{gj.get('primary_auc', {}).get('n', '?')}` rows, "
            f"`{gj.get('primary_auc', {}).get('positives', '?')}` positives "
            f"(attacker CONFIRMED_FALSE rows; currently zero unless the "
            "fleet found new unrecoverability witnesses).",
            f"Best feature AUC / logistic combo reported in "
            f"`results/phase3/signal_validity.json` (primary_* + downstream_*).",
            f"**WP3.0c verdict: {ver}**.",
            "",
        ]
        gos["WP3.0c"] = ver == "GO"
    else:
        pend.add("WP3.0c")
        lines += ["### WP3.0c ... PENDING (run the signal notebook or this stage after scaling)\n"]

    if pend:
        lines += [f"## Gate summary: PENDING -- still running: {', '.join(sorted(pend))}.", ""]
        lines += ["Interim GO tally (among available probes): "
                  f"`{sum(1 for v in gos.values() if v)}/{len(gos)}` "
                  f"-> would proceed iff WP3.0c is GO when the count is 2/3." if gos else ""]
    else:
        n_go = sum(1 for v in gos.values() if v)
        proceed = n_go >= 2 and (n_go > 2 or gos.get("WP3.0c"))
        lines += [
            "## Gate G2.6 adjudication (hard rule, Sections 7/Appendix A row 2.75)",
            f"GO tally: `{n_go}/3` ({', '.join(k + '=' + ('GO' if v else 'NO-GO') for k, v in gos.items())}).",
            ("**PROCEED to WP3.1'/WP3.2'** (re-aimed fusion benchmark behind the "
             "amended Phase 3 behind a written win-prediction; see Section 7;"
             " no WP3.2' run without it)."
             if proceed else
             "**TERMINATE the program as originally framed.** Ship the boundary "
             "paper from existing assets: degeneracy result (constant-policy "
             "equivalence at 99.95%), audited error bound (<= 0.33%), cyclic "
             "obstruction characterization with classical-blindness witness "
             "(Frechet certifies 0/899), plus these pivot-gate measurements. "
             "No Phase 3 spend. This is the pre-committed anti-zombie exit."),
            "",
        ]

    memo = RESULTS / "gate_G26_memo.md"
    memo.write_text("\n".join(lines))
    print(memo.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["collect", "scaling", "signal", "gate", "all"])
    args = ap.parse_args()
    try:
        todo = ["collect", "scaling", "signal", "gate"] if args.stage == "all" else [args.stage]
        for st in todo:
            {"collect": stage_collect, "scaling": stage_scaling,
             "signal": stage_signal, "gate": stage_gate}[st]()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
