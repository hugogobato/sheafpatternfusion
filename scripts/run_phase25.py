"""Phase 2.5 orchestrator (WP2.5.1-WP2.5.4): independent validation battery.

Stages:
  battery -- null policies N0-N4 on engine-undecided rows; S* priority sample
  audit   -- adversarial A1/A2/A3 attacks on undecided x RECOVERABLE assertions
  family  -- discordant-family construction around the n3_s03759_d0 seed
  cyclic  -- forced cyclic-poset stratum: generate + full Phase-2 pipeline
  pilot   -- tiny versions of all four with wall/RSS accounting (Section 11)

Usage:
  python3 scripts/run_phase25.py --stage battery [--limit N]
  python3 scripts/run_phase25.py --stage audit [--shard k --shards 6] [--limit N] [--profile pilot]
  python3 scripts/run_phase25.py --stage family [--limit N]
  python3 scripts/run_phase25.py --stage cyclic [--shard k --shards 4] [--limit N]
  python3 scripts/run_phase25.py --stage pilot
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "phase25"
OUT.mkdir(parents=True, exist_ok=True)
FROZEN = ROOT / "data" / "frozen" / "instances_merged.jsonl"
FALLBACK = ROOT / "results" / "phase2" / "instances_merged.jsonl"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_frozen_rows(limit: int = 0) -> list[dict]:
    src = FROZEN if FROZEN.exists() else FALLBACK
    if not src.exists():
        raise SystemExit("no frozen merge found (expected data/frozen/ or results/phase2/)")
    rows = []
    with open(src) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if limit:
        rows = rows[:limit]
    return rows


def load_cfg(name: str, profile: str | None = None) -> dict:
    cfg = json.loads((ROOT / "configs" / "phase25" / f"{name}.json").read_text())
    if profile and profile in cfg:
        cfg = {**cfg, **cfg[profile]}
    return cfg


def row_key(r: dict) -> str:
    return r["instance_id"] + "|" + json.dumps(r["target"])


# --------------------------------------------------------------------------
# WP2.5.1 battery
# --------------------------------------------------------------------------

def stage_battery(limit: int = 0):
    from sheafpatternfusion.battery import evaluate_nulls

    cfg = load_cfg("battery")
    rows = [r for r in load_frozen_rows()
            if r["gt_recoverable"].startswith("UNDETERMINED")]
    print(f"[battery] {len(rows)} undecided rows")
    if limit:
        rows = rows[:limit]
    t0 = time.perf_counter()
    out = evaluate_nulls(rows, cfg)
    dt = time.perf_counter() - t0

    with open(OUT / "null_battery.json", "w") as f:
        json.dump({"metrics": out["metrics"], "wall_s": dt}, f, indent=1)
    import csv
    cols = list(out["scored"][0].keys())
    with open(OUT / "null_battery.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out["scored"])
    with open(OUT / "priority_sample.jsonl", "w") as f:
        for r in out["priority_sample"]:
            f.write(json.dumps(r) + "\n")

    m = out["metrics"]
    print(f"[battery] done in {dt:.1f}s ({dt/max(len(rows),1)*1000:.0f} ms/row), "
          f"peak RSS {rss_mb():.0f} MB")
    print(f"[battery] N0 acc={m['N0_constant_recoverable']['accuracy']:.4f} "
          f"N4 acc={m['N4_constant_unrecoverable']['accuracy']:.4f}")
    for k in ("N1", "N2", "N3"):
        b = m["best_swept"][k]
        print(f"[battery] best {k}: tau={b['tau']} acc={b['accuracy']:.4f}")
    print(f"[battery] S* size={m['S_star_size']} "
          f"(+{len(out['priority_sample']) - m['S_star_size']} discordant refs)")


# --------------------------------------------------------------------------
# WP2.5.2 audit
# --------------------------------------------------------------------------

def build_audit_frame(srs_seed: int, srs_n: int, include_s_star: bool):
    rows = [r for r in load_frozen_rows()
            if r["gt_recoverable"].startswith("UNDETERMINED")
            and r["sheaf_recoverable"] == "RECOVERABLE"]
    frame: dict[str, dict] = {}
    for r in rows:
        key = row_key(r)
        if r["n_vars"] == 2 or r["n_vars"] == 4:
            e = frame.setdefault(key, dict(r, _strata=[]))
            e["_strata"].append(f"n{r['n_vars']}_census")
    n3 = [r for r in rows if r["n_vars"] == 3]
    rng = __import__("numpy").random.default_rng(srs_seed)
    picks = rng.choice(len(n3), size=min(srs_n, len(n3)), replace=False)
    for k in sorted(int(x) for x in picks):
        r = n3[k]
        e = frame.setdefault(row_key(r), dict(r, _strata=[]))
        e["_strata"].append("n3_srs")
    if include_s_star:
        pf = OUT / "priority_sample.jsonl"
        if pf.exists():
            want = {}
            for line in pf.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    want[rec["instance_id"] + "|" + json.dumps(rec["target"])] = rec["reason"]
                except Exception:
                    pass
            for r in rows:
                key = row_key(r)
                if key in want:
                    e = frame.setdefault(key, dict(r, _strata=[]))
                    e["_strata"].append("S_star:" + want[key])
        else:
            print("[audit] WARNING: priority_sample.jsonl missing; run battery first")
    return [frame[k] for k in sorted(frame.keys())]


def stage_audit(shard: int = -1, shards: int = 6, limit: int = 0,
                profile: str | None = None):
    from sheafpatternfusion.attackers import attack_row

    cfg = load_cfg("audit", profile)
    sampling = load_cfg("audit")["sampling"]
    frame = build_audit_frame(int(sampling["srs_seed"]), int(sampling["srs_n"]),
                              bool(sampling["include_S_star"]))
    print(f"[audit] frame: {len(frame)} jobs")
    jobs = frame if shard < 0 else frame[shard::shards]
    if limit:
        jobs = jobs[:limit]
    print(f"[audit] shard={shard}: {len(jobs)} jobs (profile={profile or 'full'})")

    vfile = OUT / ("audit_verdicts.jsonl" if shard < 0
                   else f"audit_verdicts_shard{shard:02d}.jsonl")
    done = set()
    if vfile.exists():
        for line in vfile.read_text().splitlines():
            try:
                rec = json.loads(line)
                done.add(rec["instance_id"] + "|" + json.dumps(rec["target"]))
            except Exception:
                pass
    with open(OUT / "audit_sample.jsonl", "w") as f:
        for j in frame:
            f.write(json.dumps({k: j.get(k) for k in
                                ("instance_id", "target", "_strata",
                                 "mechanism_class", "poset_shape")}) + "\n")

    t0 = time.perf_counter()
    kills = 0
    for k, job in enumerate(jobs):
        if row_key(job) in done:
            continue
        try:
            rec = attack_row(job, cfg)
        except Exception as e:
            print(f"[audit] FAILED {job['instance_id']}: {type(e).__name__}: {e}",
                  flush=True)
            continue
        with open(vfile, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["verdict"] == "CONFIRMED_FALSE_RECOVERABLE":
            kills += 1
            print(f"[audit] *** CONFIRMED FALSE RECOVERABLE: "
                  f"{rec['instance_id']} via {rec['confirming_route']} ***",
                  flush=True)
        if (k + 1) % 10 == 0:
            el = time.perf_counter() - t0
            done_n = sum(1 for _ in range(k + 1))
            eta_h = el / max(done_n, 1) * (len(jobs) - k - 1) / 3600
            print(f"[audit] {k+1}/{len(jobs)} {el/60:.1f} min "
                  f"({el/max(done_n,1):.1f}s/job, ETA {eta_h:.1f}h, "
                  f"kills={kills}, RSS {rss_mb():.0f}MB)", flush=True)
    print(f"[audit] shard done: kills={kills} -> {vfile.name}; peak RSS {rss_mb():.0f} MB")


# --------------------------------------------------------------------------
# WP2.5.3 family
# --------------------------------------------------------------------------

def stage_family(limit: int = 0):
    from sheafpatternfusion.discordant_family import run_family

    cfg = load_cfg("family")
    t0 = time.perf_counter()
    out = run_family(cfg, limit=limit)
    dt = time.perf_counter() - t0
    with open(OUT / "discordant_family.jsonl", "w") as f:
        f.write(json.dumps(out["summary"]) + "\n")
        for r in out["records"]:
            f.write(json.dumps(r) + "\n")
    print(f"[family] {json.dumps(out['summary'])}")
    print(f"[family] done in {dt/60:.1f} min, peak RSS {rss_mb():.0f} MB")


# --------------------------------------------------------------------------
# WP2.5.4 cyclic stratum
# --------------------------------------------------------------------------

def stage_cyclic(shard: int = -1, shards: int = 4, limit: int = 0):
    from sheafpatternfusion.cyclic_synth import (
        make_cyclic_jobs,
        run_cyclic_instance,
    )

    cfg = load_cfg("cyclic_grid")
    if shard >= 0:
        cfg = {**cfg, "shards": shards}
    t0 = time.perf_counter()
    jobs, stats = make_cyclic_jobs(cfg, shard_idx=shard)
    gen_dt = time.perf_counter() - t0
    if limit:
        jobs = jobs[:limit]
    print(f"[cyclic] generation: {len(jobs)} accepted in {gen_dt:.1f}s "
          f"(attempts={stats['attempts']}, rejected_shape={stats['rejected_shape']}, "
          f"duplicates={stats['duplicate']})")
    tag = "" if shard < 0 else f"_shard{shard:02d}"
    cfile = OUT / f"cyclic_instances{tag}.jsonl"
    done = set()
    if cfile.exists():
        for line in cfile.read_text().splitlines():
            try:
                done.add(json.loads(line)["instance_id"])
            except Exception:
                pass
    n_ok = 0
    mismatches = 0
    for k, job in enumerate(jobs):
        if job["iid"] in done:
            n_ok += 1
            continue
        try:
            recs = run_cyclic_instance(job, cfg["budgets"], int(cfg["ci_discovery_draws"]))
        except Exception as e:
            print(f"[cyclic] FAILED {job['iid']}: {type(e).__name__}: {e}", flush=True)
            continue
        agree = all(((r["gt_recoverable"] == "RECOVERABLE")
                     == (r["sheaf_recoverable"] == "RECOVERABLE"))
                    for r in recs)
        mismatches += int(not agree)
        with open(cfile, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        n_ok += 1
        if (k + 1) % 10 == 0:
            el = time.perf_counter() - t0
            print(f"[cyclic] {k+1}/{len(jobs)} {el/60:.1f} min "
                  f"({(time.perf_counter()-t0-gen_dt)/max(n_ok,1):.1f}s/inst, "
                  f"mismatch_inst={mismatches}, RSS {rss_mb():.0f}MB)", flush=True)
    stats_out = {"generation": stats, "generation_wall_s": gen_dt,
                 "jobs": len(jobs), "ran_or_resumed": n_ok,
                 "instance_mismatches": mismatches}
    (OUT / f"cyclic_gen_stats{tag}.json").write_text(json.dumps(stats_out, indent=1))
    print(f"[cyclic] shard done -> {cfile.name}; total {time.perf_counter()-t0:.0f}s; "
          f"peak RSS {rss_mb():.0f} MB")


# --------------------------------------------------------------------------
# Section-11 pilot
# --------------------------------------------------------------------------

def stage_pilot():
    print("=== PILOT battery (60 rows) ===")
    stage_battery(limit=60)
    print("=== PILOT audit (2 rows, pilot profile) ===")
    stage_audit(shard=-1, limit=2, profile="pilot")
    print("=== PILOT family (3 members, reduced budgets) ===")
    cfg = load_cfg("family")
    cfg.update(member_root_starts=48, member_max_roots=8, member_walk_follows=1,
               member_walk_n_seeds=4, member_walk_steps=30)
    t0 = time.perf_counter()
    out = run_family(cfg, limit=3)
    print(f"[pilot-family] {(time.perf_counter()-t0)/3:.1f}s/member avg; "
          f"witnessed={out['summary']['n_witnessed_discordant']}/3")
    print("=== PILOT cyclic (6 instances) ===")
    stage_cyclic(shard=-1, limit=6)
    print(f"[pilot] final peak RSS {rss_mb():.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["battery", "audit", "family", "cyclic", "pilot"])
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--profile", default=None, choices=[None, "pilot", "full"])
    args = ap.parse_args()
    try:
        if args.stage == "battery":
            stage_battery(args.limit)
        elif args.stage == "audit":
            n_sh = args.shards or 6
            stage_audit(args.shard, n_sh, args.limit, args.profile)
        elif args.stage == "family":
            stage_family(args.limit)
        elif args.stage == "cyclic":
            n_sh = args.shards or 4
            stage_cyclic(args.shard, n_sh, args.limit)
        else:
            stage_pilot()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
