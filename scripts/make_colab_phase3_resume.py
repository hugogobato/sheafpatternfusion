"""Generate the Phase-3b continuation fleet for the remaining scaling work.

Each notebook runs below 4 h (pinned soft wall 14400 s) and resume-appends to
the same /content/results/phase3 checkpoint files as the original fleet, so
partial results are retained.

Remaining n=5 work (as of 2026-08-26 local checkpoint snapshot):
  216 structures planned, 108 done, 108 missing (216 rows) distributed
  unevenly across the 12 shards.

The original 12 shards hit Colab's 10 h limit. This script slices the
*mising* iids per shard into chunks that provably stay <4 h even in the
worst observed tail (per-job worst ~6k s summed over both targets). Chunk
size is 4 iids per notebook (3 for the two shards whose max exceeds 5.5k s,
so worst wall = max*2.5 < 4 h).

Two fleets are produced in notebooks_colab/phase3_remaining/:

  Fleet R (engine-resume, ~33 notebooks):  nb30_b_resume_n5_sXX_pY
     -> runs ONLY the slice's n=5 engine jobs on 2 workers, checkpointed to
        scaling_probe_shardXX.jsonl. Safe to run all 33 in parallel (disjoint
        iid sets per shard). Median wall ~0.9-1.3 h, worst ~3.4 h.

  Fleet F (finish, 12 notebooks):  nb30_b_finish_shardXX
     -> after Fleet R completes a shard's n=5 (18/18), runs its trailing n=6
        pilot, its quota-6 attacks (SRS over undecided x RECOVERABLE rows),
        and rewrites scaling_summary_shardXX.json. If n=5 is still incomplete
        it exits early with a message. Wall <1 h per shard (1 n6 job + 6
        attacks on 2 workers).

Usage: python3 scripts/make_colab_phase3_resume.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUTDIR = ROOT / "notebooks_colab" / "phase3_remaining"
CHECKPOINT_DIR = ROOT / "notebooks_colab" / "phase3"
CFG_PATH = ROOT / "configs" / "phase3" / "scaling.json"

LIB_FILES = [
    "mdag_dgp.py",
    "lp_ground_truth.py",
    "enumerate_structures.py",
    "gluing.py",
    "battery.py",
    "engine2.py",
    "attackers.py",
    "phase3_probe.py",
]
REL_IMPORT = re.compile(r"^(\s*)from\s+\.[\w\.]*\s+import[\s(]", re.M)


def build_lib() -> str:
    chunks = []
    for name in LIB_FILES:
        text = (ROOT / "src" / "sheafpatternfusion" / name).read_text()
        lines = text.splitlines(keepends=True)
        out = []
        skipping = False
        for line in lines:
            if skipping:
                out.append("")
                if ")" in line:
                    skipping = False
                continue
            stripped = line.lstrip()
            if stripped.startswith("from __future__"):
                out.append("")
                continue
            if REL_IMPORT.match(line):
                out.append("")
                if "(" in line and ")" not in line:
                    skipping = True
                continue
            if stripped.startswith("import ."):
                out.append("")
                continue
            out.append(line)
        chunks.append("".join(out))
    lib = "\n".join(chunks)
    header = (
        "# ==========================================================================\n"
        "# EMBEDDED LIBRARY -- generated from src/sheafpatternfusion@\n"
        "# (" + ", ".join(LIB_FILES) + ") by scripts/make_colab_phase3_resume.py.\n"
        "# ==========================================================================\n"
        "from __future__ import annotations\n"
    )
    return header + lib


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(lines):
    if isinstance(lines, str):
        lines = lines.splitlines(keepends=True)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def notebook(name, cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {"colab": {"name": name}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": cells,
    }


def install_cell():
    return code_cell("""
import importlib.metadata as md
import subprocess
import sys

WANT = {'numpy': '2.4.3', 'scipy': '1.17.1'}


def _ver(pkg):
    try:
        return md.version(pkg)
    except Exception:
        return None


missing = {p: v for p, v in WANT.items() if _ver(p) != v}
if not missing:
    print('environment OK:', WANT)
else:
    print('installing pinned numpy/scipy (one-time per session) ...')
    res = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                          'numpy==2.4.3', 'scipy==1.17.1'])
    if res.returncode != 0:
        raise RuntimeError('pip install failed; see log above')
    print()
    print('=' * 72)
    print('DEPENDENCIES INSTALLED. One manual step left:')
    print('  1) Runtime > Restart session ...   (clears the old numpy/scipy)')
    print('  2) Runtime > Run all               (this cell will skip)')
    print('=' * 72)
    raise SystemExit('restart required before importing numpy/scipy')
""".lstrip().splitlines(keepends=True))


ENV_CELL = [
    "import functools\n",
    "import glob\n",
    "import io\n",
    "import json\n",
    "import multiprocessing as mp\n",
    "import os\n",
    "import pathlib\n",
    "import time\n",
    "import urllib.request\n",
    "\n",
    "os.environ['OMP_NUM_THREADS'] = '1'\n",
    "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n",
    "os.environ['MKL_NUM_THREADS'] = '1'\n",
]

RUNNER_HELPERS = '''
def _san(o):
    import numpy as _np
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def dump_line(obj):
    return json.dumps(obj, default=_san)


def load_done(path, key_fn):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                done.add(key_fn(json.loads(line)))
            except Exception:
                pass
    return done


def pooled_map_deadline(worker_fn, items, n_workers=2, stall_timeout_s=5400,
                        seconds_budget=None, meta=None):
    """Yield worker_fn(item) for all items on a fork-context pool with about
    2*n_workers futures in flight. Stops dispatching once seconds_budget is
    exhausted (running jobs drain; queued ones are cancelled and reported in
    meta['not_run']); falls back to sequential execution on pool failure.
    Job dicts may carry EITHER 'iid' or 'instance_id' as their key."""
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

    items = list(items)
    if meta is None:
        meta = {}
    meta['not_run'] = []
    meta['timed_out'] = False
    meta['completed'] = 0
    meta['done_keys'] = set()
    t_start = time.time()

    def left():
        return None if seconds_budget is None else seconds_budget - (time.time() - t_start)

    def jkey(it):
        if isinstance(it, dict):
            return it.get('iid') or it.get('instance_id') or id(it)
        return str(it)

    if len(items) <= 1 or n_workers <= 1:
        for it in items:
            if left() is not None and left() <= 0:
                meta['timed_out'] = True
                meta['not_run'] = [jkey(x) for x in items[items.index(it):]]
                return
            r = worker_fn(it)
            meta['completed'] += 1
            meta['done_keys'].add(jkey(it))
            yield r
        return

    ex = ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=mp.get_context('fork'))
    fut_item = {}
    nxt = 0
    try:
        while True:
            while nxt < len(items) and len(fut_item) < 2 * n_workers:
                if left() is not None and left() <= 0:
                    meta['timed_out'] = True
                    break
                f = ex.submit(worker_fn, items[nxt])
                fut_item[f] = items[nxt]
                nxt += 1
            if meta['timed_out']:
                meta['not_run'].extend(jkey(x) for x in items[nxt:])
                nxt = len(items)
            if fut_item:
                done_set, _ = wait(set(fut_item), timeout=stall_timeout_s,
                                   return_when=FIRST_COMPLETED)
                if not done_set:
                    raise RuntimeError(
                        f'pool stalled {stall_timeout_s}s with '
                        f'{len(fut_item)} futures pending')
                for f in done_set:
                    it = fut_item.pop(f)
                    meta['completed'] += 1
                    meta['done_keys'].add(jkey(it))
                    yield f.result()
            if nxt >= len(items) and not fut_item:
                break
            if meta['timed_out']:
                still = {}
                for f, it in fut_item.items():
                    if not f.cancel():
                        still[f] = it
                    else:
                        meta['not_run'].append(jkey(it))
                fut_item = still
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception as e:
        print(f'(pool yielded {meta["completed"]}/{len(items)} then '
              f'{type(e).__name__}; finishing remainder sequentially)',
              flush=True)
        for proc in (getattr(ex, '_processes', None) or {}).values():
            try:
                proc.kill()
            except Exception:
                pass
        ex.shutdown(wait=False, cancel_futures=True)
        for it in items:
            if jkey(it) in meta['done_keys']:
                continue
            r = worker_fn(it)
            meta['completed'] += 1
            meta['done_keys'].add(jkey(it))
            yield r
'''


def footer_files_cell_text():
    return """
output_files = sorted(glob.glob(str(OUT_DIR / '*.jsonl')) +
                      glob.glob(str(OUT_DIR / '*.json')) +
                      glob.glob(str(OUT_DIR / '*.csv')))
for output_file in output_files:
    try:
        from google.colab import files
        files.download(output_file)
        print('Downloaded:', output_file)
    except Exception as e:
        print('(Not on Colab / download skipped):', e)
""".lstrip()


def header_md(title, wp, purpose, expected, extra=""):
    return md_cell(
        f"# SheafPatternFusion Phase 3 (WP3.0 pivot-gate) - {title}\n"
        f"\nWork package: **{wp}**. {purpose}\n"
        f"\nRuntime: CPU-only (~2 cores). Expected wall time: **{expected}**. "
        "Everything is checkpointed to JSONL and resume-safe: re-running "
        "'Run all' continues where the session stopped.\n"
        f"\nFirst run: the first cell installs the pinned numpy/scipy and HALTS "
        "with a message. Do Runtime > Restart session once (clears the "
        f"preloaded binaries), then Runtime > Run all again; the install cell "
        "detects the pins and skips. The library is embedded in this notebook "
        f"(generated from sheafpatternfusion source); no package install is needed." + ("\n\n" + extra if extra else ""))


# --------------------------------------------------------------------------
# Engine-resume slice runner (engine only, no attacks/n6)
# --------------------------------------------------------------------------
RESUME_RUNNER_TMPL = r'''
T_START = time.time()
ENG_PATH = OUT_DIR / f"scaling_probe_shard{SHARD_IDX:02d}.jsonl"
SOFT = 14400.0  # 4 h wall - matches this fleet's guarantee

def elapsed():
    return time.time() - T_START

def mk_jobs(n_vars, count, seed, tag):
    jobs = sample_structures(n_vars, count, seed, f"{{tag}}_s{{SHARD_IDX:02d}}")
    for k, j in enumerate(jobs):
        j["iid"] = f"{{tag}}_s{{SHARD_IDX:02d}}_j{{k:04d}}"
        j["tag"] = tag
        j["do_attack"] = False
    return jobs

SLICE_IIDS = set(__SLICE_IIDS__)
CHUNK_LABEL = f"shard {SHARD_IDX:02d} slice {CHUNK_IDX} ({len(SLICE_IIDS)} iids)"

jobs_n5 = mk_jobs(5, int(SCALING_CFG["design"]["n5_structures_per_shard"]),
                  int(SCALING_CFG["seeds"]["structure_seed_base_n5"]) + SHARD_IDX,
                  "n5")

def engine_worker(job):
    return run_scaling_job(job, ENGINE_ONLY_CFG)

done = load_done(ENG_PATH, lambda r: r["instance_id"])
pending_all = [j for j in jobs_n5 if j["iid"] in SLICE_IIDS]
pending = [j for j in pending_all if j["iid"] not in done]
print(f"[resume {CHUNK_LABEL}] slice {len(pending_all)} iids, {len(done & SLICE_IIDS)} on file, {len(pending)} to go (elapsed {elapsed():.0f}s)", flush=True)
if not pending:
    print(f"[resume {CHUNK_LABEL}] nothing to do - slice already complete", flush=True)
else:
    # pilot the first pending job sequentially so the ETA is honest
    tp0 = time.time()
    pilot_recs = engine_worker(dict(pending[0]))
    per = time.time() - tp0
    eta_h = per * len(pending) / 2 / 3600 if len(pending) > 1 else per / 3600
    print(f"  self-pilot {pending[0]['iid']}: {per:.0f}s/job -> projection ~{eta_h:.1f} h on 2 workers; continuing", flush=True)
    with open(ENG_PATH, "a") as f:
        for r in pilot_recs:
            f.write(dump_line(r) + "\n")
    pending = pending[1:]
    # pooled remainder
    meta = {}
    t0 = time.time()
    for recs in pooled_map_deadline(engine_worker, pending, n_workers=2,
                                    stall_timeout_s=float(SCALING_CFG["deadlines"]["stall_timeout_s"]),
                                    seconds_budget=max(SOFT - elapsed(), 60),
                                    meta=meta):
        with open(ENG_PATH, "a") as f:
            for r in recs:
                f.write(dump_line(r) + "\n")
        if meta["completed"] % 2 == 0:
            el = time.time() - t0
            print(f"  [slice {CHUNK_IDX}] {meta['completed']}/{len(pending)} {el/60:.1f} min", flush=True)
    if meta.get("not_run"):
        print(f"  [slice {CHUNK_IDX}] deadline guard: {len(meta['not_run'])} jobs not run: {meta['not_run'][:8]}", flush=True)
    print(f"SLICE {CHUNK_IDX} DONE in {elapsed()/3600:.2f} h", flush=True)

# quick shard status
all_done = load_done(ENG_PATH, lambda r: r["instance_id"])
n5_done = len([i for i in all_done if i.startswith(f"n5_s{SHARD_IDX:02d}_")])
print(f"shard {SHARD_IDX:02d} n5 progress: {n5_done}/18 structures", flush=True)
'''

FINISH_RUNNER = r'''
T_START = time.time()
ENG_PATH = OUT_DIR / f"scaling_probe_shard{SHARD_IDX:02d}.jsonl"
ATT_PATH = OUT_DIR / f"scaling_attacks_shard{SHARD_IDX:02d}.jsonl"
SUM_PATH = OUT_DIR / f"scaling_summary_shard{SHARD_IDX:02d}.json"
SOFT = 14400.0
N6_GATE = float(SCALING_CFG["deadlines"]["n6_gate_elapsed_s"])

def elapsed():
    return time.time() - T_START

def mk_jobs(n_vars, count, seed, tag):
    jobs = sample_structures(n_vars, count, seed, f"{tag}_s{SHARD_IDX:02d}")
    for k, j in enumerate(jobs):
        j["iid"] = f"{tag}_s{SHARD_IDX:02d}_j{k:04d}"
        j["tag"] = tag
        j["do_attack"] = False
    return jobs

jobs_n6 = mk_jobs(6, int(SCALING_CFG["design"]["n6_pilot_per_shard"]),
                  int(SCALING_CFG["seeds"]["structure_seed_base_n6"]) + SHARD_IDX,
                  "n6")

def engine_worker(job):
    return run_scaling_job(job, ENGINE_ONLY_CFG)

# gate: only run if shard n=5 is complete
eng_rows = []
if ENG_PATH.exists():
    for line in ENG_PATH.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        eng_rows.append(r)
n5_ids = {r["instance_id"] for r in eng_rows if r.get("tag") == "n5"}
if len(n5_ids) < int(SCALING_CFG["design"]["n5_structures_per_shard"]):
    need = int(SCALING_CFG["design"]["n5_structures_per_shard"]) - len(n5_ids)
    print(f"[finish shard {SHARD_IDX:02d}] n5 still incomplete: {len(n5_ids)}/18 structures ({need} missing). Run the resume slices for this shard first; skipping attacks/n6/summary.", flush=True)
else:
    print(f"[finish shard {SHARD_IDX:02d}] n5 complete ({len(n5_ids)}/18) - proceeding to n6 + attacks", flush=True)

    # trailing n6 arm (individually deadline-gated)
    n6_done = load_done(ENG_PATH, lambda r: r["instance_id"])
    n6_pending = [j for j in jobs_n6 if j["iid"] not in n6_done and elapsed() < N6_GATE]
    for j in n6_pending:
        if elapsed() > N6_GATE:
            print("[n6] gate elapsed; stopping n6 arm", flush=True)
            break
        try:
            tp0 = time.time()
            recs = engine_worker(dict(j))
            with open(ENG_PATH, "a") as f:
                for r in recs:
                    f.write(dump_line(r) + "\n")
            print(f"  [n6] {j['iid']} {time.time()-tp0:.0f}s", flush=True)
        except Exception as e:
            print(f"  [n6] {j['iid']} FAILED {type(e).__name__}: {e}", flush=True)

    # attacks on undecided x RECOVERABLE n5 rows (quota via seeded SRS)
    rows = [r for r in eng_rows if r.get("tag") == "n5"]
    # refresh after n6 appended
    if ENG_PATH.exists():
        rows = [json.loads(l) for l in ENG_PATH.read_text().splitlines() if '"tag": "n5"' in l]
        # fallback parse
        rows = [json.loads(l) for l in ENG_PATH.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("tag") == "n5"]
    targets = [r for r in rows if r["gt_recoverable"].startswith("UNDETERMINED") and r["sheaf_recoverable"] == "RECOVERABLE"]
    quota = int(SCALING_CFG["design"]["attack_quota_per_shard"])
    att_done = load_done(ATT_PATH, lambda r: r["instance_id"] + "|" + json.dumps(r["target"]))
    import numpy as np
    rng_a = np.random.default_rng(int(SCALING_CFG["seeds"]["attack_srs_seed_base"]) + SHARD_IDX)
    keys = sorted(r["instance_id"] + "|" + json.dumps(r["target"]) for r in targets)
    picked = set()
    if len(keys) > quota:
        picks = rng_a.choice(len(keys), size=quota, replace=False)
        picked = {keys[int(i)] for i in sorted(picks)}
    else:
        picked = set(keys)
    pending_att = []
    for k in sorted(picked):
        if k in att_done:
            continue
        iid, tgt_json = k.split("|", 1)
        row = next(r for r in rows if r["instance_id"] == iid and json.dumps(r["target"]) == tgt_json)
        pending_att.append({
            "instance_id": iid, "target": row["target"], "n_vars": row["n_vars"],
            "var_parents": row["var_parents"], "r_parents": row["r_parents"],
            "seed": row["seed"], "fixed_cpt": row.get("fixed_cpt"),
            "mechanism_class": row["mechanism_class"],
            "poset_shape": row["poset_shape"],
            "sheaf_recoverable": row["sheaf_recoverable"],
        })
    skipped = [k for k in keys if k not in picked]
    print(f"[attacks] {len(targets)} undecided-x-REC rows; quota {quota}; {len(pending_att)} to run; {len(skipped)} skipped by quota", flush=True)
    kills = 0
    if pending_att:
        def attack_worker(row):
            return attack_row_fixed(row, SCALING_CFG["attack"])
        meta = {}
        t0 = time.time()
        for rec in pooled_map_deadline(attack_worker, pending_att, n_workers=2,
                                       stall_timeout_s=float(SCALING_CFG["deadlines"]["stall_timeout_s"]),
                                       seconds_budget=max(SOFT - elapsed(), 60),
                                       meta=meta):
            rec["skipped_by_quota"] = False
            with open(ATT_PATH, "a") as f:
                f.write(dump_line(rec) + "\n")
            if rec.get("verdict") == "CONFIRMED_FALSE_RECOVERABLE":
                kills += 1
                print(f"  *** CONFIRMED FALSE RECOVERABLE: {rec['instance_id']} ***", flush=True)
            if meta["completed"] % 2 == 0:
                el = time.time() - t0
                per = el / max(meta["completed"], 1)
                print(f"  [attacks] {meta['completed']}/{len(pending_att)} {el/60:.1f} min ({per:.0f}s/row)", flush=True)
        if meta.get("not_run"):
            print(f"  [attacks] deadline guard: {len(meta['not_run'])} not run", flush=True)
    with open(ATT_PATH, "a") as f:
        for k in skipped:
            iid, tgt_json = k.split("|", 1)
            if k not in att_done and k not in picked:
                pass
            # need to write SKIPPED only once - check if already on file
            if k in att_done:
                continue
            if k in picked:
                continue
            f.write(dump_line({"instance_id": iid, "target": json.loads(tgt_json), "verdict": "SKIPPED_QUOTA", "skipped_by_quota": True}) + "\n")
        # also record picked-but-already-skipped: the skipped set already handles quota-skipped
    # shard summary
    allrows = []
    if ENG_PATH.exists():
        for line in ENG_PATH.read_text().splitlines():
            try:
                allrows.append(json.loads(line))
            except Exception:
                pass
    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(__import__('numpy').median(xs)) if xs else None
    by_tag = {}
    for tag in ("n4t", "n5", "n6"):
        sub = [r for r in allrows if r.get("tag") == tag]
        if not sub:
            continue
        dec = [r for r in sub if not r["gt_recoverable"].startswith("UNDETERMINED")]
        und = [r for r in sub if r["gt_recoverable"].startswith("UNDETERMINED")]
        by_tag[tag] = {
            "rows": len(sub), "instances": len({r["instance_id"] for r in sub}),
            "decidable": len(dec),
            "decidability_rate": len(dec) / max(len(sub), 1),
            "verdict_counts": {"gt_" + v: sum(1 for r in sub if r["gt_recoverable"] == v) for v in {r["gt_recoverable"] for r in sub}},
            "sheaf_counts": {"sheaf_" + v: sum(1 for r in sub if r["sheaf_recoverable"] == v) for v in {r["sheaf_recoverable"] for r in sub}},
            "median_wall_cert_pipeline_s": med([r["wall_struct_s"] + r["wall_formula_s"] + r["wall_lp_s"] + r["wall_engine_r1_s"] + r["wall_engine_r2_s"] + r["wall_fiber_s"] for r in sub]),
            "median_wall_attack_s": med([r.get("wall_attack_s") or None for r in sub]),
            "median_frechet_width": med([r.get("frechet_width") for r in sub]),
            "median_jac_deficiency": med([r.get("jacobian_rank_deficiency") for r in sub]),
            "undecided_x_sheaf_REC": sum(1 for r in und if r["sheaf_recoverable"] == "RECOVERABLE"),
        }
    attacks = []
    if ATT_PATH.exists():
        for line in ATT_PATH.read_text().splitlines():
            try:
                a = json.loads(line)
            except Exception:
                continue
            if a.get("verdict") not in ("SKIPPED_QUOTA",):
                attacks.append(a)
    summary = {
        "shard": SHARD_IDX,
        "elapsed_s": round(elapsed(), 1),
        "coverage": {tag: {"rows": v["rows"], "planned_instances": int(SCALING_CFG["design"].get({"n4t": "n4_retime_per_shard", "n5": "n5_structures_per_shard", "n6": "n6_pilot_per_shard"}[tag]))} for tag, v in by_tag.items()},
        "by_tag": by_tag,
        "attacks": {"run": len(attacks), "kills": sum(1 for a in attacks if a.get("verdict")=="CONFIRMED_FALSE_RECOVERABLE"), "median_wall_s": med([a.get("total_wall_s") for a in attacks])},
        "note": "cert pipeline wall = struct+formula+lp+r1+r2+fiber",
    }
    SUM_PATH.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:3000])
    print(f"FINISH SHARD {SHARD_IDX} DONE in {elapsed()/3600:.2f} h", flush=True)
'''


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # remove stale duplicate
    dup = CHECKPOINT_DIR / "scaling_probe_shard04 (1).jsonl"
    if dup.exists():
        dup.unlink()
        print(f"removed duplicate {dup.name}")

    cfg = json.loads(CFG_PATH.read_text())
    lib = build_lib()
    # validate lib
    try:
        compile(lib, "<lib>", "exec")
        print(f"lib OK: {len(lib)} chars")
    except SyntaxError as e:
        raise SystemExit(f"lib syntax error: {e}")

    # snapshot missing as of generation time
    from collections import defaultdict
    import json as js

    shard_missing = {}
    shard_max = {}
    for shard in range(12):
        p = CHECKPOINT_DIR / f"scaling_probe_shard{shard:02d}.jsonl"
        lines = [js.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
        done_ids = {r["instance_id"] for r in lines}
        missing = [f"n5_s{shard:02d}_j{i:04d}" for i in range(18) if f"n5_s{shard:02d}_j{i:04d}" not in done_ids]
        shard_missing[shard] = missing
        # max for chunk sizing
        by_id = defaultdict(list)
        for r in lines:
            if r["instance_id"].startswith("n5"):
                by_id[r["instance_id"]].append(r)
        walls = []
        for iid, recs in by_id.items():
            s = recs[0].get("wall_struct_s", 0)
            s += sum(r.get("wall_formula_s", 0) + r.get("wall_lp_s", 0) + r.get("wall_engine_r1_s", 0) + r.get("wall_engine_r2_s", 0) + r.get("wall_fiber_s", 0) for r in recs)
            walls.append(s)
        shard_max[shard] = max(walls) if walls else 2000

    # build resume fleet
    n_resume = 0
    for shard in range(12):
        missing = shard_missing[shard]
        if not missing:
            continue
        chunk = 3 if shard_max[shard] > 5500 else 4
        for pi, start in enumerate(range(0, len(missing), chunk)):
            slice_ids = missing[start:start+chunk]
            # runner text with slice injected
            runner = RESUME_RUNNER_TMPL.replace("__SLICE_IIDS__", json.dumps(slice_ids))
            cells = [
                header_md(
                    f"WP3.0b RESUME n=5 - shard {shard:02d} slice {pi} ({len(slice_ids)} structures)",
                    "WP3.0b-RESUME (feeds gate G2.6)",
                    f"Completes the remaining n=5 engine jobs for original shard {shard:02d}: "
                    f"slices the {len(missing)} still-missing of 18 n=5 structures into 4-structure "
                    f"chunks (worst wall ~3.4 h on 2 workers). This slice handles {slice_ids}. "
                    f"Resume-safe - re-running skips already-written rows in scaling_probe_shard{shard:02d}.jsonl. "
                    f"After all resume slices for a shard finish (18/18), run its finish notebook.",
                    "<4 h (resume slice) - hard wall 4 h, median ~0.9-1.5 h",
                    extra=(f"Shard {shard:02d}: {len(missing)} n=5 structures still missing; "
                           f"this slice covers {len(slice_ids)} ({', '.join(slice_ids)}). "
                           f"Original shard's n4 re-timing already complete (8/8 rows each); "
                           f"n6 and attacks are deferred to the per-shard finish notebook.")),
                install_cell(),
                code_cell(ENV_CELL),
                code_cell(lib),
                code_cell([f"SCALING_CFG = json.loads(r'''{json.dumps(cfg)}''')\n"]),
                code_cell([f"SHARD_IDX = {shard}\n", f"CHUNK_IDX = {pi}\n",
                           "OUT_DIR = pathlib.Path('/content/results/phase3')\n",
                           "OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
                           "ENGINE_ONLY_CFG = dict(SCALING_CFG)\n"]),
                code_cell(RUNNER_HELPERS.lstrip("\n").splitlines(keepends=True)),
                code_cell(runner.lstrip("\n").splitlines(keepends=True)),
                code_cell(footer_files_cell_text().splitlines(keepends=True)),
            ]
            nb = notebook(f"nb30_b_resume_n5_s{shard:02d}_p{pi}", cells)
            out = OUTDIR / f"nb30_b_resume_n5_s{shard:02d}_p{pi}.ipynb"
            out.write_text(json.dumps(nb))
            n_resume += 1
            print(f"wrote {out.name}  shard {shard:02d} slice {pi}  {slice_ids}")

    # build finish fleet (12 notebooks, one per shard)
    for shard in range(12):
        cells = [
            header_md(
                f"WP3.0b FINISH - shard {shard:02d} (n6 + attacks + summary)",
                "WP3.0b-FINISH (feeds gate G2.6)",
                f"Finalizes shard {shard:02d} after its resume slices complete (18/18 n=5). "
                f"Runs the single trailing n=6 pilot, the quota-6 attacks on undecided x "
                f"RECOVERABLE n=5 rows (seeded SRS), and rewrites scaling_summary_shard{shard:02d}.json. "
                f"If n=5 is still incomplete the notebook exits cleanly with a message.",
                "<1 h (per shard finish) - hard wall 4 h",
                extra="Run this AFTER all resume slices for the shard report 18/18. It is resume-safe and can be re-run."),
            install_cell(),
            code_cell(ENV_CELL),
            code_cell(lib),
            code_cell([f"SCALING_CFG = json.loads(r'''{json.dumps(cfg)}''')\n"]),
            code_cell([f"SHARD_IDX = {shard}\n",
                       "OUT_DIR = pathlib.Path('/content/results/phase3')\n",
                       "OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
                       "ENGINE_ONLY_CFG = dict(SCALING_CFG)\n"]),
            code_cell(RUNNER_HELPERS.lstrip("\n").splitlines(keepends=True)),
            code_cell(FINISH_RUNNER.lstrip("\n").splitlines(keepends=True)),
            code_cell(footer_files_cell_text().splitlines(keepends=True)),
        ]
        nb = notebook(f"nb30_b_finish_shard{shard:02d}", cells)
        out = OUTDIR / f"nb30_b_finish_shard{shard:02d}.ipynb"
        out.write_text(json.dumps(nb))
        print(f"wrote {out.name}")

    # manifest
    manifest = {
        "generated": __import__('datetime').date.today().isoformat(),
        "fleet": "phase3b_remaining  (continuation of v0.3.0 12-shard scaling fleet)",
        "checkpoint_snapshot_dir": str(CHECKPOINT_DIR),
        "remaining_snapshot": {f"shard{shard:02d}": {"missing_n5": shard_missing[shard], "max_job_s": round(shard_max[shard])} for shard in range(12)},
        "total_missing_n5_structures": sum(len(v) for v in shard_missing.values()),
        "resume_notebooks": n_resume,
        "finish_notebooks": 12,
        "total_notebooks": n_resume + 12,
        "per_notebook_wall_guarantee": "<4 h (resume slices median 0.9-1.5 h worst 3.4 h, finish <1 h)",
        "expected_outputs": ["scaling_probe_shard*.jsonl (appended)", "scaling_attacks_shard*.jsonl", "scaling_summary_shard*.json"],
        "usage": "1) Run ALL resume slices (33 notebooks, fully parallel). 2) After each shard's n5 hits 18/18, run its finish notebook. 3) Download outputs into notebooks_colab/phase3/incoming/ and run python3 scripts/run_phase3.py --stage collect.",
    }
    (OUTDIR / "MANIFEST_remaining.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nDone: {OUTDIR}  ({n_resume} resume + 12 finish = {n_resume+12} notebooks)")


if __name__ == "__main__":
    main()
