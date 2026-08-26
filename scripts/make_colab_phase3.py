"""Generate the Phase-3 (WP3.0 pivot-gate) Colab notebook fleet.

Sixteen self-contained CPU notebooks in notebooks_colab/phase3/, following the
proven Phase-2/2.5 thin-runner ergonomics WITHOUT pip-installing the package:
each notebook pins numpy==2.4.3 / scipy==1.17.1 (the exact local test env),
halts once for Runtime > Restart session, then materializes the full library
(concatenation of the eight source modules with relative imports stripped --
verified by exec at generation time) in a single cell. Workers are pooled with
a FORK-context ProcessPoolExecutor (Colab is Linux), so notebook-namespace
functions are inherited by children; a stall watchdog falls back to
sequential execution exactly like the Phase-2.5 runners.

Fleet (plan Section 7 WP3.0; deviation note: 16 notebooks > the plan's
Phase-3 reserve of 10 -- user directive to maximize parallelism, still far
under the global 40 cap):

  nb30_a_prevalence.ipynb          WP3.0a natural-prevalence scan (~1-1.5 h)
  nb30_b_scaling_shard_00..09      WP3.0b scaling probe n=5 (+n4 re-timing,
                                   trailing n6 arm) (each ~2-8 h)
  nb30_c_cycattack_shard_00..03    WP3.0c labels: attacker battery on the
                                   forced-cyclic stratum's undecided rows
                                   (~2-4 h each)
  nb30_c_signal_analysis.ipynb     WP3.0c analysis (runs standalone on
                                   existing assets; upgrades itself when the
                                   scaling/cycattack outputs are dropped into
                                   /content/results/phase3)

Usage: python3 scripts/make_colab_phase3.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUTDIR = ROOT / "notebooks_colab" / "phase3"
TAG = "v0.3.0"
FROZEN_URL = ("https://raw.githubusercontent.com/hugogobato/sheafpatternfusion/"
              f"{TAG}/data/frozen/instances_merged.jsonl")
LIB_PLACEHOLDER = "<LIB>"  # replaced by build_lib() at write time

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
                out.append("")  # preserve line alignment
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
        "# (" + ", ".join(LIB_FILES) + ") by scripts/make_colab_phase3.py.\n"
        "# Relative imports are stripped; this cell defines every symbol in the\n"
        "# notebook namespace (concatenation order resolves all cross-module\n"
        "# symbols, including MDAG). Do not edit by hand -- regenerate instead.\n"
        "# ==========================================================================\n"
        "from __future__ import annotations\n"
    )
    return header + lib


# --------------------------------------------------------------------------
# notebook plumbing (mirrors the proven make_colab_phase25.py patterns)
# --------------------------------------------------------------------------

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code_cell(lines):
    if isinstance(lines, str):
        lines = lines.splitlines(keepends=True)
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": lines}


def notebook(name, cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {"colab": {"name": name},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
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


def cfg_cell(pairs):
    lines = []
    for name, obj in pairs:
        payload = obj if isinstance(obj, str) else json.dumps(obj)
        lines.append(f"{name} = json.loads(r'''{payload}''')\n")
    return code_cell(lines)


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
    meta['not_run']); falls back to sequential execution on pool failure."""
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

    items = list(items)
    if meta is None:
        meta = {}
    meta['not_run'] = []
    meta['timed_out'] = False
    meta['completed'] = 0
    meta['done_iids'] = set()
    t_start = time.time()

    def left():
        return None if seconds_budget is None else seconds_budget - (time.time() - t_start)

    if len(items) <= 1 or n_workers <= 1:
        for it in items:
            if left() is not None and left() <= 0:
                meta['timed_out'] = True
                meta['not_run'] = [x['iid'] for x in items[items.index(it):]]
                return
            r = worker_fn(it)
            meta['completed'] += 1
            meta['done_iids'].add(it['iid'])
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
                    break
                f = ex.submit(worker_fn, items[nxt])
                fut_item[f] = items[nxt]
                nxt += 1
            if left() is not None and left() <= 0 and nxt < len(items):
                meta['timed_out'] = True
                meta['not_run'].extend(x['iid'] for x in items[nxt:])
                nxt = len(items)
            still = {}
            for f, it in fut_item.items():
                if not f.cancel():
                    still[f] = it
                else:
                    meta['not_run'].append(it['iid'])
            fut_item = still
            if not fut_item:
                break
            done_set, _ = wait(set(fut_item), timeout=stall_timeout_s,
                               return_when=FIRST_COMPLETED)
            if not done_set:
                raise RuntimeError(
                    f'pool stalled {stall_timeout_s}s with '
                    f'{len(fut_item)} futures pending')
            for f in done_set:
                it = fut_item.pop(f)
                meta['completed'] += 1
                meta['done_iids'].add(it['iid'])
                yield f.result()
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
            if it['iid'] in meta['done_iids']:
                continue
            r = worker_fn(it)
            meta['completed'] += 1
            meta['done_iids'].add(it['iid'])
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
        "\n"
        f"Work package: **{wp}**. {purpose}\n"
        "\n"
        f"Runtime: CPU-only (~2 cores). Expected wall time: **{expected}**. "
        "Everything is checkpointed to JSONL and resume-safe: re-running "
        "'Run all' continues where the session stopped.\n"
        "\n"
        "First run: the first cell installs the pinned numpy/scipy and HALTS "
        "with a message. Do Runtime > Restart session once (clears the "
        "preloaded binaries), then Runtime > Run all again; the install cell "
        "detects the pins and skips. The library is embedded in this "
        "notebook (generated from sheafpatternfusion source); no package "
        "install is needed." + ("\n\n" + extra if extra else ""))


# --------------------------------------------------------------------------
# payloads for the WP3.0c family (embedded compressed local artifacts)
# --------------------------------------------------------------------------

def build_payloads():
    from sheafpatternfusion.phase3_probe import (
        compact_engine_row, compress_payload)

    cyc = []
    p = ROOT / "results" / "phase25" / "cyclic_instances.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                cyc.append(compact_engine_row(json.loads(line)))
            except Exception:
                pass
    audit = []
    p = ROOT / "results" / "phase25" / "audit_verdicts.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            a1 = r.get("A1", {}).get("wall_s")
            a2 = r.get("A2", {}).get("wall_s")
            a3 = r.get("A3", {}).get("wall_s")
            audit.append({
                "instance_id": r["instance_id"], "target": r["target"],
                "n_vars": r["n_vars"],
                "mechanism_class": r.get("mechanism_class"),
                "poset_shape": r.get("poset_shape"),
                "certificate_sheaf": r.get("certificate_sheaf"),
                "verdict": r.get("verdict"),
                "total_wall_s": r.get("total_wall_s"),
                "a1_wall_s": a1, "a2_wall_s": a2, "a3_wall_s": a3,
            })
    fam = []
    p = ROOT / "results" / "phase25" / "discordant_family.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            fam.append({
                "member_id": r.get("member_id"), "draw_seed": r["draw_seed"],
                "structure": r["structure"], "target": r["target"],
                "engine_verdict": r.get("engine_verdict"),
                "is_origin_seed": r.get("is_origin_seed"),
            })
    payloads = {
        "CYC_ROWS": compress_payload(cyc),
        "AUDIT_VERDICTS": compress_payload(audit),
        "FAMILY_ROWS": compress_payload(fam),
    }
    for k, v in payloads.items():
        kb = len(v.encode()) // 1024
        print(f"payload {k}: {kb} KB compressed "
              f"({len(cyc) if k == 'CYC_ROWS' else len(audit) if k == 'AUDIT_VERDICTS' else len(fam)} rows)")
        if kb > 900:
            raise SystemExit(f"payload {k} too large for embedding ({kb} KB)")
    return payloads


def payload_cell(payloads):
    lines = ["PAYLOADS = {}\n"]
    for k, v in payloads.items():
        lines.append(f"PAYLOADS[{k!r}] = (\n")
        for i in range(0, len(v), 96):
            lines.append(f"    {v[i:i + 96]!r}\n")
        lines.append(")\n")
    lines.append("CYC_ROWS = decompress_payload(PAYLOADS['CYC_ROWS'])\n")
    lines.append("AUDIT_VERDICTS = decompress_payload(PAYLOADS['AUDIT_VERDICTS'])\n")
    lines.append("FAMILY_ROWS = decompress_payload(PAYLOADS['FAMILY_ROWS'])\n")
    lines.append(
        "print('payloads:', len(CYC_ROWS), 'cyclic rows,',\n"
        "      len(AUDIT_VERDICTS), 'audit verdicts,', len(FAMILY_ROWS),\n"
        "      'family rows')\n")
    return code_cell(lines)


# --------------------------------------------------------------------------
# WP3.0a prevalence notebook
# --------------------------------------------------------------------------

PREVALENCE_LOADERS = '''
NH_BASE = "https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/{file}.XPT"


def load_nhanes(spec):
    frames = []
    for fname in spec["files"]:
        url = NH_BASE.format(cycle=spec["cycle"], file=fname)
        raw = urllib.request.urlopen(url, timeout=120).read()
        frames.append(pd.read_sas(io.BytesIO(raw), format="xport"))
    merged = None
    for df in frames:
        df = df.drop_duplicates(subset=[spec["merge_key"]])
        merged = df if merged is None else pd.merge(
            merged, df, on=spec["merge_key"], how="outer")
    return merged


def load_uci_zip(spec):
    raw = urllib.request.urlopen(spec["url"], timeout=180).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    parts = []
    tokens = (spec.get("na_tokens") or []) + ["NA", "", "?", "nan", "None"]
    for member in spec["members"]:
        cands = [m for m in zf.namelist() if m.split("/")[-1] == member]
        if not cands:
            cands = [m for m in zf.namelist() if m.endswith(member)]
        if not cands:
            raise FileNotFoundError(member + " not in zip")
        data = zf.read(cands[0])
        parts.append(pd.read_csv(io.BytesIO(data), header=None,
                                 na_values=tokens, skipinitialspace=True))
    return parts[0] if len(parts) == 1 else pd.concat(parts, axis=0,
                                                      ignore_index=True)


def load_openml_ds(spec):
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(name=spec["name"], version=spec.get("version"),
                      as_frame=True, parser="auto")
    X = ds.data
    try:
        if ds.target is not None:
            X = X.copy()
            X["__target__"] = np.asarray(ds.target, dtype=object)
    except Exception:
        pass
    return X


LOADERS = {"nhanes": load_nhanes, "uci_zip": load_uci_zip,
           "openml": load_openml_ds}


ID_TOKENS = ("id", "seqn", "seq", "caseid", "psu", "strata", "weight",
             "wtint", "wtmeq", "key", "index", "__target__")


def prepare_dataset(df, spec, PREV_CFG):
    """Row-cap, sentinel cleanup, candidate pool, boolean observed mask."""
    diag = {"rows_raw": int(len(df)), "cols_raw": int(df.shape[1])}
    if len(df) > int(PREV_CFG["row_cap"]):
        df = df.sample(n=int(PREV_CFG["row_cap"]),
                       random_state=int(PREV_CFG["seeds"]["subset_srs_seed_base"])
                       + abs(hash(spec["name"])) % 100000).reset_index(drop=True)
    diag["rows_used"] = int(len(df))

    df = df.replace(["?", "", "NA", "nan", "None", -999, -9999], np.nan)
    nunique = df.nunique(dropna=True)
    keep = [c for c in df.columns if nunique.get(c, 0) > 1]
    keep = [c for c in keep
            if not any(t in str(c).lower() for t in ID_TOKENS)]
    df = df[keep]
    mf = df.isna().mean()
    diag["fully_observed_cols"] = int((mf == 0).sum())

    variants = {}
    for tag, (lo, hi) in [("main", PREV_CFG["pool"]["missing_rate_main"])] + \
            [(f"sens_{lo}_{hi}", (lo, hi))
             for lo, hi in PREV_CFG["pool"]["sensitivity_missing_rates"]]:
        cand = [c for c in df.columns if lo <= mf[c] <= hi]
        cand = sorted(cand, key=lambda c: (mf[c], str(c)))
        excluded = 0
        if len(cand) > int(PREV_CFG["pool"]["max_pool_vars"]):
            idxs = np.linspace(0, len(cand) - 1,
                               int(PREV_CFG["pool"]["max_pool_vars"]))
            picked = [cand[int(round(i))] for i in idxs]
            excluded = len(cand) - len(picked)
            cand = picked
        obs = df[cand].notna().to_numpy()
        variants[tag] = {"cols": cand, "obs": obs, "excluded": excluded}
        if tag == "main":
            diag["pool_size"] = len(cand)
    diag["missing_rate_range"] = [float(mf[cand].min()), float(mf[cand].max())]
    return variants, diag


def enumerate_subsets(P, PREV_CFG, seed):
    from itertools import combinations
    subs = []
    for k in PREV_CFG["subsets"]["sizes"]:
        subs.extend(combinations(range(P), k))
    cap = int(PREV_CFG["subsets"]["max_per_dataset"])
    if len(subs) > cap:
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(subs), size=cap, replace=False)
        subs = [subs[int(i)] for i in sorted(picks)]
    return subs


def summarize_scan(records):
    elig = [r for r in records if r.get("eligible")]
    cyc = [r for r in elig if r["cyclic"]]
    return {
        "subsets_scanned": len(records),
        "eligible": len(elig),
        "cyclic": len(cyc),
        "cyclic_fraction": (len(cyc) / len(elig)) if elig else None,
        "partial_overlap_among_cyclic":
            (sum(1 for r in cyc if r.get("partial_overlap")) / len(cyc))
            if cyc else None,
        "by_size": {str(s): {
            "eligible": sum(1 for r in elig if r["size"] == s),
            "cyclic": sum(1 for r in cyc if r["size"] == s)}
            for s in sorted({r["size"] for r in elig})},
    }
'''

PREVALENCE_RUNNER = '''
ds_reports = {}
csv_rows = []
loaded = failed = 0
for di, spec in enumerate(PREV_CFG["dataset_registry"]):
    name = spec["name"]
    t0 = time.time()
    print(f"=== dataset {di + 1}/{len(PREV_CFG['dataset_registry'])}: {name} ===",
          flush=True)
    try:
        df = LOADERS[spec["kind"]](spec)
    except Exception as e:
        print(f"  LOAD FAILED ({type(e).__name__}): {e}", flush=True)
        ds_reports[name] = {"status": "load_failed",
                            "error": f"{type(e).__name__}: {e}"}
        failed += 1
        continue
    try:
        variants, diag = prepare_dataset(df, spec, PREV_CFG)
        main = variants["main"]
        if main["obs"].shape[1] < 3:
            raise ValueError(f"candidate pool too small: {main['obs'].shape[1]}")
        seed = int(PREV_CFG["seeds"]["subset_srs_seed_base"]) + di
        subs = enumerate_subsets(main["obs"].shape[1], PREV_CFG, seed)
        recs = scan_subsets(main["obs"], main["cols"], subs,
                            min_patterns=int(PREV_CFG["subsets"]["min_patterns"]),
                            min_support=int(PREV_CFG["subsets"]["min_support_main"]))
        summ = summarize_scan(recs)
        recs_r = scan_subsets(main["obs"], main["cols"], subs,
                              min_patterns=int(PREV_CFG["subsets"]["min_patterns"]),
                              min_support=int(PREV_CFG["subsets"]["min_support_robustness"]))
        summ_robust = summarize_scan(recs_r)
        ctrl_rng = np.random.default_rng(
            int(PREV_CFG["seeds"]["permutation_control_seed_base"]) + di)
        ctrl = column_permutation_control(main["obs"], ctrl_rng)
        ctrl_summ = summarize_scan(scan_subsets(
            ctrl, main["cols"], subs,
            min_patterns=int(PREV_CFG["subsets"]["min_patterns"]),
            min_support=int(PREV_CFG["subsets"]["min_support_main"])))
        sens = {}
        for tag, v in variants.items():
            if tag == "main":
                continue
            ss = enumerate_subsets(v["obs"].shape[1], PREV_CFG, seed + 1)
            sv = scan_subsets(v["obs"], v["cols"], ss,
                              min_patterns=int(PREV_CFG["subsets"]["min_patterns"]),
                              min_support=int(PREV_CFG["subsets"]["min_support_main"]))
            sens[tag] = summarize_scan(sv)["cyclic_fraction"]

        # adaptive bootstrap sized from an 8-replica pilot on this dataset
        boot = {"B": 0}
        if summ["eligible"]:
            t_b0 = time.time()
            pilot = cyclic_fraction_bootstrap(
                main["obs"], subs, B=8,
                min_patterns=int(PREV_CFG["subsets"]["min_patterns"]),
                min_support=int(PREV_CFG["subsets"]["min_support_main"]),
                seed=int(PREV_CFG["seeds"]["bootstrap_seed_base"]) + di)
            per_b = (time.time() - t_b0) / 8.0
            budget_s = float(PREV_CFG["bootstrap"]["target_minutes_total"]) * 60 \\
                * max(summ["eligible"], 1) / max(TOTAL_ELIGIBLE_EST, 1)
            B = int(np.clip(budget_s / max(per_b, 1e-6), 20,
                            int(PREV_CFG["bootstrap"]["B_max"])))
            print(f"  bootstrap pilot: {per_b:.2f}s/replica -> B={B}", flush=True)
            boot = cyclic_fraction_bootstrap(
                main["obs"], subs, B=B,
                min_patterns=int(PREV_CFG["subsets"]["min_patterns"]),
                min_support=int(PREV_CFG["subsets"]["min_support_main"]),
                seed=int(PREV_CFG["seeds"]["bootstrap_seed_base"]) + di)
            boot.pop("per_subset_cyclic_stability", None)

        for si, r in enumerate(recs):
            if r.get("eligible"):
                csv_rows.append({"dataset": name, "subset_idx": si, **{
                    k: v for k, v in r.items()},
                    "boot_stability": None})
        ds_reports[name] = {
            "status": "ok", "diag": diag,
            "main": summ, "robust_min_support": summ_robust,
            "permutation_control": ctrl_summ,
            "sensitivity_missing_window": sens,
            "bootstrap": boot,
            "wall_s": round(time.time() - t0, 1),
        }
        loaded += 1
        print(f"  eligible={summ['eligible']} cyclic={summ['cyclic']} "
              f"({summ['cyclic_fraction'] if summ['cyclic_fraction'] is not None else float('nan'):.3f}) "
              f"control={ctrl_summ['cyclic_fraction']} wall={ds_reports[name]['wall_s']}s",
              flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        ds_reports[name] = {"status": "analysis_failed",
                            "error": f"{type(e).__name__}: {e}"}
        failed += 1

with open(OUT_DIR / "prevalence_scan.csv", "w", newline="") as f:
    import csv
    if csv_rows:
        cols = list(csv_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(csv_rows)

ok = {k: v for k, v in ds_reports.values() if v.get("status") == "ok"}
datasets_with_cycles = [k for k, v in ok.items()
                        if (v["main"]["cyclic_fraction"] or 0) > 0]
elig_pool = sum(v["main"]["eligible"] for v in ok.values())
cyc_pool = sum(v["main"]["cyclic"] for v in ok.values())
pooled_fraction = cyc_pool / max(elig_pool, 1)
gates = PREV_CFG["pre_registered_gates"]
go_datasets = len(datasets_with_cycles) >= int(gates["GO_datasets_with_cycles_min"])
go_fraction = pooled_fraction >= float(gates["GO_pooled_cyclic_fraction_min"])
summary = {
    "config": "configs/phase3/prevalence.json (frozen 2026-08-26)",
    "datasets_loaded": loaded, "datasets_failed": failed,
    "datasets_with_cycles": datasets_with_cycles,
    "n_datasets_with_cycles": len(datasets_with_cycles),
    "pooled_eligible": elig_pool, "pooled_cyclic": cyc_pool,
    "pooled_cyclic_fraction": pooled_fraction,
    "gate_GO_datasets": go_datasets, "gate_GO_fraction": go_fraction,
    "WP3_0a_verdict": "GO" if (go_datasets or go_fraction) else "NO-GO",
    "reports": ds_reports,
}
(OUT_DIR / "prevalence_scan.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "reports"},
                 indent=1))
print("PREVALENCE DONE; WP3.0a verdict:", summary["WP3_0a_verdict"])
'''

TOTAL_ELIGIBLE_SNIPPET = '''
# rough cross-dataset eligibility estimate used only to split the adaptive
# bootstrap budget across datasets (refined after each dataset completes)
TOTAL_ELIGIBLE_EST = 6000
'''


def prevalence_notebook(cfg):
    cells = [
        header_md(
            "WP3.0a natural-prevalence scan",
            "WP3.0a (feeds gate G2.6)",
            "Downloads public datasets with substantive missingness (NHANES "
            "cycles, UCI sets, OpenML), realizes observed-set patterns per "
            "variable subset of size 3-6, and scores Berge-cyclicity of the "
            "overlap hypergraph with the frozen graham_acyclic. Includes "
            "min-support robustness, missing-window sensitivity, a "
            "column-permutation negative control, and an adaptive row-level "
            "bootstrap for every dataset-level cyclic fraction.",
            "~1-1.5 h"),
        install_cell(),
        code_cell(ENV_CELL),
        code_cell(LIB_PLACEHOLDER),
        cfg_cell([("PREV_CFG", cfg)]),
        code_cell([
            "import pandas as pd\n",
            "import zipfile\n",
            "OUT_DIR = pathlib.Path('/content/results/phase3')\n",
            "OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
        ]),
        code_cell(PREVALENCE_LOADERS.splitlines(keepends=True)),
        code_cell(TOTAL_ELIGIBLE_SNIPPET.splitlines(keepends=True)),
        code_cell(PREVALENCE_RUNNER.splitlines(keepends=True)),
        code_cell(footer_files_cell_text().splitlines(keepends=True)),
    ]
    return notebook("nb30_a_prevalence", cells)


# --------------------------------------------------------------------------
# WP3.0b scaling shard notebook
# --------------------------------------------------------------------------

SCALING_RUNNER = '''
T_START = time.time()
ENG_PATH = OUT_DIR / f"scaling_probe_shard{SHARD_IDX:02d}.jsonl"
ATT_PATH = OUT_DIR / f"scaling_attacks_shard{SHARD_IDX:02d}.jsonl"
SUM_PATH = OUT_DIR / f"scaling_summary_shard{SHARD_IDX:02d}.json"
SOFT = float(SCALING_CFG["deadlines"]["soft_wall_s"])
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


jobs_n4 = mk_jobs(4, int(SCALING_CFG["design"]["n4_retime_per_shard"]),
                  int(SCALING_CFG["seeds"]["structure_seed_base_n4retime"]) + SHARD_IDX,
                  "n4t")
jobs_n5 = mk_jobs(5, int(SCALING_CFG["design"]["n5_structures_per_shard"]),
                  int(SCALING_CFG["seeds"]["structure_seed_base_n5"]) + SHARD_IDX,
                  "n5")
jobs_n6 = mk_jobs(6, int(SCALING_CFG["design"]["n6_pilot_per_shard"]),
                  int(SCALING_CFG["seeds"]["structure_seed_base_n6"]) + SHARD_IDX,
                  "n6")


def engine_worker(job):
    return run_scaling_job(job, ENGINE_ONLY_CFG)


def run_engine_batch(jobs, label):
    done = load_done(ENG_PATH, lambda r: r["instance_id"])
    pending = [j for j in jobs if j["iid"] not in done]
    print(f"[{label}] {len(jobs)} planned, {len(done)} on file, "
          f"{len(pending)} to go (elapsed {elapsed():.0f}s)", flush=True)
    if not pending:
        return
    if pending:
        tp0 = time.time()
        pilot_recs = engine_worker(dict(pending[0]))
        per = time.time() - tp0
        eta_h = per * len(pending) / 2 / 3600
        print(f"  self-pilot {pending[0]['iid']}: {per:.0f}s/job -> "
              f"projection ~{eta_h:.1f} h on 2 workers; continuing", flush=True)
        with open(ENG_PATH, "a") as f:
            for r in pilot_recs:
                f.write(dump_line(r) + "\\n")
        pending = pending[1:]
    meta = {}
    t0 = time.time()
    for recs in pooled_map_deadline(engine_worker, pending, n_workers=2,
                                    stall_timeout_s=float(SCALING_CFG["deadlines"]["stall_timeout_s"]),
                                    seconds_budget=max(SOFT - elapsed(), 60),
                                    meta=meta):
        with open(ENG_PATH, "a") as f:
            for r in recs:
                f.write(dump_line(r) + "\\n")
        if meta["completed"] % 5 == 0:
            el = time.time() - t0
            print(f"  [{label}] {meta['completed']}/{len(pending)} "
                  f"{el / 60:.1f} min", flush=True)
    if meta.get("not_run"):
        print(f"  [{label}] deadline guard: {len(meta['not_run'])} jobs "
              f"not run: {meta['not_run'][:8]}{'...' if len(meta['not_run']) > 8 else ''}",
              flush=True)


run_engine_batch(jobs_n4, "n4retiming")
run_engine_batch(jobs_n5, "n5")

# ---- attacks on undecided x RECOVERABLE n5 rows (quota via seeded SRS) ----
rows = []
if ENG_PATH.exists():
    for line in ENG_PATH.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("tag") == "n5":
            rows.append(r)
targets = [r for r in rows if r["gt_recoverable"].startswith("UNDETERMINED")
           and r["sheaf_recoverable"] == "RECOVERABLE"]
quota = int(SCALING_CFG["design"]["attack_quota_per_shard"])
att_done = load_done(ATT_PATH,
                     lambda r: r["instance_id"] + "|" + json.dumps(r["target"]))
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
    row = next(r for r in rows if r["instance_id"] == iid
               and json.dumps(r["target"]) == tgt_json)
    pending_att.append({
        "instance_id": iid, "target": row["target"], "n_vars": row["n_vars"],
        "var_parents": row["var_parents"], "r_parents": row["r_parents"],
        "seed": row["seed"], "fixed_cpt": row.get("fixed_cpt"),
        "mechanism_class": row["mechanism_class"],
        "poset_shape": row["poset_shape"],
        "sheaf_recoverable": row["sheaf_recoverable"],
    })
skipped = [k for k in keys if k not in picked]
print(f"[attacks] {len(targets)} undecided-x-REC rows; quota {quota}; "
      f"{len(pending_att)} to run; {len(skipped)} skipped by quota", flush=True)
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
            f.write(dump_line(rec) + "\\n")
        if rec.get("verdict") == "CONFIRMED_FALSE_RECOVERABLE":
            kills += 1
            print(f"  *** CONFIRMED FALSE RECOVERABLE: {rec['instance_id']} ***",
                  flush=True)
        if meta["completed"] % 3 == 0:
            el = time.time() - t0
            per = el / max(meta["completed"], 1)
            print(f"  [attacks] {meta['completed']}/{len(pending_att)} "
                  f"{el / 60:.1f} min ({per:.0f}s/row)", flush=True)
    if meta.get("not_run"):
        print(f"  [attacks] deadline guard: {len(meta['not_run'])} not run",
              flush=True)
with open(ATT_PATH, "a") as f:
    for k in skipped:
        iid, tgt_json = k.split("|", 1)
        f.write(dump_line({"instance_id": iid, "target": json.loads(tgt_json),
                           "verdict": "SKIPPED_QUOTA",
                           "skipped_by_quota": True}) + "\\n")

# ---- trailing n6 arm (individually deadline-gated) ----
n6_done = load_done(ENG_PATH, lambda r: r["instance_id"])
n6_pending = [j for j in jobs_n6 if j["iid"] not in n6_done
              and elapsed() < N6_GATE]
n6_ran = 0
for j in n6_pending:
    if elapsed() > N6_GATE:
        print("[n6] gate elapsed; stopping n6 arm", flush=True)
        break
    try:
        tp0 = time.time()
        recs = engine_worker(dict(j))
        with open(ENG_PATH, "a") as f:
            for r in recs:
                f.write(dump_line(r) + "\\n")
        n6_ran += 1
        print(f"  [n6] {j['iid']} {time.time() - tp0:.0f}s", flush=True)
    except Exception as e:
        print(f"  [n6] {j['iid']} FAILED {type(e).__name__}: {e}", flush=True)

# ---- shard summary ----
allrows = []
if ENG_PATH.exists():
    for line in ENG_PATH.read_text().splitlines():
        try:
            allrows.append(json.loads(line))
        except Exception:
            pass


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


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
        "verdict_counts": {
            "gt_" + v: sum(1 for r in sub if r["gt_recoverable"] == v)
            for v in {r["gt_recoverable"] for r in sub}},
        "sheaf_counts": {
            "sheaf_" + v: sum(1 for r in sub if r["sheaf_recoverable"] == v)
            for v in {r["sheaf_recoverable"] for r in sub}},
        "median_wall_cert_pipeline_s": med([
            r["wall_struct_s"] + r["wall_formula_s"] + r["wall_lp_s"]
            + r["wall_engine_r1_s"] + r["wall_engine_r2_s"] + r["wall_fiber_s"]
            for r in sub]),
        "median_wall_attack_s": med([r.get("wall_attack_s") or None
                                     for r in sub]),
        "median_frechet_width": med([r.get("frechet_width") for r in sub]),
        "median_jac_deficiency": med([r.get("jacobian_rank_deficiency")
                                      for r in sub]),
        "undecided_x_sheaf_REC": sum(
            1 for r in und if r["sheaf_recoverable"] == "RECOVERABLE"),
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
    "coverage": {tag: {"rows": v["rows"], "planned_instances":
               int(SCALING_CFG["design"].get(
                   {"n4t": "n4_retime_per_shard", "n5": "n5_structures_per_shard",
                    "n6": "n6_pilot_per_shard"}[tag]))}
                 for tag, v in by_tag.items()},
    "by_tag": by_tag,
    "attacks": {"run": len(attacks), "kills": kills,
                "median_wall_s": med([a.get("total_wall_s") for a in attacks])},
    "note": "cert pipeline wall = struct+formula+lp+r1+r2+fiber (engine-side, "
            "comparable to the Phase-2.5 pricing convention)",
}
SUM_PATH.write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1)[:3000])
print(f"SHARD {SHARD_IDX} DONE in {elapsed() / 3600:.2f} h")
'''


def scaling_notebook(shard, n_shards, cfg):
    cells = [
        header_md(
            f"WP3.0b scaling probe, shard {shard:02d}/{n_shards}",
            "WP3.0b (feeds gate G2.6)",
            "Uniform structure sampling at n=5 (48 structures/shard, 480 "
            "fleet-wide) through the EXACT Phase-2 decision protocol (engine "
            "round1 + undecided round2 unchanged, fiber certificate, "
            "share-pinned Frechet features) with wall-clock splits per stage; "
            "a 4-instance n=4 re-timing stratum anchors the cost trend; a "
            "trailing individually-gated n=6 arm (5 structures) prices the "
            "next step; fixed-budget attackers run on undecided x "
            "RECOVERABLE rows via seeded SRS (quota 6/shard) to feed WP3.0c.",
            "~2-8 h (self-pilots and deadline-guarded)"),
        install_cell(),
        code_cell(ENV_CELL),
        code_cell(LIB_PLACEHOLDER),
        cfg_cell([("SCALING_CFG", cfg)]),
        code_cell([
            f"SHARD_IDX = {shard}\n",
            f"N_SHARDS = {n_shards}\n",
            "OUT_DIR = pathlib.Path('/content/results/phase3')\n",
            "OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "ENGINE_ONLY_CFG = dict(SCALING_CFG)\n",
        ]),
        code_cell(RUNNER_HELPERS.lstrip("\n").splitlines(keepends=True)),
        code_cell(SCALING_RUNNER.lstrip("\n").splitlines(keepends=True)),
        code_cell(footer_files_cell_text().splitlines(keepends=True)),
    ]
    return notebook(f"nb30_b_scaling_shard_{shard:02d}", cells)


# --------------------------------------------------------------------------
# WP3.0c cyclic-attack shard notebook
# --------------------------------------------------------------------------

CYCATTACK_RUNNER = '''
T_START = time.time()
VERDICT_PATH = OUT_DIR / f"cycattack_verdicts_shard{SHARD_IDX:02d}.jsonl"
SAMPLE_PATH = OUT_DIR / f"cycattack_sample_shard{SHARD_IDX:02d}.json"

und_rec = [r for r in CYC_ROWS
           if r.get("gt_recoverable", "").startswith("UNDETERMINED")
           and r.get("sheaf_recoverable") == "RECOVERABLE"]
frame = {}
for r in und_rec:
    key = r["instance_id"] + "|" + json.dumps(r["target"])
    frame[key] = r
keys_all = sorted(frame.keys())
jobs_all = [frame[k] for k in keys_all if k not in
            load_done(VERDICT_PATH,
                      lambda r: r["instance_id"] + "|" + json.dumps(r["target"]))]
jobs = jobs_all[SHARD_IDX::N_SHARDS]
with open(SAMPLE_PATH, "w") as f:
    json.dump({"n_undecided_rec_total": len(keys_all),
               "shard_jobs_pending": len(jobs),
               "shard": SHARD_IDX, "n_shards": N_SHARDS}, f)
print(f"cyclic undecided x REC rows: {len(keys_all)}; shard {SHARD_IDX} "
      f"pending: {len(jobs)}", flush=True)


def attack_worker(row):
    return attack_row_fixed(row, CYCATTACK_CFG)


kills = 0
meta = {}
t0 = time.time()
if jobs:
    tp0 = time.time()
    pilot = attack_worker(dict(jobs[0]))
    per = time.time() - tp0
    print(f"self-pilot: {per:.0f}s/row -> projection ~"
          f"{per * len(jobs) / 2 / 3600:.1f} h on 2 workers; continuing",
          flush=True)
    with open(VERDICT_PATH, "a") as f:
        f.write(dump_line(pilot) + "\\n")
    kills += int(pilot.get("verdict") == "CONFIRMED_FALSE_RECOVERABLE")
    jobs = jobs[1:]
for rec in pooled_map_deadline(attack_worker, jobs, n_workers=2,
                               stall_timeout_s=float(CYCATTACK_CFG.get("stall_timeout_s", 5400)),
                               meta=meta):
    with open(VERDICT_PATH, "a") as f:
        f.write(dump_line(rec) + "\\n")
    if rec.get("verdict") == "CONFIRMED_FALSE_RECOVERABLE":
        kills += 1
        print(f"*** CONFIRMED FALSE RECOVERABLE: {rec['instance_id']} "
              f"via {rec.get('confirming_route')} ***", flush=True)
    if meta["completed"] % 10 == 0:
        el = time.time() - t0
        per = el / max(meta["completed"], 1)
        print(f"[{meta['completed']}/{len(jobs)}] {el / 60:.1f} min "
              f"({per:.0f}s/row, kills={kills})", flush=True)
verdicts = []
if VERDICT_PATH.exists():
    for line in VERDICT_PATH.read_text().splitlines():
        try:
            v = json.loads(line)
        except Exception:
            continue
        verdicts.append(v.get("verdict"))
summary = {
    "shard": SHARD_IDX,
    "n_verdicts_on_file": len(verdicts),
    "kills_total_on_file": sum(1 for v in verdicts
                               if v == "CONFIRMED_FALSE_RECOVERABLE"),
    "wall_h": round((time.time() - T_START) / 3600, 2),
    "note": "these are sheaf-RECOVERABLE assertions on engine-undecided "
            "FORCED-CYCLIC rows; any kill feeds D2-style accounting and "
            "WP3.0c labels",
}
(OUT_DIR / f"cycattack_summary_shard{SHARD_IDX:02d}.json").write_text(
    json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print(f"CYCATTACK SHARD {SHARD_IDX} DONE")
'''


def cycattack_notebook(shard, n_shards, cfg, payloads):
    cells = [
        header_md(
            f"WP3.0c label-generation: forced-cyclic attack shard {shard:02d}/{n_shards}",
            "WP3.0c (feeds gate G2.6)",
            "Runs the full adversarial battery (A1/A2/A3, audit-identical "
            "budgets, non-shared oracles, pin-aware rebuild) on the forced "
            "cyclic stratum's engine-undecided x sheaf-RECOVERABLE rows "
            "(784 fleet-wide). Any confirmed false RECOVERABLE feeds D2-style "
            "kill accounting; every verdict becomes an attacker-derived label "
            "for the signal-validity AUC test.",
            "~2-4 h"),
        install_cell(),
        code_cell(ENV_CELL),
        code_cell(LIB_PLACEHOLDER),
        payload_cell(payloads),
        cfg_cell([("CYCATTACK_CFG", cfg)]),
        code_cell([
            f"SHARD_IDX = {shard}\n",
            f"N_SHARDS = {n_shards}\n",
            "OUT_DIR = pathlib.Path('/content/results/phase3')\n",
            "OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
        ]),
        code_cell(RUNNER_HELPERS.lstrip("\n").splitlines(keepends=True)),
        code_cell(CYCATTACK_RUNNER.lstrip("\n").splitlines(keepends=True)),
        code_cell(footer_files_cell_text().splitlines(keepends=True)),
    ]
    return notebook(f"nb30_c_cycattack_shard_{shard:02d}", cells)


# --------------------------------------------------------------------------
# WP3.0c signal-analysis notebook
# --------------------------------------------------------------------------

SIGNAL_SETUP = '''
MERGE_URL = %URL%
MERGE_PATH = pathlib.Path("/content/instances_merged.jsonl")
if not MERGE_PATH.exists():
    print("fetching frozen merge from", MERGE_URL)
    urllib.request.urlretrieve(MERGE_URL, MERGE_PATH)
MERGE_ROWS = [json.loads(l) for l in open(MERGE_PATH)]
print("frozen merge:", len(MERGE_ROWS), "rows")

OUT_DIR = pathlib.Path("/content/results/phase3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- optional ingestion of the WP3.0b/WP3.0c fleet outputs ----
UP = pathlib.Path("/content/results/phase3")
uploaded = sorted(glob.glob(str(UP / "*.jsonl")))
scale_rows, scale_attacks, cyc_attacks = [], [], []
for pth in uploaded:
    nm = os.path.basename(pth)
    try:
        for line in open(pth):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if nm.startswith("scaling_probe_") and "shard" in nm:
                scale_rows.append(rec)
            elif nm.startswith("scaling_attacks_"):
                scale_attacks.append(rec)
            elif nm.startswith("cycattack_verdicts_"):
                cyc_attacks.append(rec)
    except Exception as e:
        print("ingest failed for", nm, e)
have_fleet = bool(scale_rows or cyc_attacks)
print(f"fleet ingest: {len(scale_rows)} engine rows, "
      f"{len(scale_attacks)} scaling attacks, {len(cyc_attacks)} cycattack "
      f"verdicts -> mode: {'FINAL' if have_fleet else 'PARTIAL (existing assets only; re-run this notebook after collecting the fleet outputs for the definitive readout)'}")
'''

SIGNAL_FEATURES = '''
FEAT_PATH = OUT_DIR / "_feature_cache.jsonl"


def feature_only(row):
    """Share-pinned Frechet width + cheap structural features for one row
    (no engine, no attackers)."""
    inst, q, pp = instance_from_row_fixed(row)
    fb = frechet_bounds(inst.n_vars, q, pp, tuple(row["target"]))
    return {
        "key": row["instance_id"] + "|" + json.dumps(row["target"]),
        "frechet_width": fb["width"],
        "frac_observed": round(fraction_observed(pp, inst.n_vars), 6),
        "overlap_density": round(overlap_density(
            [tuple(p) for p in row["patterns"]]), 6),
    }


done_feats = load_done(FEAT_PATH, lambda r: r["key"])
row_lookup = {}
for r in MERGE_ROWS + CYC_ROWS:
    k = r["instance_id"] + "|" + json.dumps(r["target"])
    if k not in done_feats:
        rr = dict(r)
        rr.setdefault("fixed_cpt", None)
        row_lookup[k] = rr
pending = [row_lookup[k] for k in row_lookup]
print(f"feature cache: {len(done_feats)} on file, {len(pending)} to compute")
if pending:
    tp0 = time.time()
    feature_only(dict(pending[0]))
    per = time.time() - tp0
    print(f"self-pilot {per * 1000:.0f} ms/row -> ~{per * len(pending) / 2 / 60:.1f} min "
          "on 2 workers; continuing", flush=True)
    meta = {}
    fout = open(FEAT_PATH, "a")
    for rec in pooled_map_deadline(feature_only, pending, n_workers=2, meta=meta):
        fout.write(dump_line(rec) + "\n")
    fout.close()
FEATS = {r["key"]: r for r in map(json.loads, open(FEAT_PATH))}
print("features ready:", len(FEATS))

MERGE_KEY = {r["instance_id"] + "|" + json.dumps(r["target"]): r
             for r in MERGE_ROWS}
CYC_KEY = {r["instance_id"] + "|" + json.dumps(r["target"]): r
           for r in CYC_ROWS}


def features_for(row):
    k = row["instance_id"] + "|" + json.dumps(row["target"])
    f = FEATS.get(k, {})
    return {
        "frechet_width": f.get("frechet_width", row.get("frechet_width")),
        "frac_observed": f.get("frac_observed", row.get("frac_observed")),
        "overlap_density": f.get("overlap_density", row.get("overlap_density")),
        "jacobian_rank_deficiency": (row.get("n_free_params") - row.get("jacobian_rank"))
        if row.get("n_free_params") is not None and row.get("jacobian_rank") is not None
        else row.get("jacobian_rank_deficiency"),
        "max_cross_pattern_marginal_gap": row.get("max_cross_pattern_marginal_gap"),
    }


FEATURE_NAMES = ["frechet_width", "jacobian_rank_deficiency",
                 "max_cross_pattern_marginal_gap", "frac_observed",
                 "overlap_density"]
'''

SIGNAL_PRIMARY = '''
# ---- pool P2: attacker-labeled engine-undecided rows (PRIMARY) ----
p2 = {}


def add_p2(instance_id, target, verdict, mech, pos, nv, src):
    key = instance_id + "|" + json.dumps(target)
    if key in p2:
        p2[key]["sources"].append(src)
        return
    p2[key] = {"instance_id": instance_id, "target": target,
               "label": 1 if verdict == "CONFIRMED_FALSE_RECOVERABLE" else 0,
               "mechanism_class": mech, "poset_shape": pos, "n_vars": nv,
               "sources": [src], "verdict": verdict}


for r in AUDIT_VERDICTS:
    add_p2(r["instance_id"], r["target"], r.get("verdict"),
           r.get("mechanism_class"), r.get("poset_shape"), r.get("n_vars"),
           "audit25")
for r in cyc_attacks:
    add_p2(r["instance_id"], r["target"], r.get("verdict"),
           r.get("mechanism_class"), r.get("poset_shape"), r.get("n_vars"),
           "cycattack30")
for r in scale_attacks:
    if r.get("verdict") in ("SKIPPED_QUOTA", None):
        continue
    add_p2(r["instance_id"], r["target"], r.get("verdict"),
           r.get("mechanism_class"), r.get("poset_shape"), r.get("n_vars"),
           "scaling30")
# attack records embedded inside scaling engine rows
for r in scale_rows:
    a = r.get("attack")
    if isinstance(a, dict) and a.get("verdict") in (
            "CONFIRMED_FALSE_RECOVERABLE", "NO_FALSE_RECOVERABLE_FOUND"):
        add_p2(r["instance_id"], r["target"], a["verdict"],
               r.get("mechanism_class"), r.get("poset_shape"),
               r.get("n_vars"), "scaling_inline")

# attach features (from merge/cyc rows or from the scaling engine rows)
feat_from_scaling = {(r["instance_id"] + "|" + json.dumps(r["target"])): r
                     for r in scale_rows}
rows_p2 = []
for k, e in p2.items():
    src_row = feat_from_scaling.get(k)
    if src_row is not None:
        f = features_for(src_row)
    else:
        base = MERGE_KEY.get(k) or CYC_KEY.get(k)
        if base is None:
            continue
        f = features_for(base)
    rows_p2.append({**e, **f})
print(f"P2 attacker-labeled undecided rows: {len(rows_p2)} "
      f"(positives={sum(e['label'] for e in rows_p2)})")

# ---- pool P1: engine-decided rows (CONTEXT ONLY; trivial separation) ----
rows_p1 = []
for r in list(MERGE_ROWS) + list(CYC_ROWS):
    if r["gt_recoverable"].startswith("UNDETERMINED"):
        continue
    f = features_for(r)
    rows_p1.append({"label": 1 if r["gt_recoverable"] == "UNRECOVERABLE" else 0,
                    "n_vars": r["n_vars"],
                    "mechanism_class": r.get("mechanism_class"), **f})
print(f"P1 engine-decided context rows: {len(rows_p1)} "
      f"(positives={sum(e['label'] for e in rows_p1)})")


def auc_table(rows, tag, B_perm):
    out = {"pool": tag, "n": len(rows),
           "positives": sum(r["label"] for r in rows)}
    if not rows or len({r['label'] for r in rows}) < 2:
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
                                seed=SIGNAL_CFG["metrics"]["null_baselines"]["label_permutation_within_strata"]["seed_base"])
        out[fn] = res
    # standardized logistic combo, 5-fold stratified CV AUC
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import numpy as _np
        X = _np.array([[float(r.get(fn)) if r.get(fn) is not None else 0.0
                        for fn in FEATURE_NAMES] for r in rows])
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        aucs = cross_val_score(pipe, X, _np.array(ys), cv=cv, scoring="roc_auc")
        out["logistic_combo_cv_auc_mean"] = float(aucs.mean())
        out["logistic_combo_cv_auc_sd"] = float(aucs.std())
    except Exception as e:
        out["logistic_combo_error"] = str(e)
    return out


perm_cfg = SIGNAL_CFG["metrics"]["null_baselines"]["label_permutation_within_strata"]
res_primary = auc_table(rows_p2, "P2_attacker_labeled_undecided_PRIMARY",
                        int(perm_cfg["B"]))
res_context = auc_table(rows_p1, "P1_engine_decided_CONTEXT_CIRCULAR",
                        min(int(perm_cfg["B"]), 500))
print(json.dumps(res_primary, indent=1))
print(json.dumps(res_context, indent=1))
'''

SIGNAL_MATCH_NULL = '''
# ---- null baseline (ii): random-m-graph matches ----
K = int(SIGNAL_CFG["metrics"]["null_baselines"]["random_m_graph_matches"]["K_per_bucket"])
match_stats = {}
for n in (3, 4, 5):
    jobs = sample_structures(n, min(K, 400), seed=20460977 + n, prefix=f"match_n{n}")
    feats = []

    def match_feature(job):
        vp = {int(k): tuple(v) for k, v in job["structure"]["var_parents"].items()}
        rp = tuple(tuple(p) for p in job["structure"]["r_parents"])
        inst = instantiate((vp, rp), seed=job["draw_seed"])
        jt = inst.joint_table()
        pats = inst.realized_patterns(jt=jt)
        q = inst.observed_laws(jt)
        pp = {}
        for (v, r), pr in jt.items():
            pp[r] = pp.get(r, 0.0) + pr
        tgts = pick_targets(inst)
        if not tgts:
            return None
        fb = frechet_bounds(inst.n_vars, q, pp, tuple(tgts[0]))
        return {
            "frechet_width": fb["width"],
            "frac_observed": round(fraction_observed(pp, inst.n_vars), 6),
            "overlap_density": round(overlap_density([tuple(p) for p in pats]), 6),
        }

    meta = {}
    for f in pooled_map_deadline(match_feature, jobs, n_workers=2, meta=meta):
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
            ks[fn] = {"ks_stat": float(res.statistic), "p_value": float(res.pvalue),
                      "n_match": len(a), "n_real": len(b)}
    match_stats[f"n{n}"] = ks
print(json.dumps(match_stats, indent=1))
'''

SIGNAL_DOWNSTREAM = '''
# ---- downstream add-on: spread vs naive-pooling error across DGP draws ----
corr_inputs = []
for r in list(MERGE_ROWS) + list(CYC_ROWS):
    if r["gt_recoverable"] == "UNRECOVERABLE":
        rr = dict(r)
        rr["source_tag"] = "cyclic" if r.get("tag") == "cyclic" else "merge"
        corr_inputs.append(rr)
fam_extra = []
for fr in FAMILY_ROWS:
    try:
        vp = {int(k): tuple(v) for k, v in fr["structure"]["var_parents"].items()}
        rp = tuple(tuple(p) for p in fr["structure"]["r_parents"])
        inst = instantiate((vp, rp), seed=fr["draw_seed"])
        theta = pack(inst)
        tgt = tuple(fr["target"])
        fib = sheaf_fiber_verdict(inst, theta, tgt, n_starts=48, max_roots=12,
                                  seed=13)
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
        print("family row skipped:", fr.get("member_id"), e)
corr_rows = corr_inputs + fam_extra
table = spread_naive_table(corr_rows)
print(f"downstream rows: {len(table)} (family recomputed: {len(fam_extra)})")
corr_out = {}
sp = [t["spread"] for t in table]
ne = [t["naive_abs_err"] for t in table]
for method in SIGNAL_CFG["metrics"]["downstream"]["methods"]:
    corr_out[method] = permutation_corr_p(
        sp, ne, B=int(SIGNAL_CFG["metrics"]["downstream"]["corr_B"]),
        seed=20460901, method=method)
fw = [t.get("frechet_width") for t in table]
corr_out["frechet_vs_naive_err_spearman"] = permutation_corr_p(
    fw, ne, B=int(SIGNAL_CFG["metrics"]["downstream"]["corr_B"]),
    seed=20460902, method="spearman")
print(json.dumps(corr_out, indent=1))
'''

SIGNAL_ASSEMBLE = '''
g = SIGNAL_CFG["pre_registered_gates"]
auc_arm = None
if res_primary.get("positives", 0) > 0:
    best = max((v.get("auc") or 0.0 for k, v in res_primary.items()
                if k in FEATURE_NAMES), default=0.0)
    best_p = min((v.get("p_value") if v.get("p_value") is not None else 1.0
                  for k, v in res_primary.items() if k in FEATURE_NAMES),
                 default=1.0)
    combo = res_primary.get("logistic_combo_cv_auc_mean")
    auc_arm = (best >= float(g["GO_auc_min"]) and best_p < float(g["GO_p_max"])) \
        or (combo is not None and combo >= float(g["GO_auc_min"]))
rho_arm = None
sp_res = corr_out.get("spearman", {})
if sp_res:
    rho_arm = abs(sp_res.get("rho") or 0.0) >= float(g["GO_rho_abs_min"]) \
        and (sp_res.get("p_two_sided") or 1.0) < float(g["GO_p_max"])

signal = {
    "mode": "FINAL" if have_fleet else "PARTIAL_existing_assets_only",
    "primary_auc": res_primary,
    "context_circular": res_context,
    "random_m_graph_match_null": match_stats,
    "downstream_correlation": corr_out,
    "arm_auc_GO": auc_arm,
    "arm_rho_GO": rho_arm,
    "WP3_0c_verdict": ("GO" if (auc_arm or rho_arm) else
                       ("NO-GO" if have_fleet and res_primary.get("positives", 0) > 0
                        else "INDETERMINATE_no_positive_labels_yet")),
    "gates": g,
}
(OUT_DIR / "signal_validity.json").write_text(json.dumps(signal, indent=1))
with open(OUT_DIR / "signal_validity.csv", "w", newline="") as f:
    import csv
    wr = csv.writer(f)
    wr.writerow(["section", "feature", "metric", "value"])
    for k, v in res_primary.items():
        if isinstance(v, dict):
            for mk, mv in v.items():
                wr.writerow(["primary_auc", k, mk, mv])
        else:
            wr.writerow(["primary_auc", "", k, v])
    for k, v in res_context.items():
        if isinstance(v, dict):
            for mk, mv in v.items():
                wr.writerow(["context_circular", k, mk, mv])
        else:
            wr.writerow(["context_circular", "", k, v])
    for method, v in corr_out.items():
        for mk, mv in v.items():
            wr.writerow(["downstream_corr", "", f"{method}.{mk}", mv])
print(json.dumps({k: v for k, v in signal.items()
                  if k in ("mode", "arm_auc_GO", "arm_rho_GO",
                           "WP3_0c_verdict")}, indent=1))
print("SIGNAL ANALYSIS DONE; WP3.0c verdict:", signal["WP3_0c_verdict"])
'''


def signal_notebook(cfg, payloads):
    setup = SIGNAL_SETUP.replace("%URL%", repr(FROZEN_URL))
    cells = [
        header_md(
            "WP3.0c signal-validity analysis",
            "WP3.0c (feeds gate G2.6)",
            "Tests whether ANY sheaf-side feature predicts ground truth once "
            "the trivial separation is excluded: fiber spread is NEVER a "
            "predictor here (it is the certificate's own verdict input). "
            "Primary pool = attacker-labeled engine-undecided rows (audit "
            "verdicts embedded; upgrades automatically when the WP3.0b/WP3.0c "
            "fleet outputs are dropped into /content/results/phase3). "
            "Nulls: stratified label permutation + random-m-graph feature "
            "matches. Downstream arm: fiber spread vs naive-pooling error "
            "across DGP draws (family spreads recomputed with the protocol "
            "budgets).",
            "~1 h standalone"),
        install_cell(),
        code_cell(ENV_CELL),
        code_cell(LIB_PLACEHOLDER),
        payload_cell(payloads),
        cfg_cell([("SIGNAL_CFG", cfg)]),
        code_cell(setup.splitlines(keepends=True)),
        code_cell(RUNNER_HELPERS.lstrip("\n").splitlines(keepends=True)),
        code_cell(SIGNAL_FEATURES.lstrip("\n").splitlines(keepends=True)),
        code_cell(SIGNAL_PRIMARY.lstrip("\n").splitlines(keepends=True)),
        code_cell(SIGNAL_MATCH_NULL.lstrip("\n").splitlines(keepends=True)),
        code_cell(SIGNAL_DOWNSTREAM.lstrip("\n").splitlines(keepends=True)),
        code_cell(SIGNAL_ASSEMBLE.lstrip("\n").splitlines(keepends=True)),
        code_cell(footer_files_cell_text().splitlines(keepends=True)),
    ]
    return notebook("nb30_c_signal_analysis", cells)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

EXPECTED_OUTPUTS = {
    "nb30_a_prevalence.ipynb": ["prevalence_scan.csv", "prevalence_scan.json"],
    "nb30_b_scaling_shard_*.ipynb": ["scaling_probe_shard*.jsonl",
                                     "scaling_attacks_shard*.jsonl",
                                     "scaling_summary_shard*.json"],
    "nb30_c_cycattack_shard_*.ipynb": ["cycattack_verdicts_shard*.jsonl",
                                       "cycattack_sample_shard*.json",
                                       "cycattack_summary_shard*.json"],
    "nb30_c_signal_analysis.ipynb": ["signal_validity.json",
                                     "signal_validity.csv"],
}


def validate_lib(lib: str):
    ns = {}
    exec(compile(lib, "<phase3-lib>", "exec"), ns)
    required = ["run_scaling_job", "graham_acyclic", "frechet_bounds",
                "attack_row_fixed", "sample_structures", "scan_subsets",
                "rank_auc", "decompress_payload", "compress_payload",
                "decide2_timed", "spread_naive_table", "permutation_corr_p",
                "column_permutation_control", "realized_pattern_counts"]
    missing = [k for k in required if not callable(ns.get(k))]
    if missing:
        raise SystemExit(f"LIB validation failed, missing: {missing}")
    print(f"LIB validated: {len(lib.splitlines())} lines, "
          f"{len(lib) // 1024} KB, all {len(required)} entry points present")


def validate_nb(nb, path):
    try:
        import nbformat
        nbformat.validate(nbformat.reads(json.dumps(nb), as_version=4))
        kind = "nbformat-valid"
    except ImportError:
        json.loads(json.dumps(nb))
        assert nb["nbformat"] == 4 and nb["cells"]
        kind = "json-ok (nbformat not installed)"
    print(f"  {path.name}: {kind}, {len(nb['cells'])} cells, "
          f"{path.stat().st_size // 1024} KB")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lib = build_lib()
    validate_lib(lib)
    payloads = build_payloads()

    cfg_prev = json.loads((ROOT / "configs" / "phase3" / "prevalence.json").read_text())
    cfg_scale = json.loads((ROOT / "configs" / "phase3" / "scaling.json").read_text())
    cfg_sig = json.loads((ROOT / "configs" / "phase3" / "signal.json").read_text())
    # the cyclic-attack shards reuse the AUDIT-FULL budgets verbatim so their
    # economics are directly comparable to the Phase-2.5 pricing table
    cfg_audit = json.loads((ROOT / "configs" / "phase25" / "audit.json").read_text())
    cfg_cycattack = {**cfg_audit, **cfg_audit["full"]}

    made = []

    def write(path, nb):
        for cell in nb["cells"]:
            if cell["cell_type"] == "code" and cell["source"] == [LIB_PLACEHOLDER]:
                cell["source"] = lib.splitlines(keepends=True)
        path.write_text(json.dumps(nb, indent=1))
        validate_nb(nb, path)
        made.append(path.name)

    p = OUTDIR / "nb30_a_prevalence.ipynb"
    write(p, prevalence_notebook(cfg_prev))

    n_b = int(cfg_scale["shards"])
    for s in range(n_b):
        p = OUTDIR / f"nb30_b_scaling_shard_{s:02d}.ipynb"
        write(p, scaling_notebook(s, n_b, cfg_scale))

    n_c = 4
    for s in range(n_c):
        p = OUTDIR / f"nb30_c_cycattack_shard_{s:02d}.ipynb"
        write(p, cycattack_notebook(s, n_c, cfg_cycattack, payloads))

    p = OUTDIR / "nb30_c_signal_analysis.ipynb"
    write(p, signal_notebook(cfg_sig, payloads))

    manifest = {
        "tag_basis": TAG,
        "frozen_url": FROZEN_URL,
        "generated": "2026-08-26",
        "notebooks": {}, "expected_outputs": EXPECTED_OUTPUTS,
    }
    for name in made:
        manifest["notebooks"][name] = {"size_kb": (OUTDIR / name).stat().st_size // 1024}
    (OUTDIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"\ntotal: {len(made)} notebooks -> {OUTDIR}")
    print("collection: drop each notebook's downloaded outputs into "
          "notebooks_colab/phase3/incoming/ then run "
          "python3 scripts/run_phase3.py --stage collect")


if __name__ == "__main__":
    main()
