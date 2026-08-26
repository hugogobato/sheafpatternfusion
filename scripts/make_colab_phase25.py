"""Generate the 12 self-contained Phase-2.5 Colab notebooks (plan Section 11).

Thin-runner pattern proven in Phase 2: each notebook pip-installs
sheafpatternfusion@v0.3.0 (pulling ABI-matched numpy/scipy pins), RESTARTS
THE KERNEL ONCE so the upgraded binaries load cleanly, and on the second
"Run all" detects the pins, skips the install, fetches
data/frozen/instances_merged.jsonl from raw.githubusercontent at the tag
(battery + audit), and runs its job list with resumable JSONL checkpoints,
a self-pilot projection line, and the standard download footer.

Notebooks:
  nb25_00_nullbattery.ipynb      WP2.5.1 null battery (minutes)
  nb25_audit_shard_00..05.ipynb  WP2.5.2 adversarial audit, 6 shards
  nb25_cyclic_shard_00..03.ipynb WP2.5.4 forced cyclic stratum, 4 shards
  nb25_family.ipynb              WP2.5.3 discordant family

Usage: python3 scripts/make_colab_phase25.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUTDIR = ROOT / "notebooks_colab" / "phase25"
TAG = "v0.3.0"
REPO = "https://github.com/hugogobato/sheafpatternfusion.git"
FROZEN_URL = ("https://raw.githubusercontent.com/hugogobato/sheafpatternfusion/"
              f"{TAG}/data/frozen/instances_merged.jsonl")


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


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


def install_cell(pkg_version: str):
    return code_cell(f"""
import importlib.metadata as md
import subprocess
import sys

WANT = {{'numpy': '2.4.3', 'scipy': '1.17.1', 'sheafpatternfusion': '{pkg_version}'}}
TAG = '{TAG}'
REPO = '{REPO}'


def _ver(pkg):
    try:
        return md.version(pkg)
    except Exception:
        return None


missing = {{p: v for p, v in WANT.items() if _ver(p) != v}}
if not missing:
    print('environment OK:', WANT)
else:
    print('installing sheafpatternfusion@' + TAG + ' (one-time per session) ...')
    res = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                          f'git+{{REPO}}@{{TAG}}'])
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
    "import json\n",
    "import multiprocessing as mp\n",
    "import os\n",
    "import pathlib\n",
    "import time\n",
    "\n",
    "os.environ['OMP_NUM_THREADS'] = '1'\n",
    "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n",
    "os.environ['MKL_NUM_THREADS'] = '1'\n",
]

FETCH_CELL = [
    "import urllib.request\n",
    f"FROZEN_URL = {FROZEN_URL!r}\n",
    "FROZEN_PATH = pathlib.Path('/content/instances_merged.jsonl')\n",
    "if not FROZEN_PATH.exists():\n",
    "    print('fetching frozen merge from', FROZEN_URL)\n",
    "    urllib.request.urlretrieve(FROZEN_URL, FROZEN_PATH)\n",
    "ROWS = [json.loads(l) for l in open(FROZEN_PATH)]\n",
    "print('frozen merge:', len(ROWS), 'rows')\n",
]

FOOTER_FILES_CELL = [
    "import glob\n",
    "output_files = sorted(glob.glob(str(OUT_DIR / '*.jsonl')) + glob.glob(str(OUT_DIR / '*.json')))\n",
    "for output_file in output_files:\n",
    "    try:\n",
    "        from google.colab import files\n",
    "        files.download(output_file)\n",
    "        print('Downloaded:', output_file)\n",
    "    except Exception as e:\n",
    "        print('(Not on Colab / download skipped):', e)\n",
]

RUNNER_HELPERS = """
OUT_DIR = pathlib.Path('/content/results/phase25')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_done(path, key_fn):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done.add(key_fn(rec))
            except Exception:
                pass
    return done


def pooled_map(worker_fn, items, n_workers=2,
               stall_timeout_s=2400):
    '''Yield worker_fn(item) for all items, 2-process pool with a stall
watchdog: if no future completes within stall_timeout_s, workers are killed
and the remainder runs sequentially. Worker_fn must be picklable (an
importable function or a functools.partial thereof).'''
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

    items = list(items)
    if len(items) <= 1 or n_workers <= 1:
        for it in items:
            yield worker_fn(it)
        return
    ctx = mp.get_context('spawn')
    ex = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    done_count = 0
    try:
        futs = {ex.submit(worker_fn, it): it for it in items}
        pending = set(futs)
        while pending:
            done_set, pending = wait(pending, timeout=stall_timeout_s,
                                     return_when=FIRST_COMPLETED)
            if not done_set:
                raise RuntimeError(
                    f'pool stalled {stall_timeout_s}s with '
                    f'{len(pending)} futures pending')
            for f in done_set:
                done_count += 1
                yield f.result()
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception as e:
        print(f'(pool yielded {done_count}/{len(items)} results, then '
              f'{type(e).__name__}; finishing remainder sequentially)',
              flush=True)
        for proc in (getattr(ex, '_processes', None) or {}).values():
            try:
                proc.kill()
            except Exception:
                pass
        ex.shutdown(wait=False, cancel_futures=True)
        for it in items[done_count:]:
            yield worker_fn(it)
"""

def header_md(title, purpose, expected, extra=""):
    return md_cell(
        f"# SheafPatternFusion Phase 2.5 - {title}\n"
        "\n"
        f"{purpose}\n"
        "\n"
        f"Code: pip-installed from hugogobato/sheafpatternfusion @ {TAG}.\n"
        f"Runtime: CPU-only (~2 cores). Expected wall time: {expected}.\n"
        "\n"
        "First run: the first cell installs the pinned package and then HALTS "
        "with a message. Do Runtime > Restart session once (clears the "
        "preloaded numpy/scipy so the ABI-matched binaries load), then "
        "Runtime > Run all again; the install cell detects the pins and "
        "skips. Later runs on a warm session go straight through." + ("\n\n" + extra if extra else ""))


def cfg_cells(cfg_pairs):
    lines = []
    for name, obj in cfg_pairs:
        payload = obj if isinstance(obj, str) else json.dumps(obj)
        lines.append(f"{name} = json.loads(r'''{payload}''')\n")
    return code_cell(lines)


# --------------------------------------------------------------------------
# battery
# --------------------------------------------------------------------------

def battery_notebook(cfg):
    cells = [
        header_md(
            "WP2.5.1 degeneracy null battery",
            "Scores nulls N0-N4 against the certificate labels on all "
            "engine-undecided rows of the frozen merge and emits the priority "
            "sample S* for the audit.",
            "minutes"),
        install_cell("0.3.0"),
        code_cell(ENV_CELL),
        code_cell(FETCH_CELL),
        cfg_cells([("BATTERY_CFG", cfg)]),
        code_cell([
            "UNDECIDED = [r for r in ROWS if r['gt_recoverable'].startswith('UNDETERMINED')]\n",
            "print('undecided rows:', len(UNDECIDED))\n",
        ]),
        code_cell(RUNNER_HELPERS),
        code_cell("""
from sheafpatternfusion.workers import run_battery_row
from sheafpatternfusion.battery import aggregate_results

scored_path = OUT_DIR / 'null_battery_scored.jsonl'
done = load_done(scored_path, lambda r: r['instance_id'] + '|' + json.dumps(r['target']))
pending = [r for r in UNDECIDED
           if r['instance_id'] + '|' + json.dumps(r['target']) not in done]
print(f'{len(done)} scored on file, {len(pending)} to go')

if pending:
    tp0 = time.time()
    run_battery_row(dict(pending[0]))
    per = time.time() - tp0
    print(f'self-pilot: {per:.2f}s/row -> projected ~{per * len(pending) / 2 / 60:.1f} min '
          f'on 2 workers ({len(pending)} rows); continuing', flush=True)

fout = open(scored_path, 'a')
t0 = time.time()
completed = 0
for rec in pooled_map(run_battery_row, pending, n_workers=2):
    fout.write(json.dumps(rec) + '\\n')
    fout.flush()
    completed += 1
    if completed % 200 == 0:
        el = time.time() - t0
        eta = el / completed * (len(pending) - completed)
        print(f'[{completed}/{len(pending)}] {el:.0f}s elapsed, ETA {eta:.0f}s', flush=True)
fout.close()

scored = [json.loads(l) for l in open(scored_path)]
out = aggregate_results(scored, BATTERY_CFG)
(OUT_DIR / 'null_battery.json').write_text(json.dumps(out['metrics'], indent=1))
with open(OUT_DIR / 'priority_sample.jsonl', 'w') as f:
    for r in out['priority_sample']:
        f.write(json.dumps(r) + '\\n')
m = out['metrics']
print('S* size:', m['S_star_size'])
print('N0 acc:', round(m['N0_constant_recoverable']['accuracy'], 4),
      '| best N1:', m['best_swept']['N1'], '| best N3:', m['best_swept']['N3'])
print('BATTERY DONE')
"""),
        code_cell(FOOTER_FILES_CELL),
    ]
    return notebook("nb25_00_nullbattery", cells)


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------

AUDIT_RUNNER = """
from sheafpatternfusion.workers import build_s_star_ids, run_attack_job

sampling = AUDIT_CFG_FULL['sampling']
und_rec = [r for r in ROWS if r['gt_recoverable'].startswith('UNDETERMINED')
           and r['sheaf_recoverable'] == 'RECOVERABLE']
frame = {}


def add(row, stratum):
    key = row['instance_id'] + '|' + json.dumps(row['target'])
    e = frame.setdefault(key, dict(row, _strata=[]))
    e['_strata'].append(stratum)


for r in und_rec:
    if r['n_vars'] in (2, 4):
        add(r, f"n{r['n_vars']}_census")
rng = __import__('numpy').random.default_rng(int(sampling['srs_seed']))
n3 = [r for r in und_rec if r['n_vars'] == 3]
picks = rng.choice(len(n3), size=min(int(sampling['n3_srs_N']), len(n3)), replace=False)
for k in sorted(int(x) for x in picks):
    add(n3[k], 'n3_srs')
if sampling.get('include_S_star', True):
    s_star_ids = set(build_s_star_ids(ROWS, BATTERY_CFG))
    for r in und_rec:
        if r['instance_id'] + '|' + json.dumps(r['target']) in s_star_ids:
            add(r, 'S_star')

keys = sorted(frame.keys())
jobs_full = [frame[k] for k in keys]
jobs = [j for idx, j in enumerate(jobs_full) if idx % N_SHARDS == SHARD_IDX]
with open(OUT_DIR / 'audit_sample.jsonl', 'w') as f:
    for j in jobs_full:
        f.write(json.dumps({'instance_id': j['instance_id'], 'target': j['target'],
                            '_strata': j['_strata']}) + '\\n')
print(f'audit frame: {len(jobs_full)} jobs total; shard {SHARD_IDX}: {len(jobs)} jobs')

verdicts_path = OUT_DIR / f'audit_verdicts_shard{SHARD_IDX:02d}.jsonl'


def vkey(rec):
    return rec['instance_id'] + '|' + json.dumps(rec['target'])


done = load_done(verdicts_path, vkey)
pending = [j for j in jobs if vkey(j) not in done]
print(f'{len(done)} verdicts on file, {len(pending)} to go')
# === RUN ===

kills = 0
fout = open(verdicts_path, 'a')

if pending:
    tp0 = time.time()
    run_attack_job(dict(pending[0]), AUDIT_CFG_PILOT)
    per_pilot = time.time() - tp0
    est_ratio = float(AUDIT_CFG_FULL['a1_starts_per_round']) / max(
        float(AUDIT_CFG_PILOT['a1_starts_per_round']), 1)
    eta_h = per_pilot * max(est_ratio, 1) * len(pending) / 2 / 3600
    print(f'self-pilot: {per_pilot:.0f}s at pilot budgets -> rough full-budget projection '
          f'~{eta_h:.1f} h on 2 workers ({len(pending)} jobs); continuing', flush=True)

t0 = time.time()
completed = 0
worker = functools.partial(run_attack_job, cfg=AUDIT_CFG_FULL)
for rec in pooled_map(worker, pending, n_workers=2):
    completed += 1
    fout.write(json.dumps(rec) + '\\n')
    fout.flush()
    if rec['verdict'] == 'CONFIRMED_FALSE_RECOVERABLE':
        kills += 1
        print(f\"*** CONFIRMED FALSE RECOVERABLE: {rec['instance_id']} \"
              f\"via {rec['confirming_route']} ***\", flush=True)
    if completed % 5 == 0:
        el = time.time() - t0
        per = el / completed
        eta_h = per * (len(pending) - completed) / 3600
        print(f'[{completed}/{len(pending)}] {el/60:.1f} min '
              f'({per:.0f}s/job, ETA {eta_h:.1f} h, kills={kills})', flush=True)
fout.close()
print(f'AUDIT SHARD DONE kills={kills}')
"""


def audit_notebook(shard, n_shards, cfg_full, cfg_pilot, battery_cfg):
    cells = [
        header_md(
            f"WP2.5.2 adversarial audit shard {shard:02d}/{n_shards}",
            "Attacks undecided x RECOVERABLE assertions with non-shared oracles "
            "only (A1 deepened witness search with adaptive escalation, A2 "
            "completion enumeration + LP vertex harvest, A3 classical Frechet-cell "
            "certification). Recomputes S* deterministically from the frozen "
            "merge. Kill rule D2: any confirmed false RECOVERABLE kills C1.",
            "<= 10 h"),
        install_cell("0.3.0"),
        code_cell(ENV_CELL),
        code_cell(FETCH_CELL),
        cfg_cells([("BATTERY_CFG", battery_cfg),
                   ("AUDIT_CFG_FULL", cfg_full),
                   ("AUDIT_CFG_PILOT", cfg_pilot)]),
        code_cell([f"SHARD_IDX = {shard}\n", f"N_SHARDS = {n_shards}\n"]),
        code_cell(RUNNER_HELPERS),
        code_cell(AUDIT_RUNNER),
        code_cell(FOOTER_FILES_CELL),
    ]
    return notebook(f"nb25_audit_shard_{shard:02d}", cells)


# --------------------------------------------------------------------------
# cyclic
# --------------------------------------------------------------------------

CYCLIC_RUNNER = """
from sheafpatternfusion.cyclic_synth import make_cyclic_jobs
from sheafpatternfusion.workers import run_cyclic_job

jobs_all, stats = make_cyclic_jobs(CYCLIC_CFG, shard_idx=SHARD_IDX)
print('generation stats:', stats)
inst_path = OUT_DIR / f'cyclic_instances_shard{SHARD_IDX:02d}.jsonl'
done = load_done(inst_path, lambda r: r['instance_id'])
pending = [j for j in jobs_all if j['iid'] not in done]
print(f'{len(jobs_all)} accepted jobs; {len(done)} instances on file; {len(pending)} to go')
# === RUN ===

if pending:
    tp0 = time.time()
    pilot_recs = run_cyclic_job(dict(pending[0]), CYCLIC_CFG['budgets'],
                                int(CYCLIC_CFG['ci_discovery_draws']))
    per = time.time() - tp0
    eta_min = per * len(pending) / 2 / 60
    print(f'self-pilot: {per:.1f}s/job -> projected ~{eta_min:.0f} min on 2 workers '
          f'({len(pending)} jobs); continuing', flush=True)
    with open(inst_path, 'a') as fout:
        for r in pilot_recs:
            fout.write(json.dumps(r) + '\\n')
    pending = pending[1:]

mismatch_instances = 0
completed = 0
t0 = time.time()
fout = open(inst_path, 'a')
buf = []
worker = functools.partial(run_cyclic_job, budgets=CYCLIC_CFG['budgets'],
                           ci_draws=int(CYCLIC_CFG['ci_discovery_draws']))
for recs in pooled_map(worker, pending, n_workers=2):
    agree = all(((r['gt_recoverable'] == 'RECOVERABLE') ==
                 (r['sheaf_recoverable'] == 'RECOVERABLE')) for r in recs)
    mismatch_instances += int(not agree)
    buf.extend(recs)
    completed += 1
    if len(buf) >= 8:
        for r in buf:
            fout.write(json.dumps(r) + '\\n')
        buf = []
        fout.flush()
    if completed % 10 == 0:
        el = time.time() - t0
        print(f'[{completed}/{len(pending)}] {el/60:.1f} min, '
              f'mismatch_inst={mismatch_instances}', flush=True)
for r in buf:
    fout.write(json.dumps(r) + '\\n')
fout.close()
(OUT_DIR / f'cyclic_gen_stats_shard{SHARD_IDX:02d}.json').write_text(
    json.dumps({'generation': stats, 'ran': completed + 1,
                'mismatch_instances': mismatch_instances}, indent=1))
print('CYCLIC SHARD DONE')
"""


def cyclic_notebook(shard, n_shards, cfg):
    cells = [
        header_md(
            f"WP2.5.4 forced cyclic-poset stratum, shard {shard:02d}/{n_shards}",
            "Generates instances whose realized pattern poset is CYCLIC "
            "(impossible under Phase-2 sampling: interior mechanism probabilities "
            "realize every pattern, and the full simplex of observed sets is "
            "Berge-acyclic) via exact 0/1 indicator pins - triangle/square "
            "templates plus pinned rejection sampling - then runs the identical "
            "Phase-2 pipeline on each accepted instance.",
            "~1 h"),
        install_cell("0.3.0"),
        code_cell(ENV_CELL),
        cfg_cells([("CYCLIC_CFG", cfg)]),
        code_cell([f"SHARD_IDX = {shard}\n", f"N_SHARDS = {n_shards}\n"]),
        code_cell(RUNNER_HELPERS),
        code_cell(CYCLIC_RUNNER),
        code_cell(FOOTER_FILES_CELL),
    ]
    return notebook(f"nb25_cyclic_shard_{shard:02d}", cells)


# --------------------------------------------------------------------------
# family
# --------------------------------------------------------------------------

FAMILY_RUNNER = """
from sheafpatternfusion.workers import run_family_member

fam_path = OUT_DIR / 'discordant_family.jsonl'
done = load_done(fam_path, lambda r: str(r['member_id']))
pending = [k for k in range(FAMILY_CFG['n_members']) if str(k) not in done]
print(f"{FAMILY_CFG['n_members']} members planned; {len(done)} on file; "
      f"{len(pending)} to go")
# === RUN ===

if pending:
    tp0 = time.time()
    pilot_rec = run_family_member(min(pending), dict(FAMILY_CFG))
    per = time.time() - tp0
    eta_h = per * len(pending) / 2 / 3600
    print(f'self-pilot member {min(pending)}: {per:.0f}s -> projected ~{eta_h:.1f} h '
          f'on 2 workers; continuing', flush=True)

records = []
t0 = time.time()
fout = open(fam_path, 'a')
completed = 0
worker = functools.partial(run_family_member, cfg=dict(FAMILY_CFG))
for rec in pooled_map(worker, pending, n_workers=2):
    fout.write(json.dumps(rec) + '\\n')
    records.append(rec)
    completed += 1
    if rec.get('witnessed_discordant'):
        print(f"WITNESSED member {rec['member_id']} "
              f"({rec.get('model_pair_route')})", flush=True)
    if completed % 10 == 0:
        el = time.time() - t0
        print(f'[{completed}/{len(pending)}] {el/60:.1f} min', flush=True)
fout.close()

all_records = [json.loads(l) for l in open(fam_path)]
all_records = [r for r in all_records
               if isinstance(r, dict) and 'member_id' in r]
witnessed = [r for r in all_records if r.get('witnessed_discordant')]
summary = {
    'n_members': len(all_records),
    'n_witnessed_discordant': len(witnessed),
    'success_threshold': FAMILY_CFG['success_threshold'],
    'gate': ('PASS' if len(witnessed) >= FAMILY_CFG['success_threshold']
             else 'COLLAPSE'),
}
(OUT_DIR / 'family_summary.json').write_text(json.dumps(summary, indent=1))
print('FAMILY DONE', summary)
"""


def family_notebook(cfg):
    cells = [
        header_md(
            "WP2.5.3 discordant-family construction",
            "Grows the seed n3_s03759_d0 / mean(0) into a family over fresh "
            "mechanism draws of its frozen structure class. A member is "
            "witnessed-discordant iff the engine stays UNDETERMINED while "
            "independent samplers exhibit two model-valid completions differing "
            "on the target AND a classical witness exists. Gate G2.5d: >= 10 "
            "witnessed members greenlights the theory note after G2.5b "
            "adjudication.",
            "< 6 h"),
        install_cell("0.3.0"),
        code_cell(ENV_CELL),
        cfg_cells([("FAMILY_CFG", cfg)]),
        code_cell(RUNNER_HELPERS),
        code_cell(FAMILY_RUNNER),
        code_cell(FOOTER_FILES_CELL),
    ]
    return notebook("nb25_family", cells)


# --------------------------------------------------------------------------


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    battery_cfg = json.loads((ROOT / "configs" / "phase25" / "battery.json").read_text())
    audit_cfg = json.loads((ROOT / "configs" / "phase25" / "audit.json").read_text())
    audit_full = {**audit_cfg, **audit_cfg["full"]}
    audit_pilot = {**audit_cfg, **audit_cfg["pilot"]}
    cyclic_cfg = json.loads((ROOT / "configs" / "phase25" / "cyclic_grid.json").read_text())
    family_cfg = json.loads((ROOT / "configs" / "phase25" / "family.json").read_text())

    made = []

    p = OUTDIR / "nb25_00_nullbattery.ipynb"
    p.write_text(json.dumps(battery_notebook(battery_cfg), indent=1))
    made.append((p.name, "minutes"))

    n_audit = 6
    for s in range(n_audit):
        p = OUTDIR / f"nb25_audit_shard_{s:02d}.ipynb"
        p.write_text(json.dumps(
            audit_notebook(s, n_audit, audit_full, audit_pilot, battery_cfg),
            indent=1))
        made.append((p.name, "<= 10h"))

    n_cyc = int(cyclic_cfg.get("shards", 4))
    for s in range(n_cyc):
        p = OUTDIR / f"nb25_cyclic_shard_{s:02d}.ipynb"
        p.write_text(json.dumps(cyclic_notebook(s, n_cyc, cyclic_cfg), indent=1))
        made.append((p.name, "~1h"))

    p = OUTDIR / "nb25_family.ipynb"
    p.write_text(json.dumps(family_notebook(family_cfg), indent=1))
    made.append((p.name, "< 6h"))

    for name, est in made:
        size_kb = (OUTDIR / name).stat().st_size // 1024
        print(f"wrote {name} ({size_kb} KB, expected {est})")
    print(f"total: {len(made)} notebooks -> {OUTDIR}")


if __name__ == "__main__":
    main()
