"""Generate self-contained Google Colab notebooks for Phase 2 enumeration
sharding (plan Section 11 policy).

Each notebook embeds the required package sources (zlib+b64), its shard's job
list (computed deterministically from configs/phase2/grid.json minus locally
completed rows), runs them with per-row JSONL checkpoints, and ends with the
standard files.download footer.

Usage: python3 scripts/make_colab_shards.py [--n-shards 8] [--refresh]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tarfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUTDIR = ROOT / "notebooks_colab"
SHARD_DIR_NAME = "phase2_shards"

MODULES = [
    "__init__.py",
    "mdag_dgp.py",
    "lp_ground_truth.py",
    "enumerate_structures.py",
    "engine2.py",
    "gluing.py",
]


def package_blob() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for m in MODULES:
            p = ROOT / "src" / "sheafpatternfusion" / m
            data = p.read_bytes()
            info = tarfile.TarInfo(name=f"sheafpatternfusion/{m}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode()


def build_jobs() -> list[dict]:
    """Deterministic pending-job list: grid minus rows present in the local
    instances.jsonl."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("rp2gen", ROOT / "scripts" / "run_phase2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    grid = mod.build_grid(pilot=False)
    done = set()
    f = ROOT / "results" / "phase2" / "instances.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
                done.add(rec["instance_id"] + "|" + json.dumps(rec["target"]))
            except Exception:
                pass
    out = []
    for j in grid:
        try:
            keys = mod.expected_target_keys(j)
        except Exception:
            keys = None
        if keys is not None and keys <= done:
            continue
        out.append(j)
    return out


def notebook_source(shard_idx: int, n_shards: int, jobs: list[dict], blob: str) -> str:
    jobs_json = json.dumps(jobs)
    cfg_json = (ROOT / "configs" / "phase2" / "grid.json").read_text()
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {"colab": {"name": f"phase2_shard_{shard_idx:02d}"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                f"# SheafPatternFusion Phase 2 - shard {shard_idx:02d}/{n_shards}\n",
                f"Jobs: {len(jobs)}. Self-contained: package embedded below.\n",
                "Runtime: CPU-only. Expected wall time < 10h at ~2 cores.\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": [
                "!pip install -q numpy==2.4.3 scipy==1.17.1\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": [
                "import base64, zlib, tarfile, io, os, pathlib\n",
                f"BLOB = \"{blob}\"\n",
                "raw = zlib.decompress(base64.b64decode(BLOB))\n",
                "with tarfile.open(fileobj=io.BytesIO(raw)) as tf:\n",
                "    tf.extractall('/content/')\n",
                "sys_path = '/content'\n",
                "import sys\n",
                "if sys_path not in sys.path:\n",
                "    sys.path.insert(0, sys_path)\n",
                "print('package ready')\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": [
                "import json\n",
                f"GRID_CFG = json.loads(r'''{cfg_json}''')\n",
                f"JOBS = json.loads(r'''{jobs_json}''')\n",
                f"SHARD_IDX = {shard_idx}\n",
                f"N_SHARDS = {n_shards}\n",
                "OUT_DIR = pathlib.Path('/content/results/phase2/" + SHARD_DIR_NAME + "')\n",
                "OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
                "SHARD_FILE = OUT_DIR / f'shard_{SHARD_IDX:02d}.jsonl'\n",
                "done = set()\n",
                "if SHARD_FILE.exists():\n",
                "    for line in SHARD_FILE.read_text().splitlines():\n",
                "        try:\n",
                "            rec = json.loads(line)\n",
                "            done.add(rec['instance_id'] + '|' + json.dumps(rec['target']))\n",
                "        except Exception:\n",
                "            pass\n",
                "todo = [j for j in JOBS]\n",
                "print(f'shard {SHARD_IDX}: {len(todo)} jobs, {len(done)} rows already on file')\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": [
                "# re-implement run_instance against the embedded package (no pickle,\n",
                "# sequential loop: 2 cores are enough for the per-job budgets)\n",
                "import time\n",
                "import numpy as np\n",
                "os.environ['OMP_NUM_THREADS'] = '1'\n",
                "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n",
                "os.environ['MKL_NUM_THREADS'] = '1'\n",
                "from sheafpatternfusion.engine2 import decide2, sheaf_fiber_verdict\n",
                "from sheafpatternfusion.enumerate_structures import (\n",
                "    classify, conflict_flags, discover_slice_cis, graham_acyclic,\n",
                "    pick_targets, poset_shape, instantiate)\n",
                "from sheafpatternfusion.gluing import marginal_problem_lp\n",
                "from sheafpatternfusion.lp_ground_truth import pack, unpack\n",
                "\n",
                "def run_one(job):\n",
                "    vp = {int(k): tuple(v) for k, v in job['structure']['var_parents'].items()}\n",
                "    structure = (vp, tuple(tuple(p) for p in job['structure']['r_parents']))\n",
                "    inst = instantiate(structure, seed=job['draw_seed'])\n",
                "    info = classify(inst)\n",
                "    jt = inst.joint_table()\n",
                "    patterns = inst.realized_patterns(jt=jt)\n",
                "    q = inst.observed_laws(jt)\n",
                "    pp = {}\n",
                "    for (v, r), p in jt.items():\n",
                "        pp[r] = pp.get(r, 0.0) + p\n",
                "    fam_w = {r: {o: c * pp[r] for o, c in cells.items()} for r, cells in q.items()}\n",
                "    completability = marginal_problem_lp(inst.n_vars, fam_w)['feasible']\n",
                "    sets = [frozenset(i for i in range(inst.n_vars) if r[i] == 1) for r in patterns]\n",
                "    shape = poset_shape(patterns)\n",
                "    conflicts = conflict_flags(inst)\n",
                "    cis = discover_slice_cis(inst, n_draws=16)\n",
                "    theta_true = pack(inst)\n",
                "    recs = []\n",
                "    for tgt in pick_targets(inst):\n",
                "        eng = decide2(inst, theta_true, tgt, seed=11)\n",
                "        if eng['gt_verdict'].startswith('UNDETERMINED'):\n",
                "            eng = decide2(inst, theta_true, tgt, jump_starts=90, seed=23)\n",
                "            eng['gt_evidence'] = 'round2:' + eng['gt_evidence']\n",
                "        fib = sheaf_fiber_verdict(inst, theta_true, tgt, n_starts=48, seed=13)\n",
                "        recs.append({\n",
                "            'instance_id': job['iid'], 'tag': job['tag'], 'seed': job['draw_seed'],\n",
                "            'n_vars': inst.n_vars,\n",
                "            'var_parents': {str(k): list(v) for k, v in vp.items()},\n",
                "            'r_parents': [list(p) for p in structure[1]],\n",
                "            'mechanism_class': info['mechanism_class'],\n",
                "            'has_self_edge': info['has_self_edge'],\n",
                "            'poset_shape': shape, 'graham_acyclic': bool(graham_acyclic(sets)),\n",
                "            'n_realized_patterns': len(patterns),\n",
                "            'patterns': [list(p) for p in patterns],\n",
                "            'always_observed': list(info['always_observed']),\n",
                "            'never_observed': list(info['never_observed']),\n",
                "            'target': list(tgt), 'true_value': eng.get('true_value'),\n",
                "            'gt_recoverable': eng['gt_verdict'], 'gt_evidence': eng['gt_evidence'],\n",
                "            'lp_width': eng.get('lp', {}).get('width'),\n",
                "            'witness_delta_phi': eng.get('witness', {}).get('delta_phi'),\n",
                "            'sheaf_recoverable': fib['sheaf_verdict'],\n",
                "            'phi_spread_over_fiber': fib['phi_spread_over_fiber'],\n",
                "            'n_distinct_completions': fib['n_distinct_completions'],\n",
                "            'jacobian_rank': fib['jacobian_rank'],\n",
                "            'n_free_params': fib['n_free_params'],\n",
                "            'jacobian_full_rank': bool(fib['jacobian_rank'] == fib['n_free_params']),\n",
                "            'observed_family_completable': bool(completability),\n",
                "            'conflict_mcar_style': conflicts['conflict_mcar_style'],\n",
                "            'max_cross_pattern_marginal_gap': conflicts['max_cross_pattern_marginal_gap'],\n",
                "            'n_slice_ci_constraints': int(sum(len(v) for v in cis.values())),\n",
                "            'slice_cis': {''.join(map(str, r)): [list(c) for c in lst]\n",
                "                          for r, lst in cis.items() if lst},\n",
                "            'wall_s': None, 'shard': SHARD_IDX,\n",
                "        })\n",
                "    return recs\n",
                "\n",
                "t0 = time.time()\n",
                "for k, job in enumerate(todo):\n",
                "    try:\n",
                "        recs = run_one(job)\n",
                "    except Exception as e:\n",
                "        print(f'FAILED {job[\"iid\"]}: {type(e).__name__}: {e}', flush=True)\n",
                "        continue\n",
                "    keys = {r['instance_id'] + '|' + json.dumps(r['target']) for r in recs}\n",
                "    if keys <= done:\n",
                "        continue\n",
                "    with open(SHARD_FILE, 'a') as fout:\n",
                "        for r in recs:\n",
                "            fout.write(json.dumps(r) + '\\n')\n",
                "    if (k + 1) % 10 == 0:\n",
                "        el = time.time() - t0\n",
                "        print(f'[{k+1}/{len(todo)}] {el/60:.1f} min', flush=True)\n",
                "print('SHARD DONE', flush=True)\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": [
                "output_file = str(SHARD_FILE)\n",
                "try:\n",
                "    from google.colab import files\n",
                "    files.download(output_file)\n",
                "    print('Downloaded:', output_file)\n",
                "except Exception as e:\n",
                "    print('(Not on Colab / download skipped):', e)\n"]},
        ],
    }
    return json.dumps(nb, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, default=8)
    args = ap.parse_args()
    jobs = build_jobs()
    blob = package_blob()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    est_per_shard = len(jobs) // args.n_shards
    print(f"pending jobs: {len(jobs)} -> {args.n_shards} shards "
          f"(~{est_per_shard} jobs/shard)")
    for s in range(args.n_shards):
        shard_jobs = jobs[s::args.n_shards]
        nb = notebook_source(s, args.n_shards, shard_jobs, blob)
        path = OUTDIR / f"phase2_shard_{s:02d}.ipynb"
        path.write_text(nb)
        size_kb = path.stat().st_size // 1024
        print(f"wrote {path.name} ({len(shard_jobs)} jobs, {size_kb} KB)")


if __name__ == "__main__":
    main()
