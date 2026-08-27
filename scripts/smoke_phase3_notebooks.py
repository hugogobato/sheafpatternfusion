"""Local functional smoke test of the generated Phase-3 notebook fleet.

Executes each generated notebook's code cells IN ORDER against a sandbox
/tmp/opencode/phase3_smoke (/content paths rewritten), with configs shrunk to
toy budgets so the whole harness runs in minutes. Cells execute in the REAL
__main__ module namespace exactly like Colab does, which keeps fork-pool
pickle-by-reference semantics identical to the notebooks. Network is
exercised once (frozen merge fetch); public-dataset loaders are replaced by a
synthetic loader so the prevalence runner's analysis path runs offline.

Usage: python3 -u scripts/smoke_phase3_notebooks.py [scaling cycattack prevalence signal]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SMOKE_DIR = Path("/tmp/opencode/phase3_smoke")
NB_DIR = ROOT / "notesbooks_colab" if False else ROOT / "notebooks_colab" / "phase3"

TINY_ATTACK = {
    "a1_jump_rounds": 1, "a1_starts_per_round": 4, "a1_walk_n_seeds": 1,
    "a1_walk_steps": 5, "a1_step_size": 0.02, "a2_root_starts": 4,
    "a2_max_roots": 2, "a2_walk_follows": 0, "a2_walk_n_seeds": 1,
    "a2_walk_steps": 5, "a2_lp_vertices": 2, "a3_max_union_vars": 4,
    "a3_max_cells": 8, "a3_pin_tol": 1e-9, "phi_tol": 1e-4, "dist_tol": 1e-9,
}

CFG_MARKERS = ("SCALING_CFG = json.loads", "CYCATTACK_CFG = json.loads",
               "PREV_CFG = json.loads", "SIGNAL_CFG = json.loads")


def code_cells(name):
    nb = json.loads((NB_DIR / name).read_text())
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in nb["cells"] if c["cell_type"] == "code"]


def desandbox(s: str) -> str:
    s = s.replace("/content/results/phase3", str(SMOKE_DIR))
    s = s.replace("/content/instances_merged.jsonl",
                  str(SMOKE_DIR / "instances_merged.jsonl"))
    s = s.replace('"/content/results/phase3"', repr(str(SMOKE_DIR)))
    return s


def run_notebook(name, tweak=None):
    """Executes all cells except the installer; returns the runner source."""
    import __main__

    g = __main__.__dict__
    g.setdefault("_SMOKE_DONE", set())
    runner_cells = []
    started = False
    for i, src in enumerate(code_cells(name)):
        if "Restart session" in src or "importlib.metadata" in src:
            continue
        s = desandbox(src)
        if not started and s.lstrip().startswith(("T_START = time.time()",
                                                  "ds_reports = {}",
                                                  "MERGE_URL =",)):
            started = True
        if started:
            runner_cells.append(compile(s, f"{name}#runner{i}", "exec"))
            continue
        exec(compile(s, f"{name}#cell{i}", "exec"), g)
        if tweak and any(m in src for m in CFG_MARKERS):
            tweak(g)
            print(f"  [{name}] tweak applied")
        if "PAYLOADS = {}" in src:
            g["CYC_ROWS"] = g["CYC_ROWS"][:6]
            g["FAMILY_ROWS"] = g["FAMILY_ROWS"][:3]
            print(f"  [{name}] payloads truncated")
        print(f"  [{name}] cell {i} ok", flush=True)
    assert runner_cells, f"runner cells not found in {name}"
    return runner_cells


# --------------------------------------------------------------------------

def smoke_scaling(shard=0):
    name = f"nb30_b_scaling_shard_{shard:02d}.ipynb"
    print(f"== smoke {name}", flush=True)

    def tweak(g):
        c = g["SCALING_CFG"]
        c["design"].update({"n4_retime_per_shard": 1,
                            "n5_structures_per_shard": 2,
                            "n6_pilot_per_shard": 1,
                            "attack_quota_per_shard": 1})
        c["budgets"].update({"jump_starts": 4, "round2_multiplier": 1,
                             "fiber_starts": 6, "max_roots": 3,
                             "ci_discovery_draws": 2})
        c["attack"] = dict(TINY_ATTACK)
        c["deadlines"]["soft_wall_s"] = 1500
        c["deadlines"]["n6_gate_elapsed_s"] = 1200

    runner = run_notebook(name, tweak)
    t0 = time.time()
    for cell in runner:
        exec(cell, __import__("__main__").__dict__)
    print(f"  runner ok in {time.time() - t0:.0f}s", flush=True)
    for f in sorted(SMOKE_DIR.glob("scaling_*")):
        print("  produced:", f.name)


def smoke_cycattack(shard=0):
    name = f"nb30_c_cycattack_shard_{shard:02d}.ipynb"
    print(f"== smoke {name}", flush=True)

    def tweak(g):
        g["CYCATTACK_CFG"].update(dict(TINY_ATTACK))

    runner = run_notebook(name, tweak)
    t0 = time.time()
    for cell in runner:
        exec(cell, __import__("__main__").__dict__)
    print(f"  runner ok in {time.time() - t0:.0f}s", flush=True)


def smoke_prevalence():
    name = "nb30_a_prevalence.ipynb"
    print(f"== smoke {name}", flush=True)

    def tweak(g):
        # fast real datasets exercising the corrected UCI slugs + synthetic
        keep = {"uci_hepatitis", "uci_automobile", "uci_soybean_large"}
        reg = [d for d in g["PREV_CFG"]["dataset_registry"] if d["name"] in keep]
        reg += [{"name": "synthetic_small", "kind": "synthetic"},
                {"name": "synthetic_chainy", "kind": "synthetic"}]
        g["PREV_CFG"]["dataset_registry"] = reg
        g["PREV_CFG"]["bootstrap"]["target_minutes_total"] = 0.05
        g["PREV_CFG"]["bootstrap"]["B_max"] = 12
        g["PREV_CFG"]["subsets"]["max_per_dataset"] = 4000

    runner = run_notebook(name, tweak)
    import numpy as np

    import pandas as pd
    import __main__ as M

    def synthetic_loader(spec):
        rng = np.random.default_rng(11 if spec["name"].endswith("small") else 12)
        n, P = 800, 9
        vals = rng.random((n, P)) < 0.35
        mask = rng.random((n, P)) < 0.25
        if spec["name"].endswith("chainy"):
            for j in range(2, P):
                mask[:, j] = mask[:, j] | mask[:, j - 1]
        else:
            # forced Berge triangle on {0,1,2}: realized observed sets
            # {01},{12},{02} plus the empty set (4 realized patterns ->
            # eligible, and the three pair edges form a Berge cycle)
            MASK_TABLE = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0],
                                   [1, 1, 1]], dtype=bool)
            picks = rng.choice(4, size=n, p=[0.35, 0.25, 0.25, 0.15])
            mask[:, :3] = MASK_TABLE[picks]
            mask[:, 3:] = rng.random((n, P - 3)) < 0.25
        cols = [f"x{i}" for i in range(P)]
        df = pd.DataFrame(index=range(n), columns=cols, dtype=object)
        for j, c in enumerate(cols):
            df[c] = [None if m else int(v) for v, m in zip(vals[:, j], mask[:, j])]
        df["const"] = 1
        df["uid"] = range(n)
        return df

    M.LOADERS["synthetic"] = synthetic_loader
    t0 = time.time()
    for cell in runner:
        exec(cell, M.__dict__)
    print(f"  runner ok in {time.time() - t0:.0f}s", flush=True)
    summ = json.loads((SMOKE_DIR / "prevalence_scan.json").read_text())
    assert "WP3_0a_verdict" in summ
    print("  verdict:", summ["WP3_0a_verdict"],
          "| datasets_with_cycles:", len(summ["datasets_with_cycles"]),
          flush=True)
    small = summ["reports"].get("synthetic_small", {})
    frac = (small.get("main", {}) or {}).get("cyclic_fraction")
    assert frac is not None and frac > 0.0, f"positive control failed: {frac}"


def smoke_signal():
    name = "nb30_c_signal_analysis.ipynb"
    print(f"== smoke {name}", flush=True)

    def tweak(g):
        g["SIGNAL_CFG"]["metrics"]["null_baselines"][
            "label_permutation_within_strata"]["B"] = 60
        g["SIGNAL_CFG"]["metrics"]["null_baselines"][
            "random_m_graph_matches"]["K_per_bucket"] = 30
        g["SIGNAL_CFG"]["metrics"]["downstream"]["corr_B"] = 200
        if "MERGE_ROWS" in g:
            g["MERGE_ROWS"] = g["MERGE_ROWS"][:500]

    runner = run_notebook(name, tweak)
    t0 = time.time()
    for cell in runner:
        exec(cell, __import__("__main__").__dict__)
    print(f"  runner ok in {time.time() - t0:.0f}s", flush=True)
    sig = json.loads((SMOKE_DIR / "signal_validity.json").read_text())
    print("  mode:", sig["mode"], "| WP3.0c:", sig["WP3_0c_verdict"], flush=True)


if __name__ == "__main__":
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["scaling", "cycattack", "prevalence", "signal"]
    for w in which:
        t0 = time.time()
        {"scaling": smoke_scaling, "cycattack": smoke_cycattack,
         "prevalence": smoke_prevalence, "signal": smoke_signal}[w]()
        print(f"== {w} done in {time.time() - t0:.0f}s\n", flush=True)
