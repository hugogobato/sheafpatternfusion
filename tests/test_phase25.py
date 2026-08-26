"""Phase 2.5 module tests (battery nulls, attackers, cyclic synth, family)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sheafpatternfusion.battery import (
    evaluate_nulls,
    fraction_observed,
    frechet_bounds,
    instance_from_row,
    overlap_density,
)
from sheafpatternfusion.cyclic_synth import (
    TEMPLATES,
    make_cyclic_jobs,
    realize,
    signature,
)
from sheafpatternfusion.discordant_family import (
    SEED_ROW,
    evaluate_member,
    member_draw_seed,
)
from sheafpatternfusion.engine2 import decide2
from sheafpatternfusion.enumerate_structures import instantiate, poset_shape
from sheafpatternfusion.lp_ground_truth import pack, target_value_phi, unpack


def _joint(inst):
    m = unpack(inst, pack(inst))
    jt = m.joint_table()
    q = m.observed_laws(jt)
    pp = {}
    for (v, r), p in jt.items():
        pp[r] = pp.get(r, 0.0) + p
    return m, jt, q, pp


# ---------------- Frechet bounds ----------------

def test_frechet_bounds_sound_and_share_aware():
    from sheafpatternfusion.mdag_dgp import MDAG

    inst = MDAG(2, {0: (), 1: ()}, {0: (), 1: ()})
    inst.random_fill(np.random.default_rng(3))
    m, jt, q, pp = _joint(inst)
    miss_mass = sum(w for r, w in pp.items() if r[0] == 0)
    fb = frechet_bounds(2, q, pp, ("mean", 0))
    assert fb["lp_status"] == 0
    assert abs(fb["width"] - miss_mass) < 1e-6
    true_phi = target_value_phi(m, ("mean", 0))
    assert fb["lo"] - 1e-9 <= true_phi <= fb["hi"] + 1e-9


def test_frechet_bounds_degenerate_when_target_unobserved():
    from sheafpatternfusion.mdag_dgp import MDAG

    inst = MDAG(2, {0: (), 1: ()}, {0: (), 1: ()})
    inst.random_fill(np.random.default_rng(5))
    inst.r_cpt[0][()] = 0.0
    m, jt, q, pp = _joint(inst)
    assert all(r[0] == 0 for r in q)
    fb = frechet_bounds(2, q, pp, ("mean", 0))
    assert fb["width"] == 1.0


def test_frechet_bounds_sound_random_instances():
    rng = np.random.default_rng(11)
    for trial in range(6):
        n = 3
        vd = {i: tuple(int(x) for x in rng.choice(i, size=int(rng.integers(0, i + 1)),
                                       replace=False)) if i else ()
              for i in range(n)}
        rp = tuple(tuple(int(x) for x in rng.choice(n, size=int(rng.integers(0, 3)),
                                    replace=False))
                   for _ in range(n))
        inst = instantiate((vd, rp), seed=1000 + trial)
        m, jt, q, pp = _joint(inst)
        tgt = ("mean", int(np.argmax([sum(1 for r in q if r[i] == 0)
                                      for i in range(n)])))
        fb = frechet_bounds(n, q, pp, tgt)
        assert fb["lp_status"] == 0
        true_phi = target_value_phi(m, tgt)
        assert fb["lo"] - 1e-7 <= true_phi <= fb["hi"] + 1e-7
        assert fb["width"] <= 1.0 + 1e-9


# ---------------- battery nulls ----------------

def _battery_rows():
    rows = []
    specs = [
        ("mcar_a", {0: (), 1: (), 2: ()}, [(), (), ()], "RECOVERABLE"),
        ("selfc_a", {0: (), 1: ()}, [(0,), (1,)], "UNRECOVERABLE"),
        ("mixed_a", {0: (), 1: (0,)}, [(1,), (0, 1)], "RECOVERABLE"),
    ]
    for iid, vp, rp, label in specs:
        inst = instantiate(({k: v for k, v in vp.items()}, tuple(rp)),
                           seed=abs(hash(iid)) % 100000)
        jt = inst.joint_table()
        pats = inst.realized_patterns(jt=jt)
        rows.append({
            "instance_id": iid,
            "var_parents": {str(k): list(v) for k, v in vp.items()},
            "r_parents": [list(p) for p in rp],
            "seed": abs(hash(iid)) % 100000,
            "n_vars": len(vp),
            "target": ["mean", 0],
            "sheaf_recoverable": label,
            "patterns": [list(p) for p in pats],
            "true_value": None,
        })
    return rows


def test_evaluate_nulls_metrics_and_priority_sample():
    out = evaluate_nulls(_battery_rows(), {"priority_sample_cap": 2})
    m = out["metrics"]
    assert m["n_rows"] == 3
    for k in ("N0_constant_recoverable", "N4_constant_unrecoverable"):
        assert "accuracy" in m[k]
    assert set(m["best_swept"]) >= {"N1", "N2", "N3"}
    assert m["S_star_size"] <= 2
    reasons = {r["reason"] for r in out["priority_sample"]}
    assert any(r.startswith("widest_frechet") for r in reasons)


def test_feature_helpers_ranges():
    pp = {(1, 0): 0.25, (0, 1): 0.75}
    f = fraction_observed(pp, 2)
    assert 0.0 <= f <= 1.0 and abs(f - 0.5) < 1e-12
    d = overlap_density([(1, 1, 0), (1, 0, 1), (0, 1, 1)])
    assert abs(d - 1 / 3) < 1e-12
    assert overlap_density([(1, 0)]) == 0.0


# ---------------- attackers ----------------

def test_deepened_witness_finds_known_unrecoverable():
    from sheafpatternfusion.attackers import deepened_witness_search

    inst = instantiate(({0: (), 1: ()}, ((0,), ())), seed=2024)
    theta = pack(inst)
    res = deepened_witness_search(
        inst, theta, ("mean", 0),
        {"a1_jump_rounds": 1, "a1_starts_per_round": 25,
         "a1_walk_n_seeds": 2, "a1_walk_steps": 10,
         "a1_adaptive_stop": False},
        seed=1)
    assert res["confirmed_false_recoverable"]
    assert res["budget_starts"] >= 25


def test_frechet_cell_scan_detects_conflicting_strata():
    from sheafpatternfusion.attackers import frechet_cell_scan

    inst = instantiate(({0: (), 1: ()}, ((), (1,), )[:2]), seed=77)
    m, jt, q, pp = _joint(inst)
    scan = frechet_cell_scan(
        inst, q, pp, ("mean", 0),
        {"a3_max_union_vars": 4, "a3_max_cells": 50})
    assert scan["n_cells_candidate"] >= 1
    assert scan["attacker"] == "A3_frechet_cells"


def test_attack_row_smoke_on_frozen_row_if_available():
    frozen = Path(__file__).resolve().parents[1] / "data" / "frozen" / \
        "instances_merged.jsonl"
    alt = Path(__file__).resolve().parents[1] / "results" / "phase2" / \
        "instances_merged.jsonl"
    src = frozen if frozen.exists() else alt
    if not src.exists():
        pytest.skip("no merged phase-2 file on this machine")
    row = None
    with open(src) as f:
        for line in f:
            r = json.loads(line)
            if r["gt_recoverable"].startswith("UNDETERMINED") and \
                    r["sheaf_recoverable"] == "RECOVERABLE" and r["n_vars"] == 2:
                row = r
                break
    if row is None:
        pytest.skip("no qualifying row")
    from sheafpatternfusion.attackers import attack_row

    rec = attack_row(row, {
        "a1_jump_rounds": 1, "a1_starts_per_round": 8,
        "a1_walk_n_seeds": 2, "a1_walk_steps": 10,
        "a2_root_starts": 8, "a2_max_roots": 4, "a2_walk_follows": 1,
        "a2_walk_n_seeds": 2, "a2_walk_steps": 10, "a2_lp_vertices": 4,
        "a3_max_cells": 30})
    assert rec["verdict"] in ("CONFIRMED_FALSE_RECOVERABLE",
                              "NO_FALSE_RECOVERABLE_FOUND")
    for k in ("A1", "A2", "A3"):
        assert k in rec
    assert rec["total_wall_s"] > 0


# ---------------- cyclic synthesis ----------------

@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_templates_realize_cyclic_posets(name):
    for s in range(5):
        prop = TEMPLATES[name](seed=s)
        inst, pats, shape = realize(prop, draw_seed=12345 + s)
        assert shape == "cyclic"
        assert len(pats) >= 2


def test_make_cyclic_jobs_deterministic_and_capped():
    cfg = {"seed_base": 42, "target_per_shard": 5, "max_attempts": 400,
           "shards": 2,
           "proposal_mix": {"triangle_n3": 0.6, "square_n4": 0.2,
                            "rejection_n3": 0.2}}
    j1, s1 = make_cyclic_jobs(cfg, shard_idx=0)
    j2, s2 = make_cyclic_jobs(cfg, shard_idx=0)
    assert [j["iid"] for j in j1] == [j["iid"] for j in j2]
    assert len(j1) <= 5
    assert s1["accepted"] == len(j1)
    for j in j1:
        assert j["fixed_cpt"]
        inst, pats, shape = realize(
            {"template": j["template"],
             "var_parents": {int(k): tuple(v)
                             for k, v in j["structure"]["var_parents"].items()},
             "r_parents": j["structure"]["r_parents"],
             "fixed_cpt": j["fixed_cpt"]},
            draw_seed=j["draw_seed"])
        assert shape == "cyclic"


def test_signature_stability():
    prop = TEMPLATES["triangle_n3"](seed=1)
    assert signature(prop) == signature(dict(prop))


def test_run_cyclic_instance_end_to_end():
    import importlib.util

    from sheafpatternfusion.cyclic_synth import run_cyclic_instance

    prop = TEMPLATES["triangle_n3"](seed=3)
    inst, pats, shape = realize(prop, draw_seed=999)
    assert shape == "cyclic"
    job = {
        "iid": "cyc_test_000000",
        "n_vars": 3,
        "structure": {
            "var_parents": {str(k): list(v)
                            for k, v in prop["var_parents"].items()},
            "r_parents": [list(p) for p in prop["r_parents"]]},
        "fixed_cpt": prop["fixed_cpt"],
        "draw_seed": 999,
        "template": prop["template"],
        "tag": "cyclic",
    }
    recs = run_cyclic_instance(
        job,
        {"jump_starts": 3, "undecided_round2_multiplier": 1, "fiber_starts": 4},
        ci_draws=1)
    assert recs
    for r in recs:
        assert r["poset_shape"] == "cyclic"
        assert r["gt_recoverable"]
        assert r["sheaf_recoverable"] in ("RECOVERABLE", "UNRECOVERABLE")


def test_discordant_family_member_smoke():
    cfg = {"engine_jump_starts": 2, "engine_jump_starts_r2": 2,
           "member_root_starts": 6, "member_max_roots": 3,
           "member_walk_follows": 0, "member_walk_n_seeds": 1,
           "member_walk_steps": 4}
    assert member_draw_seed(0, cfg) == SEED_ROW["seed"]
    rec = evaluate_member(1, cfg)
    for k in ("witnessed_discordant", "engine_verdict", "model_pair_found",
              "classical_witness", "wall_s"):
        assert k in rec
    assert not rec["is_origin_seed"]
