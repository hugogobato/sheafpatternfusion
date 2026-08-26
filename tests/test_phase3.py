"""Phase 3 probe tests (WP3.0a/b/c utilities)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sheafpatternfusion.attackers import attack_row
from sheafpatternfusion.battery import instance_from_row
from sheafpatternfusion.engine2 import decide2
from sheafpatternfusion.enumerate_structures import instantiate, poset_shape
from sheafpatternfusion.lp_ground_truth import pack, target_value_phi
from sheafpatternfusion.phase3_probe import (
    attack_row_fixed,
    column_permutation_control,
    compact_engine_row,
    compress_payload,
    cyclic_fraction_bootstrap,
    decide2_timed,
    decompress_payload,
    instance_from_row_fixed,
    naive_pooling_mean,
    permutation_auc_p,
    rank_auc,
    realized_pattern_counts,
    run_scaling_job,
    sample_structures,
    scan_subsets,
    spread_naive_table,
)


LIGHT_BUDGETS = {
    "jump_starts": 6,
    "round2_multiplier": 1,
    "fiber_starts": 8,
    "max_roots": 4,
    "ci_discovery_draws": 3,
    "lp_pinch_tol": 1e-9,
    "lp_width_tol": 1e-3,
}


def _job(n_vars=3):
    return {
        "iid": "t_job", "n_vars": n_vars,
        "structure": {"var_parents": {"0": [], "1": [0], "2": [0]},
                      "r_parents": [[], [0], [1, 2]]},
        "draw_seed": 20270901,
    }


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def test_sample_structures_deterministic_valid_distinct():
    a = sample_structures(5, 12, seed=20400901, prefix="n5")
    b = sample_structures(5, 12, seed=20400901, prefix="n5")
    assert a == b and len(a) == 12
    seen = set()
    for j in a:
        vp = {int(k): tuple(v) for k, v in j["structure"]["var_parents"].items()}
        assert all(all(p < i for p in pa) for i, pa in vp.items())
        rp = tuple(tuple(p) for p in j["structure"]["r_parents"])
        assert len(rp) == 5
        assert all(set(p) <= set(range(5)) for p in rp)
        key = (tuple(sorted(vp.items())), rp)
        assert key not in seen
        seen.add(key)
        assert isinstance(inst_ok(vp, rp, j["draw_seed"]), object)
        # instantiate twice -> identical CPTs (determinism of draws)
        i1 = instantiate((vp, rp), seed=j["draw_seed"])
        i2 = instantiate((vp, rp), seed=j["draw_seed"])
        assert pack(i1).tolist() == pack(i2).tolist()


def inst_ok(vp, rp, seed):
    return instantiate((vp, rp), seed=seed)


# --------------------------------------------------------------------------
# engine parity (cyclic-stratum protocol: round1 40/seed11, round2 x2/seed23)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rp", [
    [[], [0], [1, 2]],
    [[1], [0, 2], [0, 1]],
    [[], [0], [0]],
])
def test_decide2_timed_matches_two_round_protocol(rp):
    structure = ({0: (), 1: (0,), 2: (0,)}, tuple(tuple(p) for p in rp))
    inst = instantiate(structure, seed=20270901)
    theta = pack(inst)
    from sheafpatternfusion.enumerate_structures import pick_targets
    for tgt in pick_targets(inst):
        ref = decide2(inst, theta, tgt, seed=11)
        if ref["gt_verdict"].startswith("UNDETERMINED"):
            ref = decide2(inst, theta, tgt, jump_starts=80, seed=23)
            ref["gt_evidence"] = "round2:" + ref["gt_evidence"]
        got = decide2_timed(inst, theta, tgt, jump_starts=40,
                            round2_multiplier=2, seed=11)
        assert got["gt_verdict"] == ref["gt_verdict"], (rp, tgt)
        assert got["gt_evidence"] == ref["gt_evidence"], (rp, tgt)
        assert "walls" in got


def test_run_scaling_job_schema_and_walls():
    job = _job()
    job["do_attack"] = False
    recs = run_scaling_job(job, {"budgets": LIGHT_BUDGETS})
    assert len(recs) >= 1
    need = {"instance_id", "gt_recoverable", "sheaf_recoverable",
            "phi_spread_over_fiber", "jacobian_rank_deficiency",
            "frechet_width", "frac_observed", "overlap_density",
            "max_cross_pattern_marginal_gap", "wall_engine_r1_s",
            "wall_fiber_s", "wall_features_s", "attack"}
    for r in recs:
        assert need <= set(r)
        assert r["frechet_width"] is None or r["frechet_width"] >= -1e-9
        assert r["attack"] is None
        json.dumps(compact_engine_row(r))


def test_run_scaling_job_attack_path_honors_quota_flag():
    job = _job()
    job["do_attack"] = True
    job["iid"] = "t_job_att"
    cfg = {"budgets": LIGHT_BUDGETS,
           "attack": {"a1_jump_rounds": 1, "a1_starts_per_round": 4,
                      "a1_walk_n_seeds": 1, "a1_walk_steps": 5,
                      "a2_root_starts": 4, "a2_max_roots": 2,
                      "a2_walk_follows": 0, "a2_lp_vertices": 2,
                      "a3_max_cells": 8, "phi_tol": 1e-4, "dist_tol": 1e-9}}
    recs = run_scaling_job(job, cfg)
    for r in recs:
        if r["attack_requested"]:
            assert r["attack"] is not None
            assert r["attack"]["verdict"] in (
                "CONFIRMED_FALSE_RECOVERABLE", "NO_FALSE_RECOVERABLE_FOUND")
        else:
            assert r["attack"] is None or r["attack"].get("status") == "error"


# --------------------------------------------------------------------------
# pin-aware reconstruction
# --------------------------------------------------------------------------

def test_instance_from_row_fixed_honors_pins():
    from sheafpatternfusion.cyclic_synth import realize, triangle_template
    prop = triangle_template(12345)
    draw_seed = 987654321
    inst, patterns, shape = realize(prop, draw_seed)
    assert shape == "cyclic"
    row = {
        "instance_id": "cyc_test", "seed": draw_seed,
        "var_parents": {str(k): list(v) for k, v in prop["var_parents"].items()},
        "r_parents": [list(p) for p in prop["r_parents"]],
        "fixed_cpt": prop["fixed_cpt"],
        "target": ["mean", 0],
    }
    _, q_fixed, pp_fixed = instance_from_row_fixed(row)
    m_ref = inst
    jt = m_ref.joint_table()
    q_ref = m_ref.observed_laws(jt)
    pp_ref = {}
    for (v, r), p in jt.items():
        pp_ref[r] = pp_ref.get(r, 0.0) + p
    assert sorted(q_fixed.keys()) == sorted(q_ref.keys())
    for r in q_ref:
        assert max(abs(q_fixed[r].get(o, 0.0) - c)
                   for o, c in q_ref[r].items()) < 1e-12
    # ignoring pins realizes the WRONG family (full simplex), so the plain
    # battery builder must disagree here
    _, q_plain, _ = instance_from_row(dict(row))
    assert sorted(q_plain.keys()) != sorted(q_ref.keys())
    assert len(q_plain) > len(q_ref)


def test_attack_row_fixed_agrees_with_unpinned_path_on_unpinned_row():
    merge = Path(__file__).resolve().parents[1] / "data" / "frozen" / "instances_merged.jsonl"
    rows = [json.loads(l) for l in open(merge)]
    row = dict(next(r for r in rows if r["n_vars"] == 2))
    assert not row.get("fixed_cpt")
    row["fixed_cpt"] = None
    cfg = {"a1_jump_rounds": 1, "a1_starts_per_round": 4,
           "a1_walk_n_seeds": 1, "a1_walk_steps": 5,
           "a2_root_starts": 4, "a2_max_roots": 2, "a2_walk_follows": 0,
           "a2_lp_vertices": 2, "a3_max_cells": 8,
           "phi_tol": 1e-4, "dist_tol": 1e-9}
    a = attack_row(dict(row), cfg)
    b = attack_row_fixed(dict(row), cfg)
    assert a["verdict"] == b["verdict"]
    assert a["A1"]["best_delta_phi"] == pytest.approx(b["A1"]["best_delta_phi"])
    assert a["A2"]["n_roots"] == b["A2"]["n_roots"]


# --------------------------------------------------------------------------
# prevalence utilities
# --------------------------------------------------------------------------

def test_realized_pattern_counts_basic():
    obs = np.array([[True, True], [True, False], [False, True]])
    counts = realized_pattern_counts(obs)
    assert counts == {(1, 1): 1, (1, 0): 1, (0, 1): 1}


def test_scan_subsets_triangle_vs_chain():
    # 4 realized patterns; three pairwise observed-set edges form a Berge cycle
    obs = np.array([
        [False, False, False],
        [True, True, False],
        [True, False, True],
        [False, True, True],
    ], dtype=bool)
    names = ["a", "b", "c"]
    rec = scan_subsets(obs, names, [(0, 1, 2)], min_patterns=4)[0]
    assert rec["eligible"] and rec["cyclic"] and not rec["nested_only"]
    # nested chain: acyclic
    obs2 = np.array([
        [False, False, False],
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ], dtype=bool)
    rec2 = scan_subsets(obs2, names, [(0, 1, 2)], min_patterns=4)[0]
    assert rec2["eligible"] and not rec2["cyclic"] and rec2["nested_only"]
    # min_support robustness kills rare-pattern eligibility
    rec3 = scan_subsets(obs, names, [(0, 1, 2)], min_patterns=4,
                        min_support=2)[0]
    assert not rec3["eligible"]


def test_column_permutation_control_preserves_marginals():
    rng = np.random.default_rng(0)
    obs = rng.random((500, 4)) < 0.3
    obs[:, 1] |= obs[:, 0]  # induce co-missingness dependence
    perm = column_permutation_control(obs, np.random.default_rng(7))
    assert np.array_equal(np.sort(perm[:, 0]), np.sort(obs[:, 0]))
    for j in range(4):
        assert perm[:, j].mean() == pytest.approx(obs[:, j].mean())


def test_cyclic_fraction_bootstrap_small():
    obs = np.array([
        [False, False, False], [True, True, False],
        [True, False, True], [False, True, True],
    ], dtype=bool)
    out = cyclic_fraction_bootstrap(obs, [(0, 1, 2)], B=5, min_patterns=4,
                                    min_support=1, seed=1)
    assert out["B"] == 5
    assert 0.0 <= out["fraction_median"] <= 1.0


# --------------------------------------------------------------------------
# signal-validity utilities
# --------------------------------------------------------------------------

def test_rank_auc_ties_and_degenerate():
    assert rank_auc([0.1, 0.9], [0, 1]) == 1.0
    assert rank_auc([0.9, 0.1], [0, 1]) == 0.0
    assert rank_auc([0.5, 0.5, 0.2], [1, 1, 0]) == pytest.approx(1.0)
    # a positive tied with the negative splits that pair: AUC 0.5
    assert rank_auc([0.5, 0.5, 0.2], [1, 0, 0]) == pytest.approx(0.75)
    assert rank_auc([1.0, 2.0], [1, 1]) is None


def test_permutation_auc_p_single_class_and_signal():
    rng = np.random.default_rng(3)
    y = np.zeros(60, dtype=int)
    res = permutation_auc_p(rng.random(60), y, np.ones(60, int), B=50, seed=1)
    assert res["auc"] is None
    scores = np.concatenate([rng.normal(1.0, 0.3, 30), rng.normal(0.0, 0.3, 30)])
    labels = np.array([1] * 30 + [0] * 30)
    # strata must CUT ACROSS classes (here: index parity), else the
    # within-stratum permutation cannot mix labels and p -> 1 trivially
    strata = np.array(["a", "b"] * 30)
    res2 = permutation_auc_p(scores, labels, strata, B=200, seed=5)
    assert res2["auc"] > 0.85
    assert res2["p_value"] < 0.05


def test_naive_pooling_mean_exact():
    structure = ({0: (), 1: (0,), 2: (0,)}, ((), (0,), (1, 2)))
    m = instantiate(structure, seed=20270901)
    est = naive_pooling_mean(m, 2)
    jt = m.joint_table()
    q = m.observed_laws(jt)
    pp = {}
    for (v, r), p in jt.items():
        pp[r] = pp.get(r, 0.0) + p
    num = den = 0.0
    for r in sorted(q.keys()):
        if r[2] != 1:
            continue
        pos = sum(1 for k in range(3) if r[k] == 1 and k < 2)
        mg: dict[int, float] = {}
        for o, c in q[r].items():
            mg[o[pos]] = mg.get(o[pos], 0.0) + c
        num += pp[r] * sum(k * c for k, c in mg.items())
        den += pp[r]
    assert den > 0
    assert est == pytest.approx(num / den)
    # sanity on a fully observed pattern: pooling over observers of var j
    # stays within [0, 1]
    assert 0.0 <= est <= 1.0


def test_spread_naive_table_rebuilds_and_correlates_fields():
    merge = Path(__file__).resolve().parents[1] / "data" / "frozen" / "instances_merged.jsonl"
    rows = [json.loads(l) for l in open(merge)]
    dec = [r for r in rows if r["gt_recoverable"] == "UNRECOVERABLE"
           and r["n_vars"] <= 3][:6]
    for r in dec:
        r["source_tag"] = "merge"
    tab = spread_naive_table(dec)
    assert len(tab) == len(dec)
    for t in tab:
        assert np.isfinite(t["naive_abs_err"]) and t["naive_abs_err"] >= 0
        assert 0.0 <= t["spread"] <= 1.0


# --------------------------------------------------------------------------
# payload codec
# --------------------------------------------------------------------------

def test_payload_codec_roundtrip():
    obj = [{"a": 1, "b": [1, 2, 3]}, {"a": 2, "b": None}]
    b64 = compress_payload(obj)
    assert decompress_payload(b64) == obj


def test_poset_shape_of_sampled_n5_structures_runs():
    jobs = sample_structures(5, 3, seed=99, prefix="x")
    for j in jobs[:1]:
        vp = {int(k): tuple(v) for k, v in j["structure"]["var_parents"].items()}
        rp = tuple(tuple(p) for p in j["structure"]["r_parents"])
        inst = instantiate((vp, rp), seed=j["draw_seed"])
        pats = inst.realized_patterns(jt=inst.joint_table())
        assert poset_shape(pats) in ("chain", "acyclic", "cyclic")
