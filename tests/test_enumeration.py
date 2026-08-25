"""Phase 2 infrastructure tests: structure generators, engine2 instruments,
gluing/obstruction LPs, Gaussian PSD certificates."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sheafpatternfusion.engine2 import decide2, formula_oracle, sheaf_fiber_verdict
from sheafpatternfusion.enumerate_structures import (
    all_structures,
    classify,
    conflict_flags,
    discover_slice_cis,
    graham_acyclic,
    instantiate,
    named_structures,
    pick_targets,
    poset_shape,
    r_mechanisms,
    var_dags,
)
from sheafpatternfusion.gluing import (
    canonical_cycle_cases,
    cover_consistent,
    marginal_problem_lp,
    mutually_consistent,
    psd_completion_min_eig,
    scan_poset_discrete,
    slice_marginal,
)
from sheafpatternfusion.lp_ground_truth import pack


# ---------------- generators ----------------

def test_generator_counts_closed_form():
    assert len(var_dags(3)) == 8          # 1 * 2 * 4
    assert len(r_mechanisms(3)) == 512    # 2^3 per indicator, 3 indicators
    assert len(all_structures(3)) == 8 * 512
    assert len(var_dags(2)) == 2
    assert len(all_structures(2)) == 2 * 16


def test_generated_structures_validate_and_realize():
    rng = np.random.default_rng(0)
    structs = all_structures(3)
    for k in rng.integers(0, len(structs), size=25):
        inst = instantiate(structs[int(k)], seed=int(k))
        inst.validate_topological()
        pats = inst.realized_patterns(jt=inst.joint_table())
        assert len(pats) >= 1


def test_named_classes_present():
    named = named_structures()
    vp, rp = named["mutual_selection"]
    assert rp[1] == (1,) and rp[2] == (0,)
    vp, rp = named["double_self_censor"]
    assert rp[0] == (0,) and rp[1] == (1,)


# ---------------- classification / shapes / Graham ----------------

def test_classify_mcar_mar_mnar():
    mcar = instantiate((var_dags(2)[1], ((), ())), seed=1)
    info = classify(mcar)
    assert info["mechanism_class"] == "MCAR"
    mar = instantiate((var_dags(2)[1], ((), (0,))), seed=1)
    info = classify(mar)
    assert info["mechanism_class"] in ("MAR", "MNAR_other")  # depends on realization
    self_c = instantiate((var_dags(2)[1], ((0,), (1,))), seed=1)
    assert classify(self_c)["has_self_edge"]


def test_poset_shape_labels():
    assert poset_shape([(1, 0, 0), (1, 1, 0), (1, 1, 1)]) == "chain"
    assert poset_shape([(1, 1, 0), (1, 0, 1), (0, 1, 1)]) == "cyclic"
    assert poset_shape([(1, 1, 0), (1, 0, 1)]) == "acyclic"


def test_graham_acyclic_matches_intuition():
    chain_sets = [frozenset({0}), frozenset({0, 1}), frozenset({0, 1, 2})]
    tri_sets = [frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})]
    star4 = [frozenset({0, i}) for i in (1, 2, 3)]
    cycle4 = [frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3}),
              frozenset({0, 3})]
    assert graham_acyclic(chain_sets)
    assert graham_acyclic(star4)
    assert not graham_acyclic(tri_sets)
    assert not graham_acyclic(cycle4)


# ---------------- slice-CI discovery ----------------

def test_ci_discovery_finds_independence_in_product_structure():
    inst = instantiate((var_dags(2)[0], ((), ())), seed=5)  # V0,V1 both parentless
    cis = discover_slice_cis(inst, n_draws=10)
    full = max(cis) if isinstance(cis, dict) else None
    assert any(c == ((0,), (1,), ()) or c == ((1,), (0,), ())
               for c in cis[(1, 1)]), cis


def test_ci_discovery_rejects_dependent_chain():
    inst = instantiate((var_dags(2)[1], ((), ())), seed=5)  # V0 -> V1
    cis = discover_slice_cis(inst, n_draws=12)
    assert ((0,), (1,), ()) not in cis.get((1, 1), [])
    assert ((1,), (0,), ()) not in cis.get((1, 1), [])


# ---------------- targets & conflicts ----------------

def test_pick_targets_prefers_partially_observed():
    inst = instantiate((var_dags(2)[1], ((0,), ())), seed=3)
    tgts = pick_targets(inst)
    assert tgts and all(t[0] == "mean" for t in tgts)


def test_conflict_flags_on_known_instances():
    from sheafpatternfusion.bank import load_bank
    bank = load_bank()
    mcar_info = conflict_flags(bank["x1_mcar_joint"][0])
    mnar_info = conflict_flags(bank["x4_self_censor_mean"][0])
    assert not mcar_info["conflict_mcar_style"]
    assert mnar_info["conflict_mcar_style"]


# ---------------- engine2 ----------------

def test_decide2_recovers_bank_positives_and_negatives():
    from sheafpatternfusion.bank import load_bank, load_instance, targets_of

    import yaml
    exdir = Path(__file__).resolve().parents[1] / "configs" / "examples"
    bank = load_bank()
    checks = {
        "x1_mcar_joint": "RECOVERABLE",
        "x6_anchor_under_mnar": "RECOVERABLE",
        "x4_self_censor_mean": "UNRECOVERABLE",
        "x8_mediated_mnar_joint": "UNRECOVERABLE",
    }
    for iid, expect in checks.items():
        inst, cfg = bank[iid]
        tgt = targets_of(cfg)[0]
        res = decide2(inst, pack(inst), tgt, seed=1)
        assert res["gt_verdict"] == expect, (iid, res)


def test_sheaf_fiber_verdict_agrees_with_engine_on_bank():
    from sheafpatternfusion.bank import load_bank, targets_of

    bank = load_bank()
    pairs = [("x1_mcar_joint", "RECOVERABLE"),
             ("x4_self_censor_mean", "UNRECOVERABLE")]
    for iid, expect in pairs:
        inst, cfg = bank[iid]
        res = sheaf_fiber_verdict(inst, pack(inst), targets_of(cfg)[0],
                                  n_starts=24, seed=7)
        assert res["sheaf_verdict"] == expect, (iid, res)


# ---------------- gluing LP ----------------

def test_family_from_joint_is_feasible_and_consistent():
    rng = np.random.default_rng(4)
    cells = list(np.ndindex(2, 2, 2))
    T = dict(zip(cells, map(float, rng.dirichlet(np.ones(8)))))
    fam = {(1, 1, 0): slice_marginal(T, (1, 1, 0)),
           (1, 0, 1): slice_marginal(T, (1, 0, 1)),
           (0, 1, 1): slice_marginal(T, (0, 1, 1))}
    assert marginal_problem_lp(3, fam)["feasible"]
    assert mutually_consistent(fam)


def test_inconsistent_family_is_infeasible():
    fam = {(1, 1, 0): {(1, 1): 0.3, (1, 0): 0.2, (0, 1): 0.1, (0, 0): 0.4},
           (1, 0, 1): {(1, 1): 0.05, (1, 0): 0.15, (0, 1): 0.35, (0, 0): 0.45}}
    # V1 marginal disagrees: 0.5 vs 0.2
    assert not mutually_consistent(fam)
    assert not marginal_problem_lp(3, fam)["feasible"]


def test_triangle_binary_has_genuine_obstructions():
    """Mutually consistent pair families on the triangle that admit no joint:
    certified by LP, cross-checked against a nonlinear completion attempt."""
    res = scan_poset_discrete([(1, 1, 0), (1, 0, 1), (0, 1, 1)],
                              n_families=60, seed=11)
    assert res["n_obstructed"] > 0
    assert res["n_globally_feasible"] > 0  # mixed regime, not uniform


def test_forest_pair_posets_never_obstruct():
    res = scan_poset_discrete([(1, 1, 0), (1, 0, 1)], n_families=40, seed=2)
    assert res["n_obstructed"] == 0
    assert res["n_globally_feasible"] == 40


# ---------------- Gaussian PSD certificates ----------------

def test_canonical_gaussian_cases_match_verified_expectations():
    for case in canonical_cycle_cases():
        assert case["matches_expectation"], case["name"]


def test_psd_completion_control_completable_with_explicit_matrix():
    res = psd_completion_min_eig(3, {(0, 1): 0.5, (0, 2): 0.5, (1, 2): 0.5})
    assert res["completable"]
    C = np.array(res["certificate_matrix"])
    assert np.allclose(C, C.T)
    assert np.min(np.linalg.eigvalsh(C)) > 0
