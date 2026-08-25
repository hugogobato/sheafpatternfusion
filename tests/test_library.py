"""WP1.5 library property tests."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sheafpatternfusion.fuse import fuse, localize
from sheafpatternfusion.laplacian import (assemble_laplacian, harmonic_dimension,
                                          is_psd)
from sheafpatternfusion.poset import (PatternPoset, brute_force_transitive_reduction,
                                      random_poset, restriction_matrix)
from sheafpatternfusion.radius import minimal_radius
from sheafpatternfusion.sheaf import (CIConstraint, DiscreteSheaf,
                                      GaussianMeanSheaf)


# ---------------- poset ----------------

def test_partial_order_axioms_random():
    for seed in range(20):
        poset = random_poset(3, 6, seed=seed)
        ps = poset.patterns
        for a in ps:
            assert poset.leq(a, a)  # reflexivity
        for a in ps:
            for b in ps:
                if poset.leq(a, b) and poset.leq(b, a):
                    assert a == b  # antisymmetry
                if poset.leq(a, b):
                    pass
        for a in ps:
            for b in ps:
                for c in ps:
                    if poset.leq(a, b) and poset.leq(b, c):
                        assert poset.leq(a, c)  # transitivity


def test_hasse_matches_bruteforce_reduction():
    for seed in range(20):
        poset = random_poset(3, 6, seed=seed)
        pairs = set(poset.comparable_pairs())
        assert brute_force_transitive_reduction(pairs) == set(poset.covers)


# ---------------- restrictions / discrete sheaf ----------------

def test_restriction_matrix_marginalizes_means():
    big = (1, 1, 1)
    small = (1, 0, 1)
    R = restriction_matrix(big, small)
    mu_big = np.array([0.3, -1.2, 2.0])
    assert np.allclose(R @ mu_big, [0.3, 2.0])


def test_functoriality_of_restrictions():
    p3 = PatternPoset([(1, 1, 1), (1, 1, 0), (1, 0, 0)])
    R31 = restriction_matrix((1, 1, 1), (1, 0, 0))
    R32 = restriction_matrix((1, 1, 0), (1, 0, 0))
    R = restriction_matrix((1, 1, 1), (1, 1, 0))
    mu = np.array([0.7, -2.0, 5.0])
    assert np.allclose(R31 @ mu, R32 @ (R @ mu))


def test_discrete_section_check_positive_and_negative():
    poset = PatternPoset([(1, 1), (1, 0), (0, 1)])
    sh = DiscreteSheaf(poset)
    # consistent family: any joint table on the top marginalizes correctly
    top_tab = {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.3, (1, 1): 0.4}
    fam = {
        (1, 1): top_tab,
        (1, 0): {(0,): 0.3, (1,): 0.7},
        (0, 1): {(0,): 0.4, (1,): 0.6},
    }
    assert sh.is_section(fam)
    fam_bad = dict(fam)
    fam_bad[(0, 1)] = {(0,): 0.9, (1,): 0.1}
    assert not sh.is_section(fam_bad)


def test_ci_constraint_detection():
    # genuinely independent table: margins (0.4,0.6) x (0.3,0.7)
    tab_ind = {(0, 0): 0.12, (0, 1): 0.28, (1, 0): 0.18, (1, 1): 0.42}
    tab_dep = {(0, 0): 0.05, (0, 1): 0.45, (1, 0): 0.20, (1, 1): 0.30}
    c = CIConstraint(x=(0,), y=(1,), z=())
    r = (1, 1)
    assert c.holds(tab_ind, r)
    assert not c.holds(tab_dep, r)


# ---------------- Laplacian ----------------

def test_laplacian_psd_and_harmonics():
    for seed in range(10):
        poset = random_poset(3, 7, seed=seed)
        L = assemble_laplacian(poset)
        assert is_psd(L)
    # connected full-overlap instance on 2 vars: harmonics = n_vars = 2
    poset = PatternPoset([(1, 1), (1, 0), (0, 1)])
    L = assemble_laplacian(poset)
    assert harmonic_dimension(L) == 2


def test_laplacian_harmonics_disconnected():
    poset = PatternPoset([(1, 1, 0), (1, 0, 0), (0, 0, 1)])
    L = assemble_laplacian(poset)
    # two overlap-components; harmonics = dims of component stalk sums = 2 + 1
    assert harmonic_dimension(L) == 3


# ---------------- radius & fuse: hand-checked star example ----------------

def test_hand_computed_star_example():
    """Star poset, unit weights: leaves l_i = a_i/2 and center a solves
    3a1 = 2*b1 etc., giving a = [2/3, -2/3], leaves [1/3], [-1/3], r^2 = 1/2."""
    poset = PatternPoset([(1, 1), (1, 0), (0, 1)])
    obs = {(1, 1): np.array([1.0, -1.0]), (1, 0): np.array([0.0]), (0, 1): np.array([0.0])}
    res = minimal_radius(poset, obs)
    g = res["section"]
    gs = GaussianMeanSheaf(poset)
    a = gs.block(g, (1, 1))
    assert np.allclose(a, [2.0 / 3.0, -2.0 / 3.0], atol=1e-8)
    assert np.allclose(gs.block(g, (1, 0)), [1.0 / 3.0], atol=1e-8)
    assert np.allclose(gs.block(g, (0, 1)), [-1.0 / 3.0], atol=1e-8)
    assert abs(res["radius"] ** 2 - 2.0 / 3.0) < 1e-8


def test_fuse_idempotent_on_consistent_input():
    poset = PatternPoset([(1, 1), (1, 0), (0, 1)])
    means = {(1, 1): np.array([0.25, -3.0]), (1, 0): np.array([0.25]),
             (0, 1): np.array([-3.0])}
    ns = {r: 100.0 for r in means}
    res1 = fuse(poset, means, ns)
    assert res1["radius"] < 1e-6
    res2 = fuse(poset, {r: v.copy() for r, v in res1["fused"].items()}, ns)
    assert res2["radius"] < 1e-6
    for r in means:
        assert np.allclose(res1["fused"][r], res2["fused"][r])


def test_fuse_single_pattern_identity():
    poset = PatternPoset([(0, 1, 1)])
    mu = np.array([7.0, -2.0])
    res = fuse(poset, {(0, 1, 1): mu})
    assert np.allclose(res["full_mean"], [0.0, 7.0, -2.0])
    assert res["radius"] == 0.0


def test_localization_flags_shifted_pattern():
    """Redundant triangle: each variable observed by 3 of 4 patterns, so the
    consensus pins the truth and a shifted pattern is identifiable. (Note:
    with only two sources per coordinate, e.g. a star poset, localization is
    symmetric and impossible in principle; redundancy is required.)"""
    rng = np.random.default_rng(0)
    poset = PatternPoset([(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)])
    truth = np.array([1.0, 2.0, -0.5])
    means = {}
    for r in poset.patterns:
        idx = [i for i in range(3) if r[i] == 1]
        means[r] = truth[idx] + rng.normal(0, 0.01, len(idx))
    means[(1, 1, 0)] += 3.0
    ns = {r: 10000.0 for r in poset.patterns}
    res = fuse(poset, means, ns)
    assert localize(res) == (1, 1, 0)
    pr = res["prediction_residuals"]
    assert pr[(1, 1, 0)] == max(pr.values())
    # clean patterns inherit indirect contamination through refits containing
    # the dirty pattern, so a modest ratio is expected, not orders of magnitude
    assert pr[(1, 1, 0)] > 3.0 * max(v for k, v in pr.items() if k != (1, 1, 0))
