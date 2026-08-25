"""WP1.2 ground-truth verification tests.

Mechanical checks required by the research plan:
- LP engine reproduces every expected verdict on the transcription bank.
- Positives are formula-certified at machine precision; negatives get model
  witnesses (and relaxed-LP witnesses where the target is linear).
- Discrete DGP empirical pattern laws match analytic laws.
- Gaussian closed form matches Monte Carlo within 1e-2 on estimand scale.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sheafpatternfusion.bank import load_bank, targets_of, theta_true_of
from sheafpatternfusion.gaussian_ground_truth import (
    contaminated_model,
    mar_null_model,
)
from sheafpatternfusion.lp_ground_truth import decide, lp_range
from sheafpatternfusion.mdag_dgp import MDAG

BANK = load_bank()


# --------------------------------------------------------------------------
# discrete DGP consistency
# --------------------------------------------------------------------------

@pytest.mark.parametrize("iid", sorted(BANK.keys()))
def test_dgp_matches_analytic_observed_laws(iid):
    inst, cfg = BANK[iid]
    n = 300_000
    V, R = inst.sample(n, seed=hash(iid) % (2**31))
    emp = MDAG.empirical_observed_laws(V, R)
    ana = inst.observed_laws()
    for r, cells in ana.items():
        assert r in emp, f"pattern {r} unrealized in sample"
        for o in cells:
            assert abs(emp[r].get(o, 0.0) - cells[o]) < 5e-3, (iid, r, o)


# --------------------------------------------------------------------------
# bank verdicts vs published-label expectations
# --------------------------------------------------------------------------

@pytest.mark.parametrize("iid", sorted(BANK.keys()))
def test_bank_verdicts_match_expected_labels(iid):
    inst, cfg = BANK[iid]
    th = theta_true_of(inst, cfg)
    for tgt, tspec in zip(targets_of(cfg), cfg["targets"]):
        res = decide(inst, th, tgt, tspec.get("formula"), seed=1)
        expect = tspec["expected"]
        if expect == "RECOVERABLE":
            assert res["verdict"] == "RECOVERABLE", (iid, tgt, res)
            assert "formula" in res["evidence"]
        else:
            assert res["verdict"].startswith("UNRECOVERABLE"), (iid, tgt, res)
            if res["verdict"] == "UNRECOVERABLE":
                assert res["witness"]["success"]
                assert res["witness"]["dist"] < 1e-9


def test_negative_witnesses_have_material_effect_size():
    """Unrecoverability witnesses must move the target by a material margin."""
    for iid in ["x4_self_censor_mean", "x5_conditional_dies_with_self_selection", "x8_mediated_mnar_joint"]:
        inst, cfg = BANK[iid]
        th = theta_true_of(inst, cfg)
        for tgt, tspec in zip(targets_of(cfg), cfg["targets"]):
            res = decide(inst, th, tgt, None, seed=2)
            assert res["witness"]["delta_phi"] > 0.05, (iid, tgt)


def test_lp_relaxation_soundness_on_negatives():
    """Assumption-free LP must also separate every negative target."""
    import yaml

    from sheafpatternfusion.bank import load_instance

    exdir = Path(__file__).resolve().parents[1] / "configs" / "examples"
    for iid in ["x4_self_censor_mean", "x5_conditional_dies_with_self_selection", "x8_mediated_mnar_joint"]:
        inst, _ = BANK[iid]
        jt = inst.joint_table()
        q = inst.observed_laws(jt)
        p = next(f for f in exdir.glob("*.yaml")
                 if yaml.safe_load(f.read_text())["instance_id"] == iid)
        _, ycfg = load_instance(p)
        for tgt in targets_of(ycfg):
            if tgt[0] not in ("mean", "cell"):
                continue  # LP relaxation covers linear targets only
            rng = lp_range(inst, q, tgt)
            assert rng["width"] > 0.05, (iid, tgt, rng["width"])


# --------------------------------------------------------------------------
# Gaussian closed form vs Monte Carlo
# --------------------------------------------------------------------------

def test_gaussian_null_mc_check():
    m = mar_null_model(seed=3, n=3)
    rep = m.monte_carlo_check(n=400_000, seed=11)
    assert rep["pass"], rep


def test_gaussian_contaminated_mc_check():
    m = contaminated_model(shift_on=2, delta=1.2, seed=4, n=3)
    rep = m.monte_carlo_check(n=400_000, seed=12)
    assert rep["pass"], rep


def test_gaussian_shift_moves_target_pattern_only():
    m = contaminated_model(shift_on=2, delta=2.0, seed=5, n=3)
    base = mar_null_model(seed=5, n=3)
    changed = [r for r in m.pattern_probs if np.max(np.abs(m.pattern_mean(r) - base.pattern_mean(r))) > 1e-9]
    assert len(changed) >= 1
    for r in changed:
        idx = list(m.observed_set(r))
        assert 2 in idx
