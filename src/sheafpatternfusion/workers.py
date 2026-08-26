"""Picklable top-level worker entry points for Phase-2.5 Colab notebooks.

Colab runtimes expose ~2 cores; notebooks parallelize with a 2-worker
spawn-Pool over THESE functions (module-level, importable from the embedded
package, dict-in/dict-out so nothing exotic is pickled).
"""
from __future__ import annotations


def run_battery_row(row: dict) -> dict:
    from .battery import score_row

    return score_row(row)


def build_s_star_ids(rows: list[dict], battery_cfg: dict) -> list[str]:
    """Deterministic S* id set (used by audit notebooks to self-derive the
    priority sample without depending on the battery notebook's outputs)."""
    from .battery import aggregate_results, score_row

    undecided_rec = [r for r in rows
                     if r["gt_recoverable"].startswith("UNDETERMINED")
                     and r["sheaf_recoverable"] == "RECOVERABLE"]
    scored = [score_row(r) for r in undecided_rec]
    out = aggregate_results(scored, battery_cfg)
    ids = set()
    for rec in out["priority_sample"]:
        if rec.get("reason", "").startswith("widest_frechet"):
            ids.add(rec["instance_id"] + "|" + __import__("json").dumps(rec["target"]))
    return sorted(ids)


def run_attack_job(row: dict, cfg: dict) -> dict:
    from .attackers import attack_row

    return attack_row(row, cfg)


def run_cyclic_job(job: dict, budgets: dict, ci_draws: int) -> list[dict]:
    from .cyclic_synth import run_cyclic_instance

    return run_cyclic_instance(job, budgets, ci_draws)


def run_family_member(k: int, cfg: dict) -> dict:
    from .discordant_family import evaluate_member

    return evaluate_member(k, cfg)
