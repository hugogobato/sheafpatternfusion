"""Load the Phase-1 transcription bank (configs/examples/*.yaml) into MDAG
instances with seeded ground-truth parameters."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import yaml

from .mdag_dgp import MDAG

BANK_DIR = Path(__file__).resolve().parents[2] / "configs" / "examples"


def _key(parents: tuple[int, ...], values: tuple[int, ...]) -> tuple:
    return values if parents else ()


def load_instance(path: Path) -> tuple[MDAG, dict]:
    cfg = yaml.safe_load(path.read_text())
    n = cfg["n_vars"]
    var_parents = {int(k): tuple(v) for k, v in cfg["var_parents"].items()}
    r_parents = {int(k): tuple(v) for k, v in cfg["r_parents"].items()}
    inst = MDAG(n_vars=n, var_parents=var_parents, r_parents=r_parents)
    inst.validate_topological()
    rng = np.random.default_rng(cfg["seed"])
    inst.random_fill(rng)
    for fx in cfg.get("fixed_cpt", []):
        pa = tuple(fx["parents"])
        table = inst.r_cpt if fx["kind"] == "r" else inst.var_cpt
        table[fx["node"]][pa] = float(fx["p"])
    return inst, cfg


def load_bank(bank_dir: Path | None = None) -> dict[str, tuple[MDAG, dict]]:
    d = bank_dir or BANK_DIR
    out = {}
    for p in sorted(d.glob("*.yaml")):
        inst, cfg = load_instance(p)
        out[cfg["instance_id"]] = (inst, cfg)
    return out


def targets_of(cfg: dict) -> list[tuple]:
    out = []
    for t in cfg["targets"]:
        if t["type"] == "mean":
            out.append(("mean", t["index"]))
        elif t["type"] == "cell":
            out.append(("cell", tuple(t["cell"])))
        elif t["type"] == "cond":
            out.append(("cond", t["y_index"], t["y_value"],
                        tuple(t["x_indices"]), tuple(t["x_values"])))
        else:
            raise ValueError(t["type"])
    return out


def theta_true_of(inst: MDAG, cfg: dict, seed_offset: int = 0) -> np.ndarray:
    """Seeded true CPT vector for the instance (deterministic per config)."""
    from .lp_ground_truth import pack

    return pack(inst)
