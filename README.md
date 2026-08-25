# SheafPatternFusion

Sheaf-theoretic tools for missingness-pattern fusion: enumeration of
missingness structures, LP-based ground truth for marginal completability,
and witness/fiber verdicts for target functional recovery.

## Install

```bash
pip install git+https://github.com/hugogobato/sheafpatternfusion.git@v0.2.0
```

Pinned dependencies: `numpy==2.4.3`, `scipy==1.17.1`, `pyyaml==6.0.1`
(frozen for reproducibility of the Phase 2 grid; see `requirements.txt`).

## Layout

| Path | Contents |
| --- | --- |
| `src/sheafpatternfusion/` | Library package (`engine2`, `enumerate_structures`, `gluing`, `lp_ground_truth`, ...) |
| `configs/` | Frozen experiment grids and example instances |
| `scripts/` | Local runners and shard collection |
| `notebooks_colab/` | Thin self-contained Colab runners, one per shard |
| `tests/` | pytest suite |

## Tests

```bash
pip install -e . pytest==9.0.3 && pytest
```

## Colab usage

Each notebook in `notebooks_colab/` installs the package pinned to the
`v0.2.0` tag, restarts the kernel once so numpy/scipy load with matching
ABI, then runs its shard and writes a JSONL results file.
