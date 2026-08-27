# Phase 3b - remaining scaling work (continuation fleet)

**Generated 2026-08-26 from the live checkpoint snapshot in `../phase3/scaling_probe_shard*.jsonl`.**
All notebooks are self-contained, pinned (`numpy==2.4.3` / `scipy==1.17.1`), and resume-safe - re-running `Run all` skips already-written rows.

## Snapshot

- Planned: 216 n=5 structures (18 per shard x 12 shards, each structure -> 2 engine rows for `['mean',0]` / `['mean',1]`)
- Done: 108 structures (312 engine rows already on file; `n4t` re-timing is complete 8/8 per shard)
- Remaining n=5: 108 structures -> 216 rows, distributed unevenly (shard01: 2 left, shard00: 12 left, etc.)
- `n6` pilot and attacks not yet run for any shard (gated on n=5 completion)

Worst observed per-job wall (both targets summed) is ~6.0k s (shard09). Original 12-shard notebooks hit the 10 h limit. This continuation slices the missing iids so each notebook stays **<4 h** on 2 Colab workers (hard wall `SOFT = 14400 s`).

## Fleet (45 notebooks)

### Fleet R - engine resume (33 notebooks, median ~0.9-1.5 h, worst ~3.4 h)

| notebook | shard | slice iids | wall |
|---|---|---|---|
| `nb30_b_resume_n5_s00_p0..p2` | 00 | 12 missing -> 3 slices of 4 | ~0.9-1.1 h median |
| `nb30_b_resume_n5_s01_p0` | 01 | 2 missing -> 1 slice | ~0.4 h |
| `nb30_b_resume_n5_s02_p0..p2` | 02 | 10 -> 3 slices 4,4,2 | ~1.0-1.2 h |
| `nb30_b_resume_n5_s03_p0..p1` | 03 | 7 -> 2 slices 4,3 | ~0.7-1.0 h |
| `nb30_b_resume_n5_s04_p0..p2` | 04 | 9 -> 3 slices 4,4,1 | ~0.8-1.2 h |
| `nb30_b_resume_n5_s05_p0..p2` | 05 | 11 -> 3 slices 4,4,3 | ~1.0-1.3 h |
| `nb30_b_resume_n5_s06_p0..p2` | 06 | 9 -> 3 slices | ~1.0 h |
| `nb30_b_resume_n5_s07_p0..p1` | 07 | 8 -> 2 slices 4,4 | ~1.0 h |
| `nb30_b_resume_n5_s08_p0..p3` | 08 | 11 -> 4 slices of 3 (max 5657 s -> 3.4 h worst) | ~1.1 h |
| `nb30_b_resume_n5_s09_p0..p3` | 09 | 11 -> 4 slices of 3 (max 6052 s -> 3.4 h worst) | ~1.1 h |
| `nb30_b_resume_n5_s10_p0..p2` | 10 | 10 -> 3 slices | ~1.0 h |
| `nb30_b_resume_n5_s11_p0..p1` | 11 | 8 -> 2 slices | ~1.0 h |

Each resume notebook appends only to `scaling_probe_shardXX.jsonl` for its shard (same `/content/results/phase3` path as the original fleet). Slices are disjoint iid sets, so all 33 can run in parallel. Already-written rows are skipped via `load_done`.

### Fleet F - finish (12 notebooks, <1 h each)

`nb30_b_finish_shard00..11.ipynb` - run **after** the corresponding shard's resume slices report `18/18 n5 structures`. Each:

1. Checks `n5` completeness; if <18/18 it exits cleanly with "run the resume slices first".
2. Runs the single trailing `n6` pilot (1 structure, ~0.4-0.8 h if not already gated).
3. Runs the quota-6 attacks on undecided x RECOVERABLE n=5 rows (seeded SRS, 2 workers, ~0.3-0.6 h).
4. Rewrites `scaling_summary_shardXX.json`.

If you prefer, you can also re-run the original `nb30_b_scaling_shard_XX.ipynb` as the finish step (same logic, 8 h soft wall). The dedicated finish notebooks are tighter (4 h hard wall) and named distinctly.

## How to run

1. **Fleet R first (fully parallel):** upload all 33 `nb30_b_resume_n5_*.ipynb` to Colab and `Run all` (restart once for the numpy/scipy pin, then Run all again). Each downloads its `scaling_probe_shardXX.jsonl` append.
2. **Fleet F next (per-shard, after its R slices land):** upload and run `nb30_b_finish_shardXX.ipynb` for each shard that has reached 18/18. These download `scaling_attacks_shardXX.jsonl` and `scaling_summary_shardXX.json`.
3. **Collect locally:** drop every downloaded `*.jsonl` / `*.json` into `notebooks_colab/phase3/incoming/` (or overwrite the files in `notebooks_colab/phase3/`) and run `python3 scripts/run_phase3.py --stage collect --stage gate` to produce `results/phase3/gate_G26_memo.md`.

All 45 notebooks were validated (`compile` all code cells) and the lib was syntax-checked at generation.

## Why this slicing guarantees <4 h

Per-job wall (both targets, wall_struct once + witness_r1+r2+fiber summed) median ~1.6k s, worst ~6.0k s. A slice of 4 jobs: wall ≈ pilot (sequential) + (k-1) jobs on 2 workers. Worst with k=4 and max job on shard09: `6052 + 3*6052/2 = 15130 s = 4.2 h` - hence shards 08/09 use k=3 (worst `3*~5.6k/2 + pilot ≈ 3.4 h`). All other shards: k=4 worst <4 h, median 0.9-1.5 h. Finish shards: 1 n6 + 6 attacks ≈ <1 h. Hard wall 14400 s enforces it.

## Other WP3 assets

- `nb30_a_prevalence` and `nb30_c_cycattack` fleets are complete (4x196 cyclic verdicts, prevalence scan on file). `nb30_c_signal_analysis` is partial (primary AUC single-class - needs the completed scaling attacks to become meaningful). No action needed there until scaling finishes.
