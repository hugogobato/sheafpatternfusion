# Phase 2.5 Colab notebooks (pinned to tag `v0.3.0`)

Twelve self-contained runners for the WP2.5.1-WP2.5.4 experiments (research
plan, Phase 2.5). Each notebook embeds the full package source, its config,
and a resumable JSONL checkpoint loop; nothing depends on another notebook's
runtime state or outputs. Upload each to its own Colab account/runtime
(CPU runtime is sufficient; ~2 cores, ~12 GB RAM).

## Roster and expected wall times

| Notebook | WP | Purpose | Expected |
|---|---|---|---|
| `nb25_00_nullbattery.ipynb` | 2.5.1 | Nulls N0-N4 vs certificate labels on all engine-undecided rows; emits S* priority sample | minutes |
| `nb25_audit_shard_00..05.ipynb` | 2.5.2 | Adversarial audit (A1 deepened witness / A2 completion enumeration / A3 Frechet cells) of undecided x RECOVERABLE assertions; census n=2 + n=4, SRS N=400 at n=3 (seed 20250901), census S* recomputed in-notebook | <= 10 h each |
| `nb25_cyclic_shard_00..03.ipynb` | 2.5.4 | Forced cyclic-poset stratum: generate >= 140 instances/shard with cyclic realized overlap hypergraphs, run the identical Phase-2 pipeline | ~1 h each |
| `nb25_family.ipynb` | 2.5.3 | Grow the n3_s03759_d0 discordant seed into a family over fresh mechanism draws; gate G2.5d needs >= 10 witnessed members | < 6 h |

## Before you upload

The battery and audit notebooks fetch their sampling frame from
`https://raw.githubusercontent.com/hugogobato/sheafpatternfusion/v0.3.0/data/frozen/instances_merged.jsonl`.
That tag must exist on GitHub first (it is pushed together with these
notebooks). If a fetch fails, manually upload `data/frozen/instances_merged.jsonl`
to `/content/` and re-run the fetch cell.

## Outputs (auto-downloaded at the end of each notebook)

- battery: `null_battery_scored.jsonl`, `null_battery.json`, `priority_sample.jsonl`
- audit shards: `audit_sample.jsonl`, `audit_verdicts_shard_XX.jsonl`
- cyclic shards: `cyclic_instances_shardXX.jsonl`, `cyclic_gen_stats_shardXX.json`
- family: `discordant_family.jsonl`, `family_summary.json`

Drop downloaded files into `results/phase25/downloads/` locally;
`scripts/run_phase25.py --stage collect` style merging and the gate analysis
happen afterwards.

## Behavior notes

- Every notebook prints a **self-pilot projection** (first job at pilot
  budgets) before committing to the full run; watch that line.
- All loops resume by JSONL key dedup: re-running a notebook continues where
  it stopped.
- Kill rules are pre-registered: any `CONFIRMED_FALSE_RECOVERABLE` verdict in
  an audit shard kills C1 (rule D2); family gate G2.5d passes at >= 10
  witnessed members; cyclic stratum gates per D4.
- Budgets live in `configs/phase25/*.json` and are embedded per notebook.
  The audit's "x20" escalation is implemented as an adaptive ceiling
  (`a1_adaptive_stop`); see the pricing note in
  `configs/phase25/audit.json`.
- Regenerate all twelve after library edits with:
  `python3 scripts/make_colab_phase25.py`
