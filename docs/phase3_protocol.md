# Phase 3 runbook (WP3.0 pivot-gate week, G2.6)

**Status (2026-08-26): fleet of 18 self-contained CPU notebooks generated;
collection is automatic via `scripts/run_phase3.py --stage collect`.**

Sections 7, 11, and 13 of the Research Plan govern this week. The three
probes are decisive: WP3.0a checks whether cyclic overlap structure exists
anywhere practitioners actually are (otherwise the surviving obstruction
characterization has no habitat); WP3.0b tests whether the engine still
labels instances and whether the deep search really becomes cheaper than
the certificate just past the previously tested range; WP3.0c tests whether
ANY sheaf-side summary predicts ground truth once the trivial fiber-spread
separation is excluded (fiber spread itself is never a predictor — it IS
the certificate's own verdict input and separates decided rows circularly).
Gate G2.6 runs the hard >=2/3 rule verbatim (pre-registered thresholds
embedded in the frozen configs).

> Scope revision (same-day, pre-run). The one-seed pilot priced full-budget
n=5 engine rows at ~1.0-1.6 core-hours each
(least-squares witness search dominates; measured splits at tiny budgets:
r1 55-80 s at 4 starts, fiber 110 s at 6 starts, extrapolated to the
unchanged Phase-2 budgets give 65-95 min per undecided target row).
The original ">=400 structures" target (plan text) is compute-infeasible
inside the Colab envelope (~480 core-hours). Per the mandatory pilot-gating
policy (Research Plan Section 11), scope is adjusted BEFORE any run while the
instrument stays bit-for-bit the Phase-2 protocol
(round1 40 starts seeds 11/112, undecided round2 x2 multiplier seeds 23/124,
fiber 48 starts max_roots 12 seed 13). New scope: 216 structures x 1 draw
(12 shards x 18, 12-shard fleet, deadline-guarded); every un-run job id is
logged so a follow-up wave can take over without duplication.
Budget fairness is unchanged; the binomial CI on a 50% decidability rate
widens only modestly (~+/-7-8 pp vs the original +/-4 pp, an honest
downgrade). This note, the frozen config, and the shards' collected rows are
the audit trail; no retro-fitted threshold tempers it.

## 1. Fleet at a glance

| Notebook(s) | Work package | Shards | Mode | Expected wall | Outputs (per every shard) |
|---|---|---|---|---|---|
| `nb30_a_prevalence.ipynb` | WP3.0a natural-prevalence scan | 1 | self-contained, downloads 11 public datasets (3 NHANES cycles + 5 UCI zips + 2 OpenML sets), row-cap 30k, pool<=24 vars, subset sizes 3-6 up to 60k/subset, 30k/dataset, min_patterns>=4, min_support 1 vs 5 robustness, missing-window sensitivity, column-permutation negative control, adaptive row-level bootstrap (target ~55 min collective) | ~1-1.5 h | `results/phase3/prevalence_scan.{csv,json}` |
| `nb30_b_scaling_shard_00..11.ipynb` | WP3.0b scaling probe | 12 x 18 structures + 4-instance n=4 re-timing anchor + 1-instance n=6 probe + quota-6/shard seeded-SRS fixed-budget attacker battery (lighter than audit-full, ~8-10 min/row at n=5) for WP3.0c labels; deadline-guarded; resume-safe | CPU, ~2 cores | ~8-10 h each, ~200-250 core-h fleet | `results/phase3/scaling_probe_shardXX.jsonl`, `scaling_attacks_shardXX.jsonl`, `scaling_summary_shardXX.json` |
| `nb30_c_cycattack_shard_00..03.ipynb` | WP3.0c label-generation | 4 | attack battery (audit-identical budgets, non-shared oracles, pin-aware rebuild) on the forced-cyclic stratum's engine-undecided x sheaf-RECOVERABLE rows (784 fleet-wide) | CPU, ~2 cores | ~2-4 h each | `results/phase3/cycattack_verdicts_shardXX.jsonl`, `cycattack_sample_shardXX.json`, `cycattack_summary_shardXX.json` |
| `nb30_c_signal_analysis.ipynb` | WP3.0c analysis | 1 | recomputes share-pinned Frechet widths + cheap structural features locally (cached), stratified permutation nulls (B=2000) + random-m-graph match nulls (K=500/bucket) + downstream spread-vs-naive-pooling-error correlation (family spreads recomputed at protocol fiber budgets) | CPU, ~2 cores | ~1 h standalone; ~2 h final rerun | `results/phase3/signal_validity.{json,csv}` |

Fleet total: 18 notebooks, still far under the global 40 cap. Per the
standing Colab sharding policy (Research Plan Section 11) each notebook
embeds the entire library (concatenated `src/sheafpatternfusion/*.py` with
relative imports stripped; one top-level `from __future__ import annotations`
prepended so annotations stay unevaluated), pins `numpy==2.4.3` /
`scipy==1.17.1`, halts once for a kernel restart, and otherwise runs to
completion on either a warm Colab session or a cold one (first install cell
skips cleanly when the pins already match). Only `/tmp/opencode` and the
workspace `results/` trees are touched; every config is frozen before the
first kernel boots.

Every notebook carries the standard first-cell restart dance, `OMP_*=1`
thread caps, resume-safe JSONL checkpoints joined by `instance_id|target`
(`load_done` dedup), a self-pilot projection line before committing to its
pool, a stall watchdog that falls back to sequential execution, and the
share-pinned LP through `scipy.optimize.linprog(method="highs")`. The scaling
shards carry a soft-wall deadline that stops dispatch while draining running
jobs (queued futures are cancelled and their ids are returned in the shard
summary's `not_run` list).

## 2. What to upload and in which order

Each downloaded shard output is copied into `notebooks_colab/phase3/incoming/`
(naming must match the table; overwriting is safe because every output is
resume-safe). Order that avoids bottlenecks:

1. **Prevalence + cycattack shards first.** WP3.0a is data-access bound
(11 datasets x 3 NHANES cycle merges) and WP3.0c cycattack is pure attack
compute — both share no code path with the scaling fleet and produce
readouts readable by the gate writer without the other inputs. Start them
together with the scaling shards.

2. **Scaling shards (12).** Build the MANIFEST-matched set
(`scaling_probe_shardXX.jsonl` + `scaling_attacks_shardXX.jsonl` +
`scaling_summary_shardXX.json`). Partial waves are welcome: the collector
de-duplicates by target key and reports coverage per tag (n4t/n5/n6) alongside
the n=2/3/4 historical baselines fetched from `data/frozen/instances_merged.jsonl`.

3. After ANY wave has written files, run `python3 scripts/run_phase3.py
--stage collect` locally. It copies validated JSON/JSONL/CSV shards into
`results/phase3/` and writes `results/phase3/COLLECT_REPORT_phase3.json`
with row counts, byte counts, and checksums per family. Re-running it after
each wave is idempotent.

4. For the canonical **gate adjudication**, every hard window is one pass
locally after the fleet coalesces:

```bash
python3 scripts/run_phase3.py --stage scaling   # needs scaling shards (+merge)
python3 scripts/run_phase3.py --stage signal    # needs merge + cycattack (+scaling if present)
python3 scripts/run_phase3.py --stage gate      # writes results/phase3/gate_G26_memo.md
```

The signal stage piggybacks on WP3.0b compute for `fresh undecided rows at
n>=5` labels but otherwise runs standalone: embedded payloads for the cyclic
stratum (1120 compact rows), audit verdicts (899 rows), and the discordant
family (120 rows) let it return a full `signal_validity.json` with
`mode: PARTIAL_existing_assets_only` even before a single scaling shard lands;
dropping fleet outputs into `/content/results/phase3` before re-running its
analysis cells upgrades it to `FINAL` with the fresh-label pool included
(without a code change). The local `run_phase3.py --stage signal` always
produces the definitive run when fleet outputs exist in `results/phase3/`.

## 3. Pre-registered gates (thresholds frozen in `configs/phase3/*.json`)

**WP3.0a (prevalence scan, `configs/phase3/prevalence.json`)**

*GO* iff `n_datasets_with_cycles >= 3 OR pooled_cyclic_fraction >= 0.15` over
*eligible* subsets (`size in {3,4,5,6}` with `>=4` realized patterns at
`min_support=1`; cyclicify via `graham_acyclic` on the observed-set
hypergraph, the same predicate the engine itself reports). Supporting
readouts: per-dataset cyclic fractions (+bootstrap 5/50/95% quantiles),
row-level permutation control fraction, `robust_min_support=5` fraction,
missing-window sensitivity, and whether every cyclic subset involved a
partial (non-nested) overlap. *NO-GO* means cyclic structures are artifacts
of the forced generator — the re-aimed benchmark has no home stratum and the
re-aimed program has no habitat (boundary paper per Appendix A row 2.75).

**WP3.0b (scaling probe, `configs/phase3/scaling.json`)**

Two pre-registered arms (either feeds the gate):

* Feasibility arm: `decidability_rate_n5 >= 0.50` (engine labels obtainable on
  harder instances — sufficient density for a benchmark).
* Economics arm: `ratio = median(attack wall) / median(cert-pipeline wall)`
  on engine-undecided rows has `ratio >= 3.0 AND is strictly increasing from
  n=4 to n=5` (certificate at least 3x cheaper than the search it spares, and
  the saving grows one step past the previously tested range). Historical
  baselines at n=2 (ratio ~6.0x) and n=3 (ratio ~1.03x) are carried explicitly
  in the config for context.

Either arm returning GO contributes one G2.6 vote. The cert pipeline wall is
`struct + formula + lp + r1 + r2 + fiber` (the same accounting the Phase-2.5
pricing memo used; attacks counted separately through `total_wall_s`).

**WP3.0c (signal-validity probe, `configs/phase3/signal.json`)**

One-sided alternatives at `p < 0.01` after stratified permutation (within
`n_vars x mechanism_class` buckets):

* `AUC >= 0.75` on **attacker-labeled engine-undecided rows** for any of the
  five features (`frechet_width` corrected share-pinned, `jacobian_rank_deficiency`,
  `max_cross_pattern_marginal_gap`, `frac_observed`, `overlap_density`) or the
  standardized logistic combo (5-fold stratified CV AUC) — *with the honest
  exclusion:* fiber spread is NEVER a predictor (it is the certificate's own
  verdict input and separates decided rows trivially; context table on
  `P1_engine-decided` rows is reported only as circularity documentation),
* OR `|rho(spread, naive-pooling error)| >= 0.30` with `p < 0.01` across DGP
  draws (Pearson + Spearman arms, B=5000; naive-pooling error is the
  MCAR-plugin estimate described in `phase3_probe.naive_pooling_mean`).

The honest pool lives where labels were expensive: fresh undecided rows at
n>=5 plus the forced-cyclic stratum, labeled by the full battery (audit
payload + fleet-collected scaling/cycattack verdicts). Null (i) permutation
within matched strata and null (ii) `K=500`-per-bucket random-m-graph feature
matches ride alongside. `GO` on either arm stands alone.

## 4. Gate G2.6 (hard rule, Research Plan Section 7 / Appendix A row 2.75)

```
PROCEED to WP3.1'/WP3.2'  iff  AT LEAST 2 of
    {WP3.0a, WP3.0b(feasibility OR economics), WP3.0c} return GO.
Ties toward proceeding ONLY when WP3.0c is GO
(signal validity is non-negotiable).
Otherwise: TERMINATE the program as originally framed and ship the
boundary paper from existing assets (degeneracy result at 99.95% constant-
recoverable equivalence, audited bound <= 0.33%, cyclic obstruction
characterization with the classical-blindness witness — Frechet certifies
0/899 — plus these pivot-gate measurements. Venue: workshop/UAI-short.)
No Phase 3 spend. This is the pre-committed anti-zombie exit.
```

The default is termination. A clean `G2.5b+G2.5c` (already true) strengthens
framing and the obstruction asset, but cannot override G2.6 absence — per the
post-Phase-2.5 amendment the rescue additionally requires WP3.0c to be GO.
Gate adjudication lives in `results/phase3/gate_G26_memo.md` (filled by
`--stage gate`; PENDING probes generate a status header, not a verdict).

Only behind a PROCEED: Section 7's `WP3.1'` (Hazard-E memo plus the written
win-prediction naming the instance class/DGP family, predicted margin band,
and what would falsify it — no benchmark cell is launched without it) and
`WP3.2'` (re-aimed contamination + forced-cyclic-informed benchmark grid
behind the empirically cheapest gate the surviving theory predicts).

## 5. Local diagnostics and reproducibility

```bash
# every probe's artifact family, de-duped and checksummed
python3 scripts/run_phase3.py --stage collect --help
# WP3.0a/b/c artifacts enumerated in COLLECT_REPORT_phase3.json;
# per-shard wall-clock splits, decidability by n/tag, and cost-trend table
# are aggregated by --stage {scaling,signal}; gate by --stage gate

# the signal notebook's PARTIAL vs FINAL banner is authored for Colab:
# notebooks_cyattack and scaling shards need not finish before the first
# signal notebook run -- which returns a defensible partial readout and
# upgrades itself to FINAL on a second Run-all after the fleet coalesces.

# pinned pins live in every notebook's first code cell (numpy==2.4.3,
# scipy==1.17.1) with NBFORMAT_MIN cache; each library is the concatenated
# bundle of src/sheafpatternfusion/{mdag_dgp,lp_ground_truth,
# enumerate_structures,gluing,battery,engine2,attackers,phase3_probe}.
# Relative imports are stripped and one top-level `from __future__ import
# annotations` is prepended so annotations stay unevaluated; every name is
# wired before the first Pool spins. Thread caps are OMP_* / OPENBLAS_* = 1
# per worker, no nested BLAS.
```

Expected wall times (budgeted for the 2-core Colab flavor; include self-pilot
lines):

* Prevalence: ~1-1.5 h (self-pilot times one dataset with 8 bootstrap replicas
  and scales B adaptively to fill the 55 min bootstrap budget).
* Each scaling shard: ~8-10 h single job length if the engine stays undecided,
  but the shard soft wall at `8 h` (28800 s) stops dispatch while draining (and
  logs `not_run` ids). The 4-instance n=4 anchor (~15-20 min/row at those
  n) may finish first; the 18-instance n=5 band may see the wall.
* Each cycattack shard: ~2-4 h on the cyclic stratum's undecided rows (audit-
  identical budgets).
* Signal analysis: ~1 h standalone on existing assets; ~2 h as final rerun
  (family fiber recompute at protocol fiber budgets feeds the
  `spread_vs_naive_pooling_error` arm).

Seed registry: `configs/seeds.txt` carries the Phase-3 seeds as mandated by
Research Plan Section 8 (frozen per phase). The three Phase-3 configs carry
their `seeds` blocks verbatim before any kernel runs.

## 6. Collection paths

`notebooks_colab/phase3/` stays gitted in a clean checkout (only the
notebooks themselves — data-free). Colab downloads coalesce back at either
`notebooks_colab/phase3/incoming/` (preferred landing zone for uploaded
shard outputs) or any of the `results/` trees examined by the collector.
`results/phase3/` remains the analysis-side tree on the local checkout;
`results/phase2/` and `results/phase25/` are never overwritten by this phase,
and the collector refuses to promote malformed JSON variants (they land in the
`error` list of `COLLECT_REPORT_phase3.json` instead of the aggregation).

## 7. What was new this week

* `configs/phase3/{prevalence,scaling,signal}.json` (frozen 2026-08-26 before
  any run; scaling revised same day after the mandatory one-seed pilot
  priced full-budget n=5 rows at ~1-1.6 core-h -- scope adjusted per Section
  11 while the instrument stayed bit-for-bit the Phase-2 protocol).
* `src/sheafpatternfusion/phase3_probe.py` (structure sampler n>=5, timing-
  split pipeline worker `decide2_timed` / `run_scaling_job`, pin-aware
  instance/attack constructors `instance_from_row_fixed` / `attack_row_fixed`,
  prevalence auto-fraction helpers `realized_pattern_counts` /
  `scan_subsets` / `cyclic_fraction_bootstrap`, signal-validity helpers
  `rank_auc` / `permutation_auc_p` / `spread_naive_table`, shared payload
  codec).
* `scripts/make_colab_phase3.py` (sixteen-notebook fleet generator; see
  `notebooks_colab/phase3/MANIFEST.json` and `configs/phase3/*.json`).
* `scripts/run_phase3.py` (`--stage {collect,scaling,signal,gate,all}`).
