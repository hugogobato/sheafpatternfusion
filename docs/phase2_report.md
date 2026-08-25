# Phase 2 Report: Exhaustive Small-Poset Falsification of C1/C2

Date: 2026-08-24. Runner: `scripts/run_phase2.py` (stages pilot / enumeration /
content / analyze); grid frozen in `configs/phase2/grid.json`; seeds in
`configs/seeds.txt`. Raw outputs: `results/phase2/{instances_merged.jsonl,
enumeration.csv,content.csv,habitat.csv,summary.json,COLLECT_REPORT.json}`.

STATUS: FINAL. Collected 2026-08-25 from local partial run (346 rows) plus all
8 Colab shards (`shard_00`-`shard_07`); merge is complete against the frozen
grid: 3394/3394 expected instance-target rows, 0 missing, 0 extra, 0 corrupt
lines (`COLLECT_REPORT.json`).

## 1. What was run (WP2.1)

Grid: all m-graph structures on n=2 binary variables (32 structures x 2
parameter draws); 1400 stratified-random structures on n=3 plus the 11
mandated named mechanism classes (3 draws each; mutual selection, double
self-censoring, mediated MNAR, plain self-censoring, MAR anchor, MCAR
reference, three-var mixed/double-self, collider selection); 200 random
structures on n=4. Targets per instance: means of partially observed variables
(<=2), giving ~2900 projected instance-target rows (3394 realized after
target selection; see STATUS).

Engine instruments (ground truth): formula oracle (any registered formula
verified at machine precision against the true generating model), LP pinching
(width ~0 over the assumption-free relaxation certifies recoverability with no
assumptions at all), and model-valid witness certification via two independent
rounds of null-space root-jumping. DEVIATION (documented per protocol): the
Phase-1 SLSQP witness path was removed after scipy 1.17.1's SLSQP wrapper
deterministically corrupted the interpreter heap on certain instances
(reproducer: n3_s00019_d0); Phase 1 had already identified root-jumping as the
most effective witness strategy at these sizes. Sheaf-side instrument:
distinct-factorized-completion collection (48 starts) with target spread over
the extension fiber.

Compute: local parallel run (spawn context, checkpointed JSONL, supervisor
restarts after pool breaks) PLUS Colab shards per plan Section 11 (8
notebooks, `notebooks_colab/phase2_shard_*.ipynb`; the original blob-embedded
runners were replaced mid-phase by thin runners that pip-install the package
from `github.com/hugogobato/sheafpatternfusion` at tag `v0.2.0`, so all
shards ran byte-identical library code; collected by
`scripts/collect_shards.py` with row-count/completeness checks). Local
sharding trigger: sibling-agent CPU load kept effective throughput near 1
worker-equivalent.

## 2. Biconditional evaluation (WP2.2)

Confusion matrices per structure class: see `results/phase2/summary.json` and
`enumeration.csv`. Pre-registered thresholds: agreement >= 98%, unexplained
mismatches <= 2%.

Final numbers over the 3394 merged rows: 1202 rows engine-decidable, and on
these the biconditional holds exactly: TP 865, TN 337, FP 0, FN 0; agreement
100% (threshold 98%), unexplained mismatches 0% (threshold 2%). G3a PASSES.

By mechanism class (rows / decidable / TP+TN): MCAR 20/20/20+0; MNAR_self
3002/991/654+337; MNAR_other 372/191/191+0. By size: n=2 182 rows/120
decidable (72+48); n=3 2812/1001 (725+276); n=4 400/81 (68+13). Agreement is
1.0 in every stratum; no stratum contains a false positive or false negative.

Note: `poset_shape` of the realized patterns is acyclic for every instance in
the grid (`cyclic_poset` stratum empty); cyclic-poset behavior is therefore
covered only by the dedicated obstruction content of Section 4, not by the
biconditional grid.

## 3. Habitat check (Hazard N)

Crossing of pattern-conflict flags x engine-recoverability: `habitat.csv`.

Over the 1202 decidable rows: conflict=1 x RECOVERABLE 831, conflict=1 x
UNRECOVERABLE 319, conflict=0 x RECOVERABLE 34, conflict=0 x UNRECOVERABLE
18. Conflicts are present in 95.6% of decidable rows, but they do NOT
separate the verdicts at these sizes: conditional unrecoverability is 319/1150
= 27.7% with conflicts vs 18/52 = 34.6% without. The MCAR-style conflict flag
is thus a habitat descriptor, not a predictor of engine failure.

## 4. Obstruction content and poset readout (WP2.3)

Discrete layer (mass-carrying tables, exact LP feasibility of global
completion; mutually consistent families via constructive pair-sampler):

| Poset | Graham | families | obstructed |
|---|---|---|---|
| K4 all pairs | cyclic | 90 | 40 (44.4%) |
| 4-cycle pairs | cyclic | 90 | 0 (0.0%) |
| star-of-3 pairs | acyclic | 90 | 0 (0.0%) |
| triangle K3 | cyclic | 90 | 26 (28.9%) |

Independence-constrained triangle (product-law escape): 60 families tested,
0 obstructed; the product law repairs the triangle obstruction, confirming
that the discrete obstruction is carried by the dependence structure.
Gaussian covariance-stalk layer: randomized 4-cycle assignments, 12 of 300
obstructed (4.0%); canonical witnesses match Phase-1 certificates: mixed
4-cycle witness obstructed (min eig -0.097), triangle O1 witness obstructed
(min eig -0.273), const-0.9/const-0.5/alternating and O2 controls all
completable (`content.csv`).

Minimal hand-verifiable discrete witness (triangle): singleton margins all
(1/2,1/2); associations t12=.25, t13=.45, t23=.05. Fréchet bounds on the
joint cell t111: lower .25+.45-.5=.20 > upper min(.25,.45,.05)=.05 -> empty;
LP-certified and confirmed by 200-restart nonlinear completion attempts
(residual floor 1.7e-2 >> machine zero).

## 5. Deviations and honesty notes

1. SLSQP removal (above).
2. Engine-undecided rows: 2192/3394 = 64.6%. Of these, 2191 carry
   sheaf-verdict RECOVERABLE (85.1% with a full-rank observable-fingerprint
   Jacobian), consistent with moment-pinned identification for which no
   registered formula exists. The single exception (`n3_s03759_d0`, target
   `mean(0)`) is sheaf-UNRECOVERABLE and structurally consistent: its
   observed family is LP-incompletable (max cross-pattern marginal gap
   0.283, MCAR-style conflict) and its fingerprint Jacobian is rank-deficient
   (20 of 21). These undecided rows are excluded from the primary agreement
   denominator; the exclusion is conservative ONLY IF the sheaf side is
   right for reasons independent of the engine's failure modes, which is
   NOT guaranteed since both sides search the same fingerprint manifold.
   Treated as an explicit limitation.
3. glibc heap corruption in this scipy build forced spawn-context workers,
   wave restarts, and a crash quarantine during early local runs; after the
   SLSQP removal and the switch to pinned scipy 1.17.1 on Colab, no job was
   quarantined and the final collect report lists 0 missing / 0 extra keys.

## 6. Consequences for C1/C2

G3a PASSES its pre-registered gate: on all 1202 engine-decidable rows the
sheaf instrument agrees with ground truth exactly (100% >= 98%; unexplained
mismatches 0% <= 2%), in every mechanism class and at every size n=2,3,4.
The biconditional C1 (engine-recoverable iff sheaf-recoverable) is therefore
empirically supported wherever the engine decides, and Section 4 supplies
independent discrete and Gaussian obstruction witnesses showing the sheaf
side detects genuine non-completability by construction. The 64.6%
engine-undecided mass extends nominal sheaf coverage to 3393/3394 rows but
remains an assumption of this readout, not a certified equivalence; a Phase 3
gate should target shrinking that mass directly.
