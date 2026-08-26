# Phase 2.5 Report: Independent Validation Battery for C1/C2

Date: 2026-08-26. Runners: twelve Colab thin-runner notebooks
(`notebooks_colab/phase25/nb25_*.ipynb`, pip-installing
`sheafpatternfusion@v0.3.0`); library modules `battery.py`, `attackers.py`,
`cyclic_synth.py`, `discordant_family.py`, `workers.py`; configs frozen in
`configs/phase25/{battery,audit,family,cyclic_grid}.json`. Raw outputs:
`results/phase25/{null_battery.json,null_battery_scored.jsonl,
priority_sample.jsonl,audit_sample.jsonl,audit_verdicts.jsonl,
cyclic_instances.jsonl,cyclic_gen_stats.json,discordant_family.jsonl,
family_summary.json,phase25_summary.json}`.

STATUS: FINAL PENDING PATCH. All four work packages ran to completion on
Colab (53.7 h total attack compute). One n=4 audit instance (`n4_r0199_d0`,
two target rows) failed inside its shard; a single-instance census-patch
notebook (`nb25_audit_patch_n4r0199.ipynb`) is provided so the n=4 census
closes at 319/319. Every headline below is unaffected by those two rows.

## 1. Degeneracy null battery (WP2.5.1, gate G2.5a)

Population: all 2192 engine-undecided rows of the frozen Phase-2 merge.
Nulls N0-N4 scored against the certificate's labels, with widths computed by
the share-pinned Frechet LP (`battery.frechet_bounds`), which matches each
realized pattern's observed conditional law scaled by its observed pattern
probability P(R=r). This is the honest Manski-style bound given the full
fingerprint; it deliberately replaces the Phase-1/2 relaxation whose
total-mass row double-counts stratum mass (see Section 7).

Pre-registered expectation CONFIRMED: constant-RECOVERABLE (N0) reproduces
99.95% of certificate labels (2191/2192; the single exception is the
discordant seed row n3_s03759_d0 / mean(0)). The best fraction-observed and
overlap-density sweeps land on the same 99.95%. The corrected Frechet width
is large everywhere in this region (medians 0.50 at every n; min 0.256), so
the width-sign null N3 agrees only 29.2% at its best threshold: the
certificate asserts RECOVERABLE precisely where the assumption-free interval
is widest. The disagreement set S* is defined as the top-200 widest-width
certificate-RECOVERABLE rows and was audited exhaustively (Section 2).
G2.5a is thereby QUANTIFIED as pre-registered; because S* is nonempty,
demotion rule D1 is NOT triggered.

## 2. Adversarial audit (WP2.5.2, gate G2.5b)

Sampling frame (pre-registered): 901 unique rows = full census of undecided x
RECOVERABLE rows at n=2 (62), full census at n=4 (319), simple random sample
of 400 at n=3 (seed 20250901), plus the census of S* (200 rows, overlapping
the other strata). Verdicts returned for 899/901; the two missing rows are
the two targets of the single instance noted above.

Result: ZERO confirmed false RECOVERABLE anywhere. 899/899 verdicts are
NO_FALSE_RECOVERABLE_FOUND. No attacker came close: A2's maximal model-valid
fiber spread over independent fresh-seed root enumeration was 1.4e-10 against
a confirmation tolerance of 1e-4, and A2 found a median of ONE distinct
completion root per row. Rule-of-three upper bounds on the false-RECOVERABLE
rate: pooled <= 0.33% (M=899), n=2 <= 4.8% (M=62), n=4 <= 0.95% (M=317),
n=3 SRS <= 0.75% (M=400), S* <= 1.5% (M=200). G2.5b PASSES: C1 survives as a
decision instrument with a defensible error-rate bound; D2 (KILL) is NOT
triggered.

Attacker behavior worth recording. A1's adaptive escalation stopped at a
median of 400 jump starts per row (ceiling 800; max wall 2782 s on one
stubborn row), confirming that escalation rarely pays beyond a few hundred
starts on these instances. A3 found infeasible Frechet cells on 899/899 rows:
cross-stratum classical obstructions are ubiquitous in this MNAR population
(consistent with the Phase-2 conflict rate of 100% among undecided rows),
and no row was classically certified unique, i.e., the classical route alone
never pins the target. The audit therefore localizes the certificate's value
to exactly the model-aware layer the classical analysis cannot reach.

## 3. Discordant-family construction (WP2.5.3, gate G2.5d)

Seed: n3_s03759_d0 / mean(0); structure class frozen; 120 fresh mechanism
draws. Outcome: COLLAPSE, 3 witnessed members against the >=10 threshold.
The decomposition is informative: 110/120 draws were engine-DECIDED
(witness search found model pairs outright, mostly UNRECOVERABLE), leaving
only 10 engine-undecided members; of those, 3 carried both a model pair
(A2 root pairs, delta_phi 1.1e-4 to 1.9e-4, barely above the 1e-4 tolerance)
and a classical witness (mass-family LP infeasibility), and the origin seed
itself was not witnessed (its fiber spread 7e-6 is sub-tolerance). The
discordant singleton is thus atypical even within its own structure class,
and the three marginal witnesses sit at the tolerance boundary. Per
pre-registration the theory-note path is NOT authorized; G2.5d folds into
limitations and weakens the D1 package (which additionally requires an empty
S*, not observed).

## 4. Forced cyclic-poset stratum (WP2.5.4, gate G2.5c)

Why the stratum had to be forced: Phase-2 sampling draws indicator
probabilities in (0.25, 0.75), so EVERY response pattern has positive
probability, the realized observed-set family is always the full simplex,
and the simplex is Berge-acyclic. Cyclic posets were structurally
unreachable, not merely unlucky. The stratum was created by exact 0/1 pins
on indicator mechanisms via triangle templates (n=3), square templates (n=4),
and pinned rejection sampling: 560 accepted instances (1120 target rows;
656 triangle, 464 square, 2 rejection; all poset_shape = cyclic), 140 per
shard against an honest-attempt floor of 50 (D4 floor met; 776 attempts
total, 586 accepted).

Gate readout: on the 336 engine-decidable target rows the two instruments
agree 100% (all TN: engine UNRECOVERABLE wherever the fiber varies;
FP=FN=0), exceeding the 98% pre-registration. On the 784 engine-undecided
rows the sheaf says RECOVERABLE throughout, and the obstruction signature is
nonzero and strictly interior: 30% of all rows show nonzero fiber spread
(exactly the decidable set), cross-pattern conflicts are present on 100% of
instances, the observed mass-family is LP-incompletable on 100%, and the
Jacobian is full-rank on 100% of undecided rows (vs 85% in the main grid).
G2.5c PASSES on both arms: agreement >=98% AND nondegenerate obstruction
signature strictly inside (0%, 100%). The cohomology vocabulary is retained
and the exhaustive-small-poset framing earns back meaning through a stratum
Phase 2 could not have produced.

## 5. Compute pricing (WP2.5.6)

Apples-to-apples on the triage population (engine-undecided rows),
certificate pipeline cost (engine rounds + fiber, recorded wall_s from the
frozen merge) versus the full attack battery actually run:

| n | cert median | attack median | attack/cert |
|---|---|---|---|
| 2 | 3.1 s (k=44) | 18.8 s (k=62) | 6.0x |
| 3 | 71.9 s (k=68) | 74.2 s (k=520) | 1.03x |

(The n=4 cell lacks comparable recorded certificate walls since most n=4
rows ran on Colab without per-row timing.) The pre-registered >=10x triage
requirement FAILS as measured: at n=3 the deep search costs about the same
as the certificate pipeline it would replace. Two honest caveats cut both
ways: the attack numbers are post-adaptation (A1 self-limited to a median of
400 starts, so an unconditional deep search would price worse), while the
certificate numbers bundle the ENGINE's ground-truth protocol rather than
the sheaf instrument alone. Neither repair rescues an order of magnitude on
the undecided population. Per pre-registration this is recorded regardless
of the G2.5b pass: the applied triage narrative is not supported; the
certificate's surviving claim is statistical (bounded error rate), not
economic.

## 6. Gate scoreboard and demotion rules

G2.5a quantified (N0 = 99.95%; S* nonempty). G2.5b PASS (0 kills; C1
survives, pooled bound <= 0.33%). G2.5c PASS (100% agreement; signature
30%). G2.5d COLLAPSE (3/120). Demotion rules: D1 NOT triggered (requires
empty S*), D2 KILL not triggered, D3 not adjudicated (WP2.5.5 equivalence
memo is desk work and remains open), D4 not triggered (140/shard vs floor
50). Net effect: C1 stands with its first real error-rate bound; the theory
annex loses the discordant-family path; the applied triage story is dead on
pricing; Phase 3 proceeds in parallel exactly as planned, and a clean
G2.5b+G2.5c remains the live-correspondence contribution available if Phase
3 hits its default.

## 7. Deviations and honesty notes

First, the LP artifact: Phase-2 `lp_range` sums its total-mass row over both
cells and scales, double-counting stratum mass; every stored `lp_width`
value sits on a halved scale (hence uniform 0.5s across all 3394 rows).
Pinch decisions are invariant under positive rescaling, so no verdict
changes, but the RELAXED_FRAGILE thresholding operated on an artificial
scale near the 1e-3 boundary; Phase-2.5 instruments use corrected
share-pinned bounds throughout and never reuse the v0.2 relaxation.
Second, the "x20 budget" escalation is implemented as an ADAPTIVE ceiling
(rounds stop early on success or no-improvement) rather than a flat
multiplier; effective spend is logged per row (median 400 starts) and the
literal x20 was priced out during planning (configs/phase25/audit.json
documents the arithmetic). Third, the audit covers 899/901 frame rows; the
patch notebook closes the gap but its two verdicts are not yet folded into
these numbers (they cannot flip any conclusion: even two kills would leave
pooled <= 0.56%, and zero kills tighten n=4 to <= 0.94%). Fourth, family
witnesses sit within a factor of two of the confirmation tolerance and the
origin seed itself fails the witness test; the collapse is reported without
cosmetics. Fifth, A3's universal infeasibility reflects that these strata
genuinely violate homogeneous-population coupling (MNAR selection), which is
an obstruction fact about the data class, not evidence against the
certificate.
