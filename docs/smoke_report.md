# WP1.4 Smoke Report: Correspondence and Content

Date: 2026-08-24. Runner: `scripts/run_smoke.py`; raw outputs in `results/phase1/` (`correspondence.jsonl`, `hazard_A.json`, `obstructions.json`, `smoke_summary.json`). Library under test: `src/sheafpatternfusion` (33 property/ground-truth tests green at time of writing).

## 1. Correspondence on the transcription bank

Method. For every bank instance and target: (i) the engine verdict comes from `decide()` (identification formula at machine precision for positives; model-valid witness pairs from multistart root-jumping on the observable fingerprint, which includes pattern probabilities P(R=r) plus per-pattern conditionals, for negatives); (ii) the sheaf-side verdict collects distinct factorized completions of the true observed fingerprint and measures the spread of the target across that fiber ("fiber constancy"); (iii) both are compared with the published-label expectation recorded in each config.

| Instance | Target | Expected | Engine | Sheaf fiber | Spread over fiber | Distinct completions |
|---|---|---|---|---|---|---|
| x1_mcar_joint | E[V1], E[V2], P(11) | REC | REC | REC | 0 | 1 |
| x2_mar_conditional | P(V2=1\|V1=1) | REC | REC | REC | 0 | 1 |
| x2_mar_conditional | E[V2] | REC | REC | REC | 0 | 1 |
| x3_mar_joint | P(1,1) | REC | REC | REC | 0 | 1 |
| x4_self_censor_mean | E[V2] | UNREC | UNREC | UNREC | 0.329 | 40 |
| x5_conditional_dies_with_self_selection | P(V1=1\|V2=0) | UNREC | UNREC | UNREC | 0.370 | 40 |
| x6_anchor_under_mnar | E[V1] | REC | REC | REC | 0 | 1 |
| x7_self_censor_conditional | P(V1=1\|V2=0) | REC | REC | REC | 0 | 1 |
| x8_mediated_mnar_joint | P(1,1) | UNREC | UNREC | UNREC | 0.135 | 40 |

Agreement: engine 11/11, sheaf-fiber criterion 11/11, no mismatches, so no repair iteration was needed at the verdict level. Positives are exactly identified (unique completion); negatives exhibit large fibers (40 distinct completions found within 40 starts) with material target spreads (0.14 to 0.37).

Slice validity check: mass-carrying slices W_r(o) = P(V_O=o, R=r) drawn from random factorized models are valid (nonnegative, mass equal to P(r)) in 100% of draws across all instances; they reassemble into one full table by construction. MCAR section characterization: cross-pattern equality of population-marginal conditionals holds in 100% of draws for the MCAR instance and fails under MAR/MNAR mechanisms, matching the refined claim in Section 4 below.

## 2. Hazard A probe (constraint-free linear sheaf)

Constructive test on 200 random posets (2-3 variables, random patterns): slice any global mean vector to its pattern blocks; the weighted least-squares projection returns radius exactly zero every time (200/200). The linear mean-coordinate sheaf used by the fused estimator and r* has H^1 = 0 identically: consistent families always glue, so no diagnostic or estimator behavior in B2 can rest on higher obstructions of this sheaf. This is the correct sense in which the dossier's Hazard A intuition holds.

## 3. Obstruction content witnesses (richer stalks)

Covariance-completion triangle: poset {12},{23},{13}, matched unit margins, family assigns pairwise correlations (rho12, rho13, rho23):

| Case | (rho12, rho13, rho23) | min eigenvalue | Status |
|---|---|---|---|
| O1 obstructed | (0, 0.9, 0.9) | -0.2728 | OBSTRUCTED (PSD certificate) |
| O2 control | (0, 0.5, 0.5) | +0.2929 | GLUES |
| O3 boundary | (0, -0.7071, -0.7071) | ~-2e-17 | BOUNDARY (singular completion) |

Interpretation, stated carefully because it refines the dossier rather than confirming it: the O1 obstruction is created by moment geometry (correlation matrices must be PSD for any law whatsoever), NOT by m-graph conditional-independence constraints; removing the CI constraint on stalk {12} while keeping the assigned correlation 0 leaves the PSD certificate intact. Two consequences. First, genuine higher obstructions exist and are mechanically certifiable, so the cohomology vocabulary has non-vacuous content available. Second, where the content comes from differs from the dossier's guess: constraints do not create these particular obstructions, and conversely the constraint-free FULL-LAW sheaf is not flasque in the relevant sense (surjectivity of single restrictions does not give simultaneous gluing). Additionally, a discrete triangle whose pair stalks carry marginal-independence constraints always escapes through the product law (matched singleton margins compose), so discrete zero-CI obstructions of this shape appear impossible; whether richer discrete constraint patterns obstruct is deferred to the Phase 2 enumeration (WP2.3), which is exactly where the plan places that question.

## 4. Corrections discovered and applied during WP1.4 (all documented, all tested)

1. Stalk objects corrected (formalization_v0 updated): pattern-CONDITIONAL laws do not restrict to one another by plain marginalization because different patterns condition on different events. The mass-carrying tables W_r(o) = P(V_O=o, R=r) restrict correctly and linearly; conditionals remain recoverable inside each stalk by normalization. The original B1 statement failed the mechanical check (section rate 0 on MAR instances) and was replaced before any downstream code depended on it.
2. Observable fingerprint completed: the observed data include the pattern probabilities P(R=r). Root searches matching only per-pattern conditionals admit spurious completions (this produced false "variation" on formula-certified positives x2/x3/x6 in an intermediate run); adding P(r) restored exact agreement.
3. Verdict semantics tightened: variation of a target over the assumption-free relaxation is now reported as VARIABLE_UNCONSTRAINED_ONLY, not UNRECOVERABLE, because the relaxation is larger than the model class. Negatives in the bank all carry model-valid witnesses instead.
4. Identification surprises worth recording for Phase 2: mutual selection (R1 <- V2, R2 <- V1, no self-edges, independent base) behaved IDENTIFIED once pattern probabilities entered the fingerprint, and double self-censoring (R1 <- V1, R2 <- V2 with V1 -> V2) resisted witness search in earlier runs; neither received a negative label. Both are candidates for systematic study in the enumeration phase, since they sit between the classical positives and negatives.

## 5. Instance redesign history (engines catching mislabels)

Slot x5 held two successive mislabeled negatives before its current occupant: first "selection on outcome with fully observed cause" (E[V1] is directly identified when V1 is never missing), then "double self-censoring" (E[V1] appears locally identified; the overidentified moment system pinned it). The current x5 (conditional dies when selection depends on Y itself) is engine-certified with spread 0.370. Slot x8 similarly moved from mutual selection to mediated MNAR after identification evidence emerged. This is WP1.2 doing its job; labels were never forced to match engines by fiat.

## 6. Gate G0 verdict

Pass rule: full semantic agreement on the bank (met: 11/11 engine, 11/11 sheaf-fiber, slice validity 1.0) and the content question answered either way (met: constructive H^1 = 0 for the linear sheaf; certified higher obstructions exist for covariance-valued stalks, with provenance clarified). **G0: GO**, with the Hazard A refinement and the C1 restatement (mass-carrying stalks; MCAR characterization via the marginal sheaf) carried forward as binding context for Phases 2-5.
