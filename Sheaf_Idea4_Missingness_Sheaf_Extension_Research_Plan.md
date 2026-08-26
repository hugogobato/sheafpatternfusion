# Research Plan: Missingness as Sheaf Extension (Recoverability and Robust Pattern Fusion)

Project: `SheafPatternFusion` (Idea 4 of `Sheaf_Research_Ideas.md`)
Prepared: August 2026
Status: planning document, pre-registration of gates and give-up rules

---

## 1. Executive verdict

1. **Classification:** promising but unproven. The novelty gap survived a fresh arXiv sweep today, but every load-bearing claim is empirical and untested, and two structural hazards (incumbent reduction, minimum-distance equivalence) could shrink the paper to a reformulation.
2. **Confidence and evidence level:** medium confidence. Verified today at abstract/metadata level: Robinson-Szulczewski-Thorson DSEM chapter (arXiv:2511.04603), Scott-Valdano-Assaad cluster m-graphs (arXiv:2605.20943), Idrissova-Rekik multimodal sheaf network (arXiv:2508.09717). Core tool papers (Hansen-Ghrist arXiv:1808.01513, Kearney-Palmowski-Robinson arXiv:2012.00120) verified previously. No direct prior art found for sheaf-theoretic missing-data recoverability or pattern-fusion estimators (arXiv API, Aug 2026). All empirical claims below are `UNTESTED`.
3. **Proposed contribution (one sentence):** place a cellular sheaf on the poset of missingness patterns so that (i) m-graph recoverability becomes a global-section/fiber statement that can be checked constructively, (ii) a consistency-radius statistic diagnoses pattern disagreement beyond sampling noise *with localization* (which pattern is contaminated), and (iii) a projection-based pattern-fusion estimator dominates complete-case analysis and multiple imputation in a characterizable band of MNAR contamination while never being catastrophically worse.
4. **Contribution vs engine vs application:** the contributions are the fusion estimator (C3) and the diagnostic test (C4); the recoverability correspondence (C1/C2) is grounding theory that must reproduce known results or die. The engine is the shared sheaf library (C5). The application is a public-data smoke test (C6). Decoration risk: cohomology vocabulary, if no enumerated instance ever exhibits genuine higher obstruction; cut without sentimentality if so.
5. **Strongest reason it could become a strong field paper:** current missing-data practice returns either a point estimate (complete-case, IPW, MI) or a binary modeling verdict, with no principled scalar that says "your patterns disagree beyond sampling noise, here is by how much, here is where, and here is an estimate robust to it." A working C3+C4 pair supplies exactly the missing layer, anchored to the rigorous m-graph semantics.
6. **Strongest reason it could fail or become incremental:** three named hazards. Hazard C: MI (with pattern covariates) and pattern-mixture models may already dominate the fusion estimator everywhere it matters, and Little's test plus pairwise overlap checks may already suffice for diagnosis. Hazard E: in the linear-Gaussian case the fused projection may collapse to ordinary minimum-distance pooling of pattern-specific moments, i.e., known econometrics in sheaf notation. Hazard N (helpful-regime emptiness): if conflicting patterns force unrecoverability of the target in every enumerated structure, the estimator has no legitimate habitat (the regime where fusion could help is exactly the regime where the estimand dies).
7. **Next unresolved gate:** Gate G0 (fatal viability), inside Phase 1, decided by WP1.4.
8. **Cheapest decisive next action:** WP1.4, the correspondence smoke test: verify on a handful of textbook m-graph instances that the sheaf section/fiber criterion reproduces published recoverability verdicts, and probe whether the construction has any obstruction content at all. Days of work, no new theory required.

> **Amendment (2026-08-26, post-Phase-2.5).** Phases 1-2.5 have run. Outcomes: correspondence holds everywhere it is testable (100% on decidable rows; audited clean, 0 false claims in 899), BUT the certificate is marginally VACUOUS in its claimed value region (constant-RECOVERABLE reproduces 99.95% of its labels; marginal info ~1/2192) and ECONOMICALLY DEAD (~1x the deep search it would replace; >=10x required). The forced cyclic stratum delivered genuine, classical-tools-blind obstruction content (G2.5c PASS); the discordant-family theory path collapsed (G2.5d). Consequence adopted by council review: C1 is DEMOTED to motivation/appendix effective immediately; the program continues ONLY through the pivot-gate week WP3.0 (Section 7, Phase 3 preamble) with a re-aimed Phase 3 behind it; termination-with-boundary-paper is the default if the gate fails. See Sections 6-7 and Appendix A.

---

## 2. Idea reconstruction and claim decomposition

**Scientific problem.** Missing data arrives in patterns: each pattern observes a different subset of variables, and the m-graph framework (Rubin 1976; Mohan-Pearl-Tian 2013; Mohan-Pearl JASA 2021) says when functionals of the full-data law are recoverable from pattern-specific observed laws. Incumbents answer with graphical algorithms on exactly known diagrams and exact laws; practitioners face estimated pattern-laws that disagree due to MNAR contamination, reporting error, or model error, and currently just pick a pattern (complete-case) or pool blindly (MI).

**Unit of analysis.** A dataset with categorical missingness patterns p over variables V; per-pattern samples giving estimated laws Q_p on observed sets O_p; an assumed (possibly approximate) m-graph G.

**Target estimand.** A functional phi of the full-data law P (mean E[Y], ATE), plus meta-quantities: recoverability status of phi, consistency radius r* (distance from {Q_p} to the near-global-section set), contamination location.

**Method.** Base space: poset of patterns ordered by observed-set inclusion. Stalk F(p): families of laws on O_p satisfying the m-graph's conditional-independence constraints restricted to O_p. Restrictions: marginalization. Global section: a full-data law whose every pattern-marginal agrees with the assigned locals. Recoverability: phi constant on the fiber of global extensions of the observed partial sections. Disagreement handled by Robinson-style assignment-sheaf consistency filtration: tau-thresholding and minimal radius r*, computed in the Gaussian case by sheaf-Laplacian pseudoinverse projection. The fused estimator projects pattern-specific estimates onto the (near-)global-section affine subspace.

**Assumptions.** Overlapping observed sets across comparable patterns (else no restrictions exist); m-graph correct up to the contamination regime studied; in Phase 3, linearity/Gaussianity for closed forms; positivity of pattern sample sizes.

**Audience and venue.** Missing-data/causal community and statistics-methods venues: Journal of Causal Inference, UAI/CLeaR; AoS/JMAS-style outlets only if the deferred theory lands strong. Statistical software angle (R/Python package) increases uptake.

### Claim decomposition

| ID | Claim | Type | Feasibility | Novelty | Importance | Evidence | Keep / cut / pivot |
|----|-------|------|-------------|---------|-----------|----------|--------------------|
| C1 (T1) | The m-graph observed-data factorization (Mohan-Pearl-Tian) is equivalent to the section condition of the canonically built pattern sheaf; recoverability of phi equals constancy of phi on the fiber of global extensions | enabling/grounding | High | Low-medium (translation hazard) | Necessary: anchors all semantics | TESTED P2/P2.5: 100% consistency on decidable rows; audited clean (0/899); BUT marginally vacuous (constant-RECOVERABLE null = 99.95%; info ~1/2192) and ~1x deep-search cost -> DEMOTED to motivation/appendix per Amendment 2026-08-26 | Demoted to grounding proposition; no further investment |
| C2 (T2) | On chains/trees of patterns, pairwise gluing suffices (closed form); on general posets there exist genuine higher obstructions (pairwise-consistent families that fail triple consistency under m-graph constraints) | contribution (theory layer) | Uncertain (Hazard A: obstruction may never fire) | Medium | Medium | PARTIAL P2.5: forced cyclic stratum exhibits genuine signature (30% nonzero fiber spread; 100% LP-incompletable families; classical Frechet certifies uniqueness 0/899). Natural-prevalence untested -> gated by WP3.0a | Keep behind WP3.0a prevalence gate |
| C3 (T3) | Projection-based pattern-fusion estimator dominates complete-case AND MI simultaneously in a characterizable MNAR-contamination band, and never loses to the best baseline by more than a bounded margin | contribution (expected headline) | Moderate-high | Medium-high (Hazard E) | High | UNTESTED | Keep; primary methods bet |
| C4 (T4) | Consistency radius r* is a calibrated omnibus test of pattern disagreement (bootstrap-calibrated size) with localization power where pairwise tests are blind | contribution (co-headline) | Moderate | High | High | UNTESTED | Keep; second methods bet |
| C5 | Mini-library: pattern-poset sheaf, restriction assembly, sheaf Laplacian, H^0/H^1 dims, r*, Gaussian projection | engine | High | Low (tooling, shared with Ideas 1/5) | Enabler | n/a | Keep, tightly scoped |
| C6 | Public-data smoke test (NHANES subscale or UCI adult-style) with diagnostics reported beside baselines | application | High | Low | Credibility | UNTESTED | Keep at smoke-test scale only |

Load-bearing contribution: C3 first, C4 second (the dossier itself flags C1/C2 as translation risks, so the methods layer is the durable core). Load-bearing assumption: there exist practically relevant regimes (target recoverable, pattern-laws mildly conflicting) where projection fusion beats both complete-case and MI and r* detects what standard tests miss. Most dangerous prior-art collision: the Robinson group pointing their existing consistency-radius machinery at survey/statistical missingness (they already advertise sheaf-based inference of missing data in DSEMs). Strongest simple baseline: multiple imputation (properly specified, e.g., MICE with pattern-aware terms); complete-case as the second incumbent. Cheapest decisive experiment: WP1.4. Hardest referee objection: Section 5 item 6. Attractive component to cut if needed: the general-poset cohomology story (C2); the paper survives on C3+C4 without it.

---

## 3. Fatal-flaw certificate (Gate G0)

Checks performed at planning time (symbolic reasoning, no code yet):

1. **Typing and coherence.** Stalks as constrained families of laws on O_p, restrictions as marginalizations, sections as compatible families: types match; everything finite-dimensional in the Gaussian phase. Pass (informal).
2. **Hazard A (vacuous obstruction).** Between *unconstrained* law families, marginalization maps are surjective, so a constraint-free sheaf is flasque-like and its H^1 vanishes identically; the certificate would never fire. The obstruction content, if any, must come from the m-graph's conditional-independence constraints restricting which families count as sections. Consequence: WP1.4/WP2.3 must exhibit enumerated instances with genuine higher obstruction (pairwise-consistent, globally impossible under constraints); otherwise the cohomology vocabulary is decoration and gets stripped. Status: unresolved gap, drives WP1.4 and WP2.3.
3. **Hazard B (sampling noise).** Estimated pattern-laws are never exactly compatible, so exact gluing is the wrong question in practice; the assignment-sheaf/tau-consistency machinery (thresholds, minimal radius) is mandatory, and the diagnostic layer needs valid null calibration. Design driver, not fatal. Status: acknowledged; shapes Phase 4.
4. **Hazard C (incumbent reduction).** Properly specified MI and pattern-mixture models may match or beat the fused estimator everywhere; Little's MCAR test plus pairwise overlap comparisons may match the diagnostic. This is the main scientific risk of the methods layer and is answered only by the pre-registered fair fights of Phases 3-4. Status: unresolved gap, drives WP3.2/WP4.1.
5. **Hazard E (minimum-distance equivalence).** In the linear-Gaussian case, projecting pattern estimates onto the consistency subspace plausibly coincides with minimum-distance/GMM combination of overidentified pattern moments. If so, the point-estimation algebra is not novel; novelty must live in (a) the m-graph determining *which* restrictions enter (semantics generic MD lacks), (b) the robustness band under inconsistency, and (c) the diagnostic layer. Status: unresolved gap; WP3.1 must characterize the equivalence explicitly rather than hide it.
6. **Identification content.** Ground truth is independent of the sheaf machinery: brute-force LP feasibility over latent missingness mechanisms for binary instances (cross-checked against published Mohan-Pearl verdicts) and closed-form Gaussian algebra (cross-checked against Monte Carlo). No circularity. Pass.
7. **Degenerate cases.** Single pattern (r* = 0, fusion degenerates to that pattern's estimate): fine. Fully observed pattern present: complete-case enters the benchmark bank naturally. Patterns with empty overlaps: no restrictions exist, fusion undefined; handled by scoping (overlap required, stated as an assumption). Pass with noted scoping.
8. **Helpful-regime emptiness (Hazard N).** The fusion estimator is only interesting where patterns disagree while phi remains recoverable (approximately). If enumeration shows conflicts always force unrecoverability, the estimator's habitat is empty and the project dies. Explicitly tested at WP2.2/WP3.2 before any investment in Phase 4+. Status: unresolved gap, explicit gate.

**G0 verdict at planning:** CONDITIONAL GO. No demonstrated fatal defect of the idea; four named hazards (A, C, E, N) with concrete resolution experiments wired into Phases 1-3. If the correspondence fails (WP1.4) or the habitat is empty (WP2.2), the project terminates or pivots per the give-up rules below.

---

## 4. Verified prior art and nearest-neighbor map (Gate G1)

Searches run today (arXiv API): `"sheaf" AND "missing data"` (2 hits, inspected below), `"cellular sheaf" AND ("missingness" OR "missing values")` (0 hits), `(m-graph OR "missingness graph") AND recoverability` in stat.* (1 hit, inspected below). Remaining query families queued as WP1.1 with quotas.

| Source | Same problem? | Same target? | Same method? | Same evidence? | Remaining gap | Direct-hit risk |
|--------|---------------|--------------|--------------|----------------|---------------|-----------------|
| Mohan, Pearl, Tian, Graphical Models for Inference with Missing Data, NeurIPS 2013; Mohan, Pearl, JASA 2021, DOI 10.1080/01621459.2021.1872461 [DOI digits to verify] | Yes, the problem inventory | Per-query recoverability decisions | Graphical criteria on exactly known diagrams and laws | Symbolic theory + small examples | Binary verdicts, no noise model, no quantitative disagreement readout, no localization, no robust estimator | None (this is the incumbent semantics we extend) |
| Scott, Valdano, Assaad, Missing data and cluster graphs (m-C-DMG/cm-C-DMG), arXiv:2605.20943 (verified today) | Adjacent (abstraction of m-graphs) | Recoverability of joints and macro effects under coarse graphs | Graphical conditions | Theory | No topology, no fusion estimator, no diagnosis layer; confirms the m-graph community is active but purely graphical | Low (watch monthly) |
| Robinson, Szulczewski, Thorson, Analyzing the Topological Structure of Composite Dynamical Systems, arXiv:2511.04603 (verified today) | Partially (sheaf consistency, "missing data can be inferred") | Consistency testing and inference in DSEMs | Assignment sheaves, consistency radius | Theory + food-web case study | No missingness semantics (MCAR/MAR/MNAR), no recoverability, no estimands/estimators, no causal content | **Medium**: same toolkit, adjacent application; monitor |
| Kearney, Palmowski, Robinson, arXiv:2012.00120 (verified) | No (network control) | Minimum consistency radius | Sheaves on posets, consistency filtration | Theory | Tool provider (our computational template) | None |
| Hansen, Ghrist, arXiv:1808.01513 (verified) | No (spectral sheaf theory) | Hodge Laplacians, harmonic sections | Spectral linear algebra | Theory | Tool provider | None |
| Idrissova, Rekik, Multimodal Sheaf-Based Network for Glioblastoma Subtype Prediction, arXiv:2508.09717 (verified today) | Vocabulary only ("missing data") | Classification under dropped modalities | Deep sheaf network with reconstruction | Medical imaging benchmark | Not statistical missingness, no identification content | Low (cite to defuse confusion) |
| Rubin 1976 (Biometrika, DOI 10.1093/biomet/63.3.581 [to verify]); Little's MCAR test (JASA 1988 [to verify]); MICE (van Buuren); pattern-mixture literature; IPW missing-data lineage [refs to verify] | Yes (practice incumbents) | Point estimates under MCAR/MAR assumptions | CC, IPW, MI, PMM | Methodological literature | No omnibus disagreement scalar with localization; degrade unpredictably under pattern conflict | None (benchmarks) |
| Shpitser-lineage semiparametric ID with missing data [exact ref to verify in WP1.1] | Adjacent (semiparametric recoverability) | Identifying functionals from observed law | Semiparametric theory | Theory | Acknowledge as the semiparametric counterpart; no sheaf/quantitative-disagreement layer | Low-medium |

**Statement of novelty (source-backed, current as of Aug 2026):** no verified work treats statistical missing-data recoverability or pattern-fusion estimation with sheaf machinery, and none returns a calibrated scalar disagreement measure with localization. The gap is real. Two live watch items: the Robinson group (owns the toolkit) and the m-graph community (owns the semantics; currently expanding along abstraction axes, not algebraic ones).

**G1 verdict at planning:** GO (provisional, pending WP1.1 completion). Strongest incumbents for all comparisons: properly specified MI, complete-case, IPW, pattern-mixture; Little's test plus pairwise overlap statistics for the diagnostic layer.

---

## 5. Impact thesis and skeptical-referee test

1. **Why the problem matters.** Nearly every applied dataset has missingness; analysts currently choose between trusting one pattern, weighting, or imputing, with no principled readout of whether their patterns tell a coherent story.
2. **What changes if this works.** Practitioners get (i) a graded, calibrated disagreement score with localization (which pattern is off), (ii) an estimator that fuses patterns robustly instead of picking or pooling blindly, and (iii) a constructive, mechanical route to recoverability checks grounded in m-graph semantics.
3. **Who would use or cite it.** Applied statisticians handling surveys/cohort data, the m-graph community, the broader sheaf-ML community looking for substantive applications, and eventually software users (an R/Python implementation is a realistic deliverable).
4. **Why simpler incumbents are insufficient.** CC discards data and breaks under MNAR; MI imputes away exactly the disagreement signal we want to detect; IPW needs correct mechanism models; none returns a scalar "how incompatible are my patterns" with a bias reading, and none localizes contamination to a pattern.
5. **Why not merely a combination.** The claim is a correspondence (factorization iff section condition; recoverability iff fiber constancy) plus two new capabilities (robust projection fusion; localized disagreement testing) derived from one object (the pattern sheaf and its Laplacian), not notation wrapped around MI.
6. **Most damaging plausible referee paragraph.** "The sheaf is bookkeeping for the pattern poset. In the Gaussian case the fused estimator is minimum-distance pooling of pattern-specific moments, which is textbook econometrics (Hazard E admitted by the authors). In the discrete case recoverability is decidable by existing graphical algorithms. The consistency radius is a monotone function of ordinary overlap discrepancies. Nothing required sheaf theory."
7. **Evidence needed to answer it.** Enumerated instances where the m-graph-derived restriction structure demonstrably changes which constraints bind (vs naive pairwise pooling); r* detecting joint multi-pattern disagreement at matched size where all pairwise statistics are blind, with localization; the dominance band of the fused estimator mapped honestly against tuned MI and pattern-mixture baselines; the correspondence reproducing published recoverability verdicts mechanically (constructive checkability as a service to the community).

### Impact dimensions (planning-time audit trail)

| Dimension | Score | Note |
|-----------|-------|------|
| Problem importance | 3 | Missingness is ubiquitous; recoverability is consequential |
| Novelty after prior art | 2 | Gap confirmed today; two live watch items |
| Mechanism or insight | 1 (UNTESTED) | Plausible; hazards A/C/E/N unresolved |
| Empirical advantage | 0 (UNTESTED) | Nothing run yet |
| Applied value | 0 (UNTESTED) | Smoke test only planned |
| Generality | 2 | Toolbox shared with Ideas 1/5 |
| Credibility | 2 | Hazards named, gates pre-registered |
| Paper coherence | 2 | One story if C3/C4 carry it; diffuse if theory half-survives |

---

## 6. Dependency graph and gate map

```text
P1 De-risk & enable (G0 validity, G1 novelty, G2 implementability)      [COMPLETE 2026-08: all GO]
   -> P2 Enumeration falsification of C1/C2 (G3a)                       [COMPLETE 2026-08-25: PASS, engine undecided on 64.6%]
       -> P2.5 Independent validation battery C1/C2 (G2.5a-d)           [COMPLETE 2026-08-26: b,c PASS; d COLLAPSE; pricing FAIL]
           -> P2.75 Pivot-gate week (G2.6)                              [ACTIVE NEXT: three probes, hard >=2/3 gate]
               -> P3' Fusion benchmark C3, RE-AIMED (G3b)               [CONDITIONAL on G2.6 pass]
                    -> P4 Diagnosis C4 + scaling + real-data smoke (G3c, G4a)  [DORMANT UNTIL P3' gate]
                         -> P5 Evidence-earned theory + paper (G5, G6)     [DORMANT UNTIL P4 gate]
   -> (parallel branch) Boundary/obstruction paper from P2.5 assets       [ACTIVE if G2.6 fails]
```

Parallelism note: P2 (representation theory falsification) and P3 (estimation benchmark) consume disjoint Phase 1 outputs and kill independently; running them concurrently wastes nothing if either dies. P2.5 validated the certificate with non-shared oracles only and shares no code path with P3, so it gates framing and demotion, never the estimator race. P4 waits on P3 (diagnosis is only worth building around a surviving estimator) and uses P2/P2.5 findings as framing only.

**Post-Phase-2.5 note.** The original rescue clause ("a clean G2.5b+G2.5c+G2.5d overrides a Phase-3 KILL default") is weakened: G2.5d collapsed. Any future correspondence-rescue of a failed Phase 3 additionally requires WP3.0c (signal validity) to return GO — the audit alone cannot carry it, because a certificate that commits to UNRECOVERABLE once per 2192 rows cannot fail an audit informatively.

---

## 7. Phase-by-phase execution program

**Math policy (per portfolio rules).** Phase 1 contains only *enabling* formalization: definitions precise enough to code, no proofs. Synthetic experiments (correctness smoke test, then enumeration) come before any substantial theory. All proofs live in Phase 5, which stays dormant behind the Phase 4 gate.

**Compute policy (summary; details in Section 11).** Pilot every experiment locally with one seed, recording wall time and peak RSS. Any experiment projected to exceed **2 hours wall time or 4 GB peak memory** is split into independent, self-contained Google Colab notebooks (up to 40 available). Local runs use 12-16 process-level workers on the i9-13900H (20 logical CPUs, headroom left for other projects), one BLAS thread per worker, no nested parallelism, RAM-capped.

---

### Phase 1: De-risk and enable (target: 2-3 weeks)

**Purpose and scientific question.** Does the pattern-sheaf construction reproduce known m-graph semantics correctly, does it have any obstruction content at all, and can we implement it correctly? Gates G0, G1, G2.

**Prerequisites.** None beyond the dossier and verified sources.

#### WP1.1 Complete prior-art and .bib verification
- Objective: close the novelty search; produce a verified bibliography.
- Actions: 1. run remaining query families on arXiv/Semantic Scholar/Google Scholar: "sheaf imputation", "cohomology missing data", "topological missing data", "consistency radius survey", "pattern mixture sheaf", "poset missingness patterns"; 2. inspect beyond abstracts: Robinson et al. arXiv:2511.04603 (what exactly their missing-data inference does), Scott et al. arXiv:2605.20943 (full text), Mohan-Pearl JASA 2021 (theorem statements C1/C2 must reproduce); 3. verify flagged references (Rubin DOI, Mohan-Pearl JASA DOI digits, Little 1988, IPW lineage, Shpitser-lineage ID reference, Robinson Information Fusion 2017) and build `refs.bib` with URLs/DOIs, flagging anything unverifiable as comments; 4. set monthly alerts on the two watch lines (Robinson group; m-graph community).
- Outputs: `/home/hugo_souto/Stuff/Research/sheafpatternfusion/docs/evidence_register.md`, `refs.bib`.
- Verification: every load-bearing row of the Section 4 table upgraded to E3 or downgraded with recorded consequence.
- Pass rule: no unexamined direct hit remains. Fail rule: a direct hit found (sheaf-theoretic missing-data recoverability or a pattern-fusion estimator with diagnosis already published) -> KILL, write a diagnostic document instead of continuing.
- Compute: hours, local.

#### WP1.2 Canonical example bank and ground-truth engines
- Objective: machine-readable suite of m-graph instances with ground-truth recoverability labels, independent of any sheaf code.
- Actions: 1. implement the brute-force recoverability engine: binary variables, LP feasibility over latent missingness-mechanism cardinalities (inflation-style encoding), returning recoverable/unrecoverable per query; 2. transcribe 6-10 textbook instances from Mohan-Pearl-Tian 2013 and Mohan-Pearl JASA 2021 (both positive and negative results) with labels from the published analyses; 3. implement the linear-Gaussian DGP simulator with pattern-specific shift knobs (MNAR contamination injection) and the closed-form Gaussian ground truth; 4. verify closed form against Monte Carlo intervention sampling; freeze seeds and configs.
- Outputs: `src/sheafpatternfusion/{mdag_dgp.py,lp_ground_truth.py,gaussian_ground_truth.py}`, `configs/examples/*.yaml`, `tests/test_ground_truth.py`.
- Verification (mechanical): unit tests green; closed form matches Monte Carlo within 1e-2 on estimand scale. Scientific: LP engine reproduces every published verdict in the transcription bank.
- Pass rule: ground truth trustworthy. Fail rule: LP and published verdicts disagree on any transcribed instance -> stop, resolve the encoding before anything else (this would contaminate every downstream evaluation).
- Compute: hours-days, local.

#### WP1.3 Formalization v0 (enabling only)
- Objective: one unambiguous construction, written precisely enough to code, with no proofs.
- Actions: 1. specify B1 (pattern-poset sheaf: constrained stalks, marginalization restrictions, global sections, extension fibers); 2. specify B2 (assignment-sheaf variant: tau-consistency, minimal radius r*, Gaussian projection via Laplacian pseudoinverse); 3. define the fused estimator and the diagnostic statistic exactly, including the weighting across patterns; 4. state the degenerate-case handling (empty overlaps, single pattern).
- Outputs: `docs/formalization_v0.md`.
- Verification: a coder unfamiliar with the math can implement from it (WP1.5 doubles as the check).
- Gate consequence: feeds WP1.4 execution and all later phases.

#### WP1.4 Correspondence and content smoke test (THE decisive cheap test)
- Objective: determine whether the construction (i) reproduces known recoverability semantics and (ii) has any obstruction content.
- Actions: 1. on the transcription bank, check section-condition iff observed-data factorization, and fiber-constancy recoverability iff LP label (expect 100% agreement; investigate any mismatch); 2. deliberately demonstrate Hazard A on the constraint-free variant (H^1 = 0 always, confirming the hazard is real); 3. engineer 2-3 small instances with m-graph constraints and check whether any nonzero higher obstruction appears anywhere; 4. record which variants carry content.
- Outputs: `results/phase1/smoke_report.md` with witness tables.
- Verification (scientific): 100% agreement with ground truth on the bank; at least one variant exhibits non-degenerate behavior on the engineered instances (documented either way).
- Pass rule: full semantic agreement; content question answered (either answer is informative). Fail rule: any unexplained semantic mismatch after one documented repair iteration -> C1 dead as claimed -> evaluate pivots in order: (P-a) pure methods framing (C3/C4) without the correspondence claim, only if the user accepts a methods-only paper; (P-b) KILL.
- Compute: 1-2 days, local, negligible RAM.

#### WP1.5 Mini-library v0.1
- Objective: correct computational substrate (shared with the Ideas 1/5 toolbox where it exists).
- Actions: 1. implement `PatternPoset` (patterns, overlaps, inclusion order), `Sheaf` (constrained stalk parameterizations, restriction matrices), connection/sheaf Laplacian assembly per Hansen-Ghrist, dim H^0/H^1 via sparse solves, `radius()` (r* via pseudoinverse projection in the Gaussian case; generic optimizer elsewhere), `fuse()` (projection estimator); 2. property tests: functoriality of restrictions, Laplacian psd, hand-computed 3-pattern example with known answer, projection idempotence.
- Outputs: `src/sheafpatternfusion/{poset.py,sheaf.py,laplacian.py,radius.py,fuse.py}`, `tests/`.
- Verification: pytest green; hand-checked example matches.
- Pass rule: all properties hold. Fail rule: n/a (fix until green; nothing downstream runs otherwise).
- Dependencies: WP1.3. Parallel with WP1.4 once WP1.3 lands.

**Phase 1 gate evidence and decisions.**
- G0 GO: WP1.4 passes with full semantic agreement; content question resolved either way.
- G0 KILL/PIVOT: per WP1.4 fail rule. G1 GO requires WP1.1 clean. G2 GO requires WP1.5 green plus formalization sufficient to write Phase 2/3 code without further math decisions.
- Estimated effort: 2-3 weeks calendar, all local compute (<4 GB, <2 h per job).

---

### Phase 2: Exhaustive small-poset falsification of C1/C2 (target: 2-4 weeks; runs parallel with Phase 3 start)

**Purpose and scientific question.** Does the sheaf criterion agree with brute-force recoverability on every enumerated small m-graph, and does the obstruction content exist (genuine H^1-type phenomena), or is it uniformly degenerate? Gate G3a. This is the dossier's promised exhaustive verification.

**Prerequisites.** Phase 1 gates passed.

#### WP2.1 Enumeration infrastructure
- All m-graphs on 3-4 binary observable variables with binary missingness indicators, bounded latent cardinalities (cap 4 per latent), monotone and non-monotone pattern structures; estimated 10^3-10^4 structures x queries. Ground truth per instance via the WP1.2 LP engine. Outputs `configs/phase2/*.json`, `results/phase2/instances.jsonl` with schema `{instance_id, seed, structure, gt_recoverable, sheaf_verdict, obstruction_signature, ...}`.

#### WP2.2 Biconditional evaluation and habitat check
- Confusion matrix per structure class. Pre-registered thresholds: agreement >= 98%, unexplained mismatches <= 2%. Degeneracy check: fraction of instances with nonzero obstruction signature must lie strictly between 0% and 100% for the cohomology layer to have content. **Habitat check (Hazard N):** tabulate jointly the classes {conflicting patterns} x {phi recoverable}; if the intersection is empty across the entire enumeration (disagreement always forces unrecoverability of every tested functional), the fusion estimator has no legitimate habitat -> escalate to KILL review before Phase 4.

#### WP2.3 Poset-structure readout
- Verify the chain/tree closed-form claim (pairwise gluing suffices on interval posets) against enumeration; hunt for at least one general-poset instance where pairwise consistency holds but global consistency fails under constraints (genuine higher obstruction, ground-truth confirmed). Success feeds C2's statement; failure strips the cohomology vocabulary from the paper.

**Give-up rules (Phase 2).**
1. Unexplained mismatch rate > 2% after one debug round: C1/C2 dead as claimed -> demote correspondence to a remark; proceed to Phase 3/4 with methods-only framing if the user accepts it, else KILL.
2. Obstruction uniformly degenerate (0% or 100% nonzero across the grid): cohomology layer decorative -> strip it; the paper proceeds on C3/C4 alone or dies with them.
3. Habitat empty (Hazard N confirmed): the central methods premise is unidentifiable-in-practice -> KILL (record the enumeration as the witness).
4. LP ground truth exceeds feasible compute: scope honestly to tractable classes (monotone first) and say so; shard to Colab (anticipated; see Section 11).

Compute: the LP engine is the first heavy job. Pilot locally; projected total 20-80 CPU-hours -> **Colab sharding triggered** (est. 12-20 notebooks).

---

### Phase 2.5: Independent validation battery for C1/C2 (target: ~1 week calendar; runs parallel with Phase 3 start)

**Purpose and scientific question.** Phase 2 closed (2026-08-25) with a formal G3a PASS that an adversarial read reduces to implementation-consistency evidence: agreement was exactly 100%, but only on the 1202/3394 (35.4%) rows where the brute-force engine decided; the engine was UNDETERMINED on 64.6% of rows, and there the sheaf instrument asserted RECOVERABLE on 2191/2192 rows with no independent check. Because both instruments search the same fingerprint manifold, that mass is currently statistically indistinguishable from the free policy "always say RECOVERABLE." The main grid additionally produced ZERO cyclic realized-pattern posets, so the exhaustive-small-poset framing was never stressed by its own experiment, and the hand-built side obstructions re-derive classical phenomena (Frechet-Hoeffding bounds, PSD completion). Phase 2.5 therefore prices the certificate's marginal information using NON-SHARED oracles only, forces the cyclic stratum into existence, sizes the audit statistics honestly, and prices compute. Gates G2.5a-d. This phase exists because a false RECOVERABLE is actionable and kills C1 outright, while UNDETERMINED merely invites caution; validation effort must be allocated accordingly.

**Prerequisites.** Phase 2 merged artifacts frozen (`results/phase2/instances_merged.jsonl`, 3394/3394 rows, `COLLECT_REPORT.json` all-zero anomalies); library tag `v0.3.0` containing the battery modules below; the frozen merge committed to the repository (`data/frozen/instances_merged.jsonl`) so Colab runners self-fetch their sampling frames at pinned tag instead of relying on manual uploads.

#### WP2.5.1 Degeneracy null battery (CPU-light; 1 notebook)
- Population: all 2192 UNDETERMINED rows of `instances_merged.jsonl`. Null policies scored against the certificate's labels: (N0) constant RECOVERABLE; (N1) fraction-observed threshold sweep; (N2) pattern-overlap density; (N3) Frechet-width sign on the target mean interval (wide interval -> predict UNRECOVERABLE); (N4) constant UNRECOVERABLE control. Stratified by n in {2,3,4}.
- Pre-registered expectation and reading: N0 will reproduce >= 99.9% of labels by construction (the certificate is almost-all-RECOVERABLE there), so the headline metric is NOT raw agreement but the DISAGREEMENT SET: every row where the certificate contradicts the best null (chiefly N3) defines the priority sample S* for WP2.5.2. If S* is empty AND WP2.5.2 finds no false positives AND WP2.5.3 collapses, the certificate has demonstrated exactly zero marginal information in its claimed value region and C1/C2 demote to motivation-only (rule D1).
- Outputs: `results/phase25/null_battery.{json,csv}`, priority sample `results/phase25/priority_sample.jsonl`.

#### WP2.5.2 Adversarial audit of RECOVERABLE assertions (main compute; 6 Colab shards)
- Sampling frame and design (pre-registered): census of UNDETERMINED x RECOVERABLE rows at n=2 (all 62), census at n=4 (all 319), simple random sample N=400 at n=3, plus census of the priority set S* from WP2.5.1. Zero errors on a stratum of size M bounds its false-RECOVERABLE rate by 3/M (Chernyuk-style rule of three); with M >= 300 everywhere the pooled upper bound is <= 1%. Any CONFIRMED false RECOVERABLE anywhere -> C1 KILL (rule D2); this asymmetric tolerance is deliberate.
- Attackers must not reuse the fingerprint manifold: (A1) deepened witness search at x20 Phase 2 budgets (restarts, jump horizons, randomized objectives) hunting a model pair that differs on the target while matching observed laws; (A2) exact rational vertex/completion enumeration plus independent constructive completion samplers at fresh seeds, testing whether two completions disagree on the target; (A3) for the discrete layer, direct Frechet-cell certification on every triple of patterns adjacent through the target variable (classical route, deliberately non-sheaf).
- Every verdict logged with attacker identity, budget, and wall time (feeds WP2.5.6). Outputs: `results/phase25/audit_sample.jsonl`, `audit_verdicts.jsonl`.

#### WP2.5.3 Discordant-family construction (CPU-light; 1 notebook)
- Seed: the single observed instrument divergence `n3_s03759_d0` / target mean(0) (engine UNDETERMINED_RELAXED_FRAGILE; sheaf UNRECOVERABLE; observed family LP-incompletable, max cross-pattern gap 0.283; Jacobian rank 20/21). Parametrize its structure class and mechanism draw to grow a family; each member must carry BOTH (i) two distinct completions differing on the target, found by the independent samplers of WP2.5.2/A2, AND (ii) a classical witness where applicable (explicit Frechet-cell violation or LP dual infeasibility certificate).
- Success: >= 10 witnessed members -> the theory-note path greenlights post-adjudication (this becomes the demonstration that the certificate flags incompletability the engine cannot decide). Collapse to the singleton seed -> recorded as negative; weakens the theory path and strengthens demotion (feeds rule D1).
- Output: `results/phase25/discordant_family.jsonl`, `docs/discordant_family.md`.

#### WP2.5.4 Forced cyclic-poset stratum (Colab; 4 shards)
- The Phase 2 sampler never realized a cyclic `poset_shape`, so build the missing stratum directly: rejection-sample m-graph structures until the realized pattern poset is cyclic, seeded additionally by hand-constructed overlapping-pattern families known to induce cycles on 3-4 variables. Target >= 500 instances achieving cyclic realized posets (pilot-gated; honest-attempt floor 50, else rule D4).
- Run the identical Phase 2 pipeline (same engine budgets, same seed protocol) restricted to this stratum. Pre-registered: agreement >= 98% within-stratum, and nonzero obstruction signature fraction strictly inside (0%, 100%), finally exercising the WP2.2 degeneracy check that the main grid skipped.
- Kill rules scoped to this stratum mirror Phase 2 rules 1-2: unexplained mismatches > 2% after one debug round -> C1/C2 dead as claimed; uniformly degenerate obstruction signature -> cohomology vocabulary stripped and "exhaustive small-poset" retitled.
- Outputs: `configs/phase25/cyclic_grid.json`, `results/phase25/cyclic_instances.jsonl`, `results/phase25/cyclic_summary.json`.

#### WP2.5.5 Equivalence/inclusion memo (desk work, parallel; no compute)
- Prove or refute, on the binary m-graph class of Phase 2: fiber-spread-zero iff LP-width-zero wherever the engine decides (both directions). An equivalence theorem converts the shared-manifold circularity into mathematics (agreement stops being surprising and starts being a lemma); a proven divergence case is a novelty seed feeding WP2.5.3. Either resolution is publishable material; agnosticism is not.
- Source mapping (foundations to lean on): Frechet 1951 and Hoeffding 1940 for the pairwise marginal bound geometry; Grone et al. 1984 for positive-definite completion as the Gaussian-layer analogue whose obstruction theory the sheaf layer must reproduce or exceed; Manski 2003 for partial-identification bounds semantics; Daniel et al. 2012 and Mohan-Pearl 2013/2021 for the m-graph recoverability lineage the certificate claims to compress; Robinson 2018 for the assignment-sheaf consistency filtration the fiber instrument instantiates; Cheesman et al. 1991 cited ONLY as the cautionary analogy for why the 66->36->20% decidability collapse must NOT be narrated as a phase transition without a hardness parameter (three points on one engine's budget curve is an observation about the engine until proven otherwise).

#### WP2.5.6 Compute pricing (piggybacks on WP2.5.2 logs)
- Certificate cost per row vs deep-witness route per row, medians by n. Triage rationale requires the certificate >= 10x cheaper than the search it spares; otherwise the applied story dies even if the mathematics lives (recorded in the gate memo regardless of G2.5b).

**Gates and consequences (pre-registered).**
- G2.5a (degeneracy): quantified always; demotion trigger is the joint condition in WP2.5.1/D1, not raw agreement.
- G2.5b (audit): zero confirmed false RECOVERABLE across censused and sampled strata -> C1 survives with a defensible error-rate bound; any confirmation -> C1 KILL, program proceeds methods-only on C3 (per Phase 3 default) with the audit published as the boundary marker.
- G2.5c (cyclic stratum): agreement >= 98% AND non-degenerate obstructions -> C2 keeps H^1 vocabulary and the "small-poset" title earns back its meaning; otherwise strip per D4.
- G2.5d (family): >= 10 witnessed discordant members -> standalone theory note authorized AFTER G2.5b returns clean; fewer -> fold into limitations.
- Interaction with Phase 3: Phase 3 starts NOW in parallel (it shares no code path with the certificate, so none of this gates it); Phase 2.5 outcomes only change framing, demotion, and the theory annex. If Phase 3 later hits its KILL default, a clean G2.5b+G2.5c+G2.5d is precisely the "live correspondence contribution" that overrides the default.

**Give-up/demotion rules (Phase 2.5).**
1. D1 (demotion): S* empty, audit clean, family collapsed -> C1/C2 demoted to motivation; ship C3 alone or die with it.
2. D2 (KILL of C1): any confirmed false RECOVERABLE in the audit -> certificate dead as a decision instrument; publish the counterexample either way.
3. D3 (theory halt): equivalence memo proves divergence impossible on this class AND family failed -> higher-obstruction vocabulary stripped, memo filed as a lemma.
4. D4 (stratum failure): honest generator attempt yields < 50 cyclic-realized instances OR within-stratum agreement < 98% after one debug round -> retitle the falsification claim and strip the cohomology layer per Phase 2 rule 2.

Compute: local machine excluded (reserved for concurrent experiments). All shards are thin-run Colab notebooks pinning `v0.3.0`: `nb25_00_nullbattery` (minutes), `nb25_audit_shard_00..05` (6 x <= 10 h at 2 cores, pilot first per Section 11 policy), `nb25_cyclic_shard_00..03` (4 x <= 10 h), `nb25_family` (< 2 h). Twelve notebooks total; each self-contained, embedding its frozen job manifest, resuming via JSONL key dedup, auto-downloading outputs with a safe fallback. Projected total 30-60 CPU-hours.

**Outcome (2026-08-26; full detail in `docs/phase25_report.md`, raw data in `results/phase25/`).**
- G2.5a QUANTIFIED: constant-RECOVERABLE null reproduces 99.95% of certificate labels on the 2192 undecided rows; corrected Frechet widths large everywhere (medians ~0.50); S* = 200 rows (nonempty).
- G2.5b PASS: 0 confirmed false RECOVERABLE in 899/901 audited rows (one instance, n4_r0199_d0, pending a single-instance patch notebook); pooled rule-of-three bound <= 0.33%; A2 maximal model-valid fiber spread 1.4e-10 vs tolerance 1e-4. CAVEAT recorded: with ~1 UNRECOVERABLE commitment per 2192 rows the audit has almost no discriminating power; cleanliness is weak evidence of value.
- G2.5c PASS: forced cyclic stratum realized (560 instances, 1120 target rows, all Berge-cyclic); decidable-row agreement 100% (336 TN, FP=FN=0); obstruction signature 30% strictly inside (0%,100%); observed families LP-incompletable on 100%; classical Frechet certification on 0/899 audit rows.
- G2.5d COLLAPSE: 3/120 witnessed discordant members (threshold >= 10); origin seed itself not witnessed; theory-note path closed, folded into limitations.
- WP2.5.5 equivalence memo remains OPEN (desk work).
- WP2.5.6 pricing FAILS the >=10x bar where it matters: attack/certificate pipeline ~1.03x at n=3, 6.0x at n=2 (apples-to-apples on undecided rows). Applied triage narrative dead as pre-registered.
- Demotion rules: D1 NOT triggered (S* nonempty), D2 not triggered, D4 not triggered (140/shard vs floor 50). Net: C1 survives statistically but is demoted economically and informationally per the Amendment in Section 1; C2's cyclic-obstruction asset is real but its natural prevalence is untested -> motivates WP3.0 below.

---

### Phase 3: Pivot-gate week (WP3.0) and re-aimed fusion benchmark C3 (target: 1 gate week + 3-4 weeks benchmark; supersedes the original unconditional Phase 3)

**Purpose and scientific question (amended 2026-08-26 after council review of Phase 2.5).** The original Phase 3 pitch leaned on the certificate layer for its differentiating mechanism; Phase 2.5 measured that mechanism at ~zero on small binary m-graphs (marginal information ~1/2192 over a constant policy) and priced it at ~1x the deep search. Running the original 3-4-week benchmark without new justification is the zombie scenario the portfolio rules exist to prevent; abandoning the program outright ignores the one verified-live asset (cyclic-stratum obstruction content that classical tools provably cannot certify). Phase 3 therefore opens with a ONE-WEEK PIVOT GATE (WP3.0, three cheap decisive probes, pre-registered thresholds, hard >=2/3 rule), and only behind that gate runs a RE-AIMED benchmark concentrated where the surviving theory predicts wins.

**Prerequisites.** Phase 1 gates passed (done). Phase 2/2.5 artifacts frozen (done). C1 demoted to motivation per Amendment; no further certificate investment except inside WP3.0 probes.

#### WP3.0 Pivot-gate week (NEW; three parallelizable probes + hard gate; target <= 7 days)

##### WP3.0a Natural-prevalence scan of cyclic missingness (~3 days; CPU-light)
- Question: does the obstruction phenomenon live anywhere practitioners actually are? Every cyclic instance so far was FORCED by construction; if natural data never realizes cyclic overlap structure, C2 has no audience and the re-aimed benchmark has no home stratum.
- Design: >= 8 public datasets with substantive missingness (NHANES cycles/subscales, UCI adult-style sets, any Kaggle/social-survey set with >= 5% missing); for each dataset and each variable subset of size 3-6 with >= 4 realized patterns, build the observed-set overlap hypergraph and test Berge-cyclicity (existing `graham_acyclic`); record cyclic fraction per dataset and overall, plus whether cyclic subsets involve partial (not nested-only) overlaps.
- Pre-registered GO: cyclic realized pattern-structure appears in >= 3 independent datasets OR >= 15% of eligible subsets pooled. NO-GO: cyclic structures are artifacts of our generator only.
- Output: `results/phase3/prevalence_scan.{json,csv}`.
- Compute: minutes-hours, local (read-only dataset access per Section 9 early-feasibility rules).

##### WP3.0b Scaling probe: silence rate and cost crossover at n=5-6 (~3 days; 1-2 Colab shards)
- Question: both headline failures (65% engine silence; attack/certificate ~1x) were measured at n<=4. Exact-search cost grows combinatorially while LP/fiber costs stay mild; does the picture invert just past the tested range?
- Design: extend the enumeration sampler to n=5 (n=6 if pilot says tractable): sample >= 400 structures x 1 draw; run engine round1+round2 unchanged; run attackers at fixed budget on undecided x RECOVERABLE rows only; log wall times split engine/LP/fiber/attack.
- Pre-registered readings (recorded regardless): engine decidability rate at n=5 vs n=4; attack/cert cost ratio trend in n. GO-for-benchmark-feasibility: decidability >= 50% at n=5 (labels obtainable on harder instances). GO-for-certificate-economics: ratio >= 3x AND strictly increasing from n=4 to n=5. Either GO feeds the gate; both NO-GO recorded as confirmation that the certificate has no accessible regime at any testable scale.
- Output: `results/phase3/scaling_probe.{jsonl,json}`.
- Compute: est. 20-60 CPU-hours -> 1-2 thin-run Colab notebooks (v0.3.0 pattern).

##### WP3.0c Signal-validity probe: does ANY sheaf-side feature predict ground truth? (~3 days; piggybacks on WP3.0b compute)
- Question: on rows carrying REAL labels, do sheaf features beat chance? Note the circularity trap: among decided rows, fiber spread trivially separates verdicts (it IS the verdict input). The honest test lives where labels are expensive: fresh undecided rows at n>=5, plus the forced-cyclic stratum, labeled by the full attacker battery.
- Design: pool all labeled rows (3394-row merge + 1120 cyclic rows + WP3.0b fresh rows). Labels: engine-decided verdicts where available; attacker-found unrecoverability for undecided rows. Features: corrected Frechet width, Jacobian rank deficiency, max cross-pattern marginal gap, fraction observed, overlap density (all already implemented). Metrics: AUC of each feature (and a small logistic combo) for predicting UNRECOVERABLE among undecided-labeled rows, against TWO null baselines: (i) label-permutation within matched strata, (ii) features computed on random-m-graph matches. Downstream add-on: correlation of spread with naive-pooling error across DGP draws.
- Pre-registered GO: AUC >= 0.75 on attacker-labeled undecided rows (permutation p < 0.01) OR downstream |rho| >= 0.30 with p < 0.01. NO-GO: no feature clears chance once the trivial separation is excluded -> certificate-class instruments carry zero information at accessible scales, permanently closing the correspondence track (C2's obstruction characterization may still stand on WP3.0a).
- Output: `results/phase3/signal_validity.{json,csv}`.

##### Gate G2.6 (hard rule)
- PROCEED to WP3.1'/WP3.2' iff AT LEAST 2 of {WP3.0a, WP3.0b(feasibility arm or economics arm), WP3.0c} return GO. Ties broken toward proceeding ONLY when WP3.0c is one of the GOs (signal validity is non-negotiable: without it there is nothing for the estimator or the paper to lean on).
- FAIL (< 2 GO, or the tie-break fails): TERMINATE the program as originally framed. Write the boundary paper from existing assets: degeneracy result (constant-policy equivalence at 99.95%), audited error bound (<= 0.33%), cyclic obstruction characterization with classical-blindness witness (Frechet certifies 0/899). Venue: workshop/UAI-short. No Phase 3 spend. This is the pre-committed anti-zombie exit.

#### WP3.1' Gaussian closed-form engine, Hazard E characterization, and the WRITTEN WIN-PREDICTION (gate to the benchmark)
- As original WP3.1 (fused estimator via sheaf-Laplacian pseudoinverse projection; bias expression under pattern-specific shifts; explicit minimum-distance equivalence class), PLUS the council-mandated addition: a one-to-two-page WRITTEN PREDICTION, committed before any benchmark cell runs, naming (i) the exact instance class/DGP family where the fused estimator beats tuned MI (expected: contaminated-pattern and cyclic-stratum cells, with the restriction structure doing the work generic MD cannot), (ii) the predicted margin band, and (iii) what observed result would falsify the prediction. No prediction document -> no WP3.2' runs. Output `results/phase3/hazard_E_memo.md` + `results/phase3/win_prediction.md`.

#### WP3.2' Re-aimed benchmark grid
- Estimators unchanged: fused projection; complete-case; IPW; MI (normal-model and MICE-style, tuned per protocol); pattern-mixture baseline; naive pooling; oracle.
- DGPs re-weighted by the evidence: PRIMARY axis = contaminated-pattern and cyclic-informed regimes (single pattern shifted; patterns drawn from forced-cyclic structures; shift localized where the restriction structure differs from naive pooling), since these are where the surviving theory predicts separation and where baselines lack guidance. GUARD cells kept unchanged: MAR null (all tie within MC error), baseline-favorable (CC/MI should win), MNAR sweep, crossover grid (shift x n x dimension x pattern-count).
- Metrics and thresholds unchanged and non-negotiable: RMSE on E[Y] and ATE, coverage where intervals exist, worst-decile error; >= 100 replicates for decisive cells; frozen seed lists; fused beats complete-case AND tuned MI simultaneously over >= 25% of the primary-axis grid; never worse than best baseline by > 10% anywhere; null-cell ties within MC error; empirical dominance boundary matches the WP3.1' prediction within stated tolerance.
- Reporting duty added post-2.5: every table carries the silence-rate context (what fraction of each DGP cell the ground-truth engine could label), so readers see label provenance explicitly.

**Give-up rules (Phase 3, amended).**
1. Fused estimator worse than the strongest simple baseline across ALL regimes including its design-favorable one: methods contribution dead -> KILL. Rescue clause AMENDED: the old "live correspondence contribution" override now requires G2.5b+G2.5c clean (already true) AND WP3.0c GO; absent that, default KILL stands and the boundary paper ships alone.
2. Wins confined to a fragile sliver (< 5% of the grid) or erased by modest misspecification/tuning sensitivity: INCREMENTAL-ONLY -> terminate per portfolio bar; archive artifacts, write a short diagnostic.
3. Estimator requires unavailable information (true shift magnitudes, true m-graph beyond stated assumptions) with no working data-driven replacement after one repair round: treat as fail rule 1 or 2 depending on severity.
4. Null cell shows spurious fused-estimator gains beyond MC error: implementation bug or leakage; halt, fix, rerun before reading any other cell.
5. NEW: the written win-prediction of WP3.1' must exist and pre-date the first benchmark run; retro-fitted predictions void the gate and count as fail rule 2 severity.

Compute: WP3.0 week ~20-80 CPU-hours total (mostly local + 1-2 Colab shards). WP3.1' local days. WP3.2' thousands of small jobs; pilot first; mostly local; reserve up to ~10 Colab notebooks for MI-heavy high-dimensional cells if projected > 2 h or > 4 GB.

---

### Phase 4: Diagnosis C4 + scaling + real-data smoke test (target: 3-4 weeks; DORMANT UNTIL PHASE 3 GATE)

**Purpose and scientific question.** Is r* a calibrated, powerful, localizing test of pattern disagreement where pairwise statistics are blind; does everything scale to honest pattern counts; and does the machinery run meaningfully on real data? Gates G3c and G4a.

**Prerequisites.** Phase 3 gate passed (a surviving estimator to diagnose around). Phase 2 findings used as framing only.

#### WP4.1 Consistency-radius test battery
- Null: size control at alpha = 0.05 (+/- 0.01 achieved) under correct m-graphs including MAR regimes; bootstrap calibration validated (null-quantile coverage). Power: planted contamination grid (which pattern shifted, shift magnitude, number of jointly shifted patterns). Localization: top-1 identification of the contaminated pattern. Competitors at matched information: Little's MCAR test, overlap-wise chi-square/KS batteries, MAR-vs-MNAR graphical diagnostics.
- Pre-registered: power >= competitor best + 10 percentage points in the multi-pattern joint-disagreement regime (where pairwise tests are structurally blind); top-1 localization >= 70% in that regime; size within tolerance everywhere. Otherwise C4 is cut from the paper (non-terminal if C3 lives).

#### WP4.2 Scaling study
- Pattern-count growth (up to 2^|V| naive), monotone and coarse-patterned regimes, runtime/memory curves, sparse-Laplacian behavior at realistic dimension; identify practical limits; feeds the honest scoping section. Local pilot; shard to Colab if projected > 2 h or > 4 GB (reserve <= 8 notebooks).

#### WP4.3 Real-data smoke test
- NHANES subscale or UCI adult-style data with substantial missingness; end-to-end pipeline: pattern poset, r*, fused estimate beside CC/IPW/MI; read-only feasibility permitted from Phase 2 onward (access, schema, pattern frequencies), confirmatory run here with preprocessing frozen beforehand; validation by reproducing one trusted published complete-case/MI estimate on the same data; negative control: permute pattern labels (r* must collapse to noise level); drop-one-pattern sensitivity; interpretation boundaries stated.
- Give-up rule: pipeline crashes or produces numerically meaningless diagnostics on real scales after one documented fix attempt -> report honestly; the paper proceeds simulation-only (weaker, viable).

**Give-up rules (Phase 4).**
1. Diagnostic has no power/localization edge at matched size: cut C4; proceed if C3 alive, else KILL.
2. C4 dead AND Phase 3 returned INCREMENTAL-ONLY or KILL: nothing above the bar remains -> KILL.
3. Real-data failure: non-terminal per WP4.3 rule.

---

### Phase 5: Evidence-earned theory and paper (DORMANT UNTIL PHASE 4 GATE)

**Purpose.** Prove only what surviving evidence makes worth proving; assemble the smallest coherent paper. Gates G5, G6.

Theory targets (each mapped to a source result, with a numerical falsification hook already implemented in earlier phases):

| Target | Why it matters (evidence link) | Sketch | Tag | Source result to build on | Stop rule |
|--------|-------------------------------|--------|-----|---------------------------|-----------|
| Th-A factorization correspondence (C1) | Interpretation and anchoring of everything; demanded by the m-graph community | Observed-data factorization iff section condition; recoverability iff fiber constancy; induction over the pattern poset (sheaf axiom made precise) | direct/adaptation | Mohan-Pearl-Tian 2013 factorization; Mohan-Pearl JASA 2021 theorems; sheaf axiom (Hansen-Ghrist Sec. 2) | Numerical hook: WP2.2 confusion tables. If it needs new hypotheses beyond the construction, state as scoped proposition |
| Th-B poset characterization (C2) | Only if WP2.3 found genuine obstruction content | Pairwise-gluings-suffice on interval posets; exhibited general-poset counterexample instance | adaptation | Standard Cech vanishing on intervals; Robinson poset-sheaf techniques (arXiv:2012.00120) | Counterexample search precedes proof (scripted in WP2.1); abandon on first genuine counterexample to the positive claim |
| Th-C Gaussian dominance band (C3) | Only if WP3.2 passed | Bias expression under pattern shifts; dominance-band characterization; projection = regularized MD with m-graph-selected restrictions (Hazard E made precise and turned into a feature) | adaptation | Hansen-Ghrist pseudoinverse projection formulas; minimum-distance/GMM combination theory; Kearney-Palmowski-Robinson minimum-radius bounds | If the band conditions never occur in the Phase 3 grid, downgrade to conjecture with numerics or cut |
| Th-D r*-test validity (C4) | Only if WP4.1 passed | Asymptotic size control; bootstrap consistency for the radius statistic | adaptation/conjecture | Bootstrap theory for nonstandard statistics; empirical-process tools [exact sources selected in Phase 5] | If calibration was achieved purely empirically and asymptotics resist, ship the bootstrap procedure validated by simulation, labeled honestly |

Paper skeleton decision at G6: methods-led (C3+C4) with Th-A as the grounding proposition; C2 included only if WP2.3 delivered. Venue fit assessed against recent UAI/CLeaR/Journal of Causal Inference programs, with statistics-methods outlets as the stretch tier depending on Th-C/Th-D strength. Referee simulation must answer the Section 5 item 6 paragraph and the "tuned MI suffices" attack. Reproducibility audit: seeds, configs, notebook regeneration, one-command figures.

Give-up rules: if during any theorem a genuine counterexample appears that earlier numerics missed, halt proofs, rerun the owning falsification hook, and reopen the owning phase gate; no theorem is kept that contradicts recorded evidence. If G5 lands INCREMENTAL-ONLY (valid but routine), the project terminates per portfolio bar; archive and write a short diagnostic.

---

## 8. Simulation study specification

Claims-to-experiments matrix (decisive rows first):

| Claim | Mechanism | DGP | Metric | Baseline | Ablation | Threshold | Falsifier | Output |
|-------|-----------|-----|--------|----------|----------|-----------|-----------|--------|
| C1/C2 correspondence | section/fiber criterion mirrors m-graph semantics | enumerated binary m-graphs, 3-4 vars | agreement %, unexplained mismatch % | LP ground truth + published verdicts | constraint-free sheaf (expected degenerate) | agreement >= 98%, unexplained <= 2% | systematic mismatch bands | `results/phase2/enumeration.csv` |
| C2 content | constraints create genuine obstruction | engineered + enumerated instances | % instances nonzero obstruction | constraint-free ablation | - | strictly between 0% and 100% | uniformly 0% or 100% | `results/phase2/content.csv` |
| Habitat (N) | conflict-compatible recoverability exists | enumerated structures x queries | |class(conflict, recoverable)| > 0 | LP labels | - | nonempty intersection | empty intersection | `results/phase2/habitat.csv` |
| C3 fusion | projection removes incompatible component | Gaussian MAR/MNAR/contamination grids; PRIMARY axis post-2026-08-26: contaminated-pattern + forced-cyclic-informed regimes (behind G2.6 gate) | RMSE vs oracle; dominance %; worst-decile | tuned MI, CC, IPW, PMM, pooling | no-projection control; equal-weight pooling | dominance >= 25% of MNAR band; never lose > 10%; null ties | uniform loss | `results/phase3/fusion.csv` |
| C3 band | T3 predicts the dominance boundary | shift sweep | boundary match | analytic prediction | - | within stated tolerance | boundary mismatch | `results/phase3/band.csv` |
| C4 r* test | joint disagreement detection + localization | planted contamination grid | power at matched size; top-1 localization | Little, pairwise chi-square/KS | r* on wrong poset | +10 pp power; localization >= 70%; size ok | no separation | `results/phase4/diagnosis.csv` |
| Null calibration | no free lunches | MAR null, clean patterns | size; FP; RMSE ties | all | - | size <= alpha + tol | inflated size / fake gains | `results/phase{3,4}/null.csv` |

Fair comparison protocol: matched information sets (same inputs fed to every method), tuning budgets registered before runs (MI variants get honest tuning effort; no strawmen), seed lists frozen per phase (`configs/seeds.txt`), failed runs logged with reasons, >= 100 replicates for decisive cells, one primary metric per row declared above. Gate memos written before moving phases (`results/phaseN/gate_memo.md`).

## 9. Applied smoke-test protocol (lite)

Early feasibility (read-only, allowed from Phase 2 onward): dataset access, schema, pattern frequencies, effective per-pattern n; no outcome-pattern inspection that leaks into design. Confirmatory (WP4.3): preprocessing frozen before comparative numbers, reproduction of one trusted published estimate as pipeline validation, negative control (pattern-label permutation collapses r*), drop-one-pattern sensitivity, interpretation boundaries. This is a smoke test, not a flagship application; "the machinery runs on real data and produces readable, sane diagnostics" is success at this stage.

## 10. Deferred theory program

Covered by the Phase 5 table (Th-A to Th-D). Ordering principle: highest decision value first (Th-A always, since it anchors semantics; Th-C secures the headline if C3 leads; Th-B only if content exists; Th-D only if the diagnostic survived). Every target names its source result and a falsification hook already coded in Phases 2-4; special-case reductions and counterexample searches precede any long proof, per portfolio rule 7.

## 11. Compute, Colab sharding policy, and reproducibility

**Local-first rule.** Every experiment starts as a one-seed local pilot recording wall time and peak RSS. Full runs stay local if projected < 2 h wall time AND < 4 GB peak memory: 12-16 process-level workers (machine: i9-13900H, 10 physical cores / 20 threads; leave capacity for other ongoing experiments), one BLAS thread per worker, no nested parallelism, RAM-capped well below machine limit, JSONL checkpoint per completed instance (resume-safe by construction).

**Colab sharding rule.** Any experiment projected > 2 h wall time OR > 4 GB peak RAM is split into independent, self-contained Google Colab notebooks; **up to 40 notebooks are available**, each running independently (roughly 2-core CPU, ~12 GB RAM each; GPU irrelevant here). Shard budget allocation: Phase 2 up to 20 (priority), Phase 3 up to 10 (reserve), Phase 4 up to 8 (reserve); total capped at 40. Each notebook embeds: pinned dependencies, its shard config and seed list, full generation logic (no reliance on other notebooks' runtime state), periodic checkpoint appends to its output CSV/JSONL, and ends with:

```python
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
```

Sharded outputs land in `results/<phase>/shards/` and are concatenated by `scripts/collect_shards.py` with row-count/checksum completeness checks before any gate memo cites them.

Anticipated triggers: Phase 2 LP enumeration (wall-time driven, est. 12-20 notebooks); Phase 3 MI-heavy cells (conditional); Phase 4 bootstrap sweeps (conditional). Phases 1 and 5 stay local.

**Artifact map.**

```
/home/hugo_souto/Stuff/Research/sheafpatternfusion/
  src/sheafpatternfusion/   poset.py sheaf.py laplacian.py radius.py fuse.py mdag_dgp.py lp_ground_truth.py gaussian_ground_truth.py battery.py attackers.py cyclic_synth.py discordant_family.py workers.py (v0.3.0 Phase-2.5 modules)
  tests/                    property + ground-truth + hand-checked-example tests
  configs/                  examples/ phase2/ phase3/ phase4/ seeds.txt
  docs/                     formalization_v0.md evidence_register.md smoke_report.md hazard_E_memo.md
  results/                  phase1..4 jsonl/csv outputs + gate memos
  notebooks_colab/          shard_NN.ipynb templates + collect_shards.py
  refs.bib                  verified bibliography (DOI/URL, flagged gaps commented)
```

Environment lock (`requirements.txt` with versions) frozen at end of Phase 1; every figure/table regenerable by one command per phase from configs and seeds.

## 12. Risk register

| Risk | Prob. | Damage | Earliest detector | Prevention/recovery | Terminal? | Package |
|------|-------|--------|-------------------|---------------------|-----------|---------|
| Hazard A: obstruction uniformly degenerate | Medium-high | Strips theory layer | WP1.4 witnesses, WP2.3 | Strip vocabulary; proceed on C3/C4 | No (paper survives smaller) | WP1.4, WP2.3 |
| Hazard C: tuned MI/PMM dominate fusion | Medium | Kills headline method | WP3.2 grid | Honest dominance-band reporting; pivot emphasis to C4 | Conditional | WP3.2 |
| Hazard E: estimator = generic minimum distance | Medium-high | Weakens novelty framing | WP3.1 memo | Characterize precisely; sell m-graph-selected restrictions + robustness + C4 | No (framing damage) | WP3.1 |
| Hazard N: helpful regime empty (conflict forces unrecoverability) | Low-medium | Kills methods premise | WP2.2 habitat table | Scope to approximate recoverability if a defensible version exists | Yes | WP2.2 |
| Direct hit: Robinson group applies consistency radius to statistical missingness | Low-medium | Kills novelty | Monthly alert (WP1.1) | Move fast; differentiate on m-graph semantics + estimators | Yes | WP1.1 |
| Direct hit: m-graph community adopts algebraic/quantitative layer | Low-medium | Partial novelty loss | Monthly alert | Differentiate on fusion estimator + diagnosis | Partial | WP1.1 |
| LP ground truth intractable at scale | Medium-high | Delays Phase 2 | WP2.1 pilot | Monotone-class scoping; Colab sharding | No | WP2.1 |
| Tuning leakage / strawman baselines | Low-medium | Invalidates G3 | Protocol registration | Matched info sets, registered budgets, honest MI tuning | No | WP3.2 |
| Real-data smoke failure | Medium | Weaker paper | WP4.3 | Fix once; else simulation-only paper, stated honestly | No | WP4.3 |
| Decorative theory temptation | Medium | Wasted months, diffuse paper | G5 discipline | Phase 5 dormant until Phase 4 gate | No | Phase 5 |
| Diffuse story (theory + two method bets) | Medium | Rejects at strong venues | G6 referee simulation | Methods-led skeleton; theory earns paragraphs via Phase 3/4 results | No | Phase 5 |
| Certificate marginally vacuous + uneconomic (REALIZED P2.5) | Resolved->fact | C1 demoted; applied triage story dead | WP2.5.1 battery, WP2.5.6 pricing | Demote C1 to motivation (Amendment 2026-08-26); program re-gated behind WP3.0 probes | Partially (program continues only through G2.6) | WP3.0 |
| Cyclic-stratum asset has no natural habitat | Medium-high | Kills C2 audience and the re-aimed benchmark's home stratum | WP3.0a prevalence scan (3 days) | If NO-GO: boundary paper exit per Appendix A row 2.75 | Yes (for this framing) | WP3.0a |

## 13. Immediate actions (stop at the next unresolved gate)

Status: items 1-5 below were the Phase-1 program and are COMPLETE (G0/G1/G2 all GO; Phases 2, 2.5 also complete). The current immediate actions are:

1. **WP3.0a** natural-prevalence scan (3 days, local) -> `results/phase3/prevalence_scan.{json,csv}`.
2. **WP3.0b+c** scaling + signal-validity probes (<= 7 days, 1-2 Colab shards at tag `v0.3.0`) -> `results/phase3/scaling_probe.*`, `signal_validity.*`.
3. Gate **G2.6** adjudication memo (`results/phase3/gate_G26_memo.md`): >=2/3 GO proceeds to WP3.1'; otherwise boundary-paper exit per Appendix A.
4. Only after G2.6 GO: WP3.1' hazard-E memo plus the written win-prediction; then WP3.2'.

Original Phase-1 list (historical):

1. WP1.1: run the remaining query families; upgrade/downgrade the Section 4 table; draft `refs.bib`; set watch alerts. Output: `docs/evidence_register.md`.
2. WP1.2: implement the LP recoverability engine + Gaussian ground truth; transcribe the textbook bank; tests green. Output: `src/sheafpatternfusion/lp_ground_truth.py`, `tests/test_ground_truth.py`.
3. WP1.3: write `docs/formalization_v0.md` covering B1/B2 (about four pages, no proofs).
4. WP1.4: run the correspondence/content smoke test against the bank. Output: `docs/smoke_report.md`. This resolves Gate G0.
5. In parallel once WP1.3 lands: WP1.5 library core with property tests.

Nothing beyond the current gate is scheduled until G2.6 returns its verdict; Phase 3' and everything downstream remain conditional branches.

## 14. References (verification status as of Aug 2026)

Verified today (abstract/metadata inspected via arXiv API):
- Robinson, Szulczewski, Thorson. Analyzing the Topological Structure of Composite Dynamical Systems. arXiv:2511.04603, 2025. https://arxiv.org/abs/2511.04603 (nearest neighbor; different problem, watched)
- Scott, Valdano, Assaad. Missing Data and Cluster Graphs: Cluster-Level vs Variable-Level Missingness. arXiv:2605.20943, 2026. https://arxiv.org/abs/2605.20943 (active m-graph community evidence)
- Idrissova, Rekik. Multimodal Sheaf-based Network for Glioblastoma Molecular Subtype Prediction. arXiv:2508.09717, 2025. https://arxiv.org/abs/2508.09717 (vocabulary neighbor only)
- Hansen, Ghrist. Toward a Spectral Theory of Cellular Sheaves. arXiv:1808.01513; journal DOI 10.1007/s41468-019-00038-7. https://arxiv.org/abs/1808.01513
- Kearney, Palmowski, Robinson. Sheaf-Theoretic Framework for Optimal Network Control. arXiv:2012.00120. https://arxiv.org/abs/2012.00120

Known-primary, to verify in WP1.1 (flagged in dossier and/or needing exact locator):
- Rubin. Inference and missing data. Biometrika 63(3), 1976, DOI 10.1093/biomet/63.3.581.
- Mohan, Pearl, Tian. Graphical Models for Inference with Missing Data. NeurIPS 2013 (proceedings URL to record).
- Mohan, Pearl. Graphical Models for Processing Missing Data. JASA 2021, DOI 10.1080/01621459.2021.1872461 [digits to verify].
- Robinson. Sheaves are the canonical data structure for sensor integration. Information Fusion 2017 [venue/volume to verify].
- Little. A test of missing completely at random for multivariate data with missing values. JASA 1988 [to verify].
- van Buuren et al. MICE (reference + package URL to record); sklearn IterativeImputer docs.
- IPW missing-data lineage (Robins-Rotnitzky-Scharfstein) and pattern-mixture literature (Daniels-Hogan) [exact refs to select].
- Shpitser-lineage semiparametric identification with missing data [exact ref to verify].

Phase 2.5 additions (foundations for the validation battery; entries mirrored in `refs.bib` with locators):
- Frechet. Sur les tableaux de correlation dont les marges sont donnees. Ann. Univ. Lyon, Sect. A 14(3), 1951 [no DOI; scanned copies exist via persee/archive.org — verify pagination before citing in the paper].
- Hoeffding. Masstabinvariante Korrelationstheorie. Skand. Aktuarietidskr. 23, 1940 [no DOI; German; verify pages].
- Grone, Johnson, Sa, Wolkowicz. Positive definite completions of partial Hermitian matrices. Linear Algebra Appl. 58, 1984, DOI 10.1016/0024-3795(84)90207-6.
- Manski. Partial Identification of Probability Distributions. Springer, 2003, DOI 10.1007/978-1-4757-3639-7.
- Daniel, Kenward, Cousens, De Stavola. Using causal diagrams to guide analysis of missing data problems. Stat. Methods Med. Res. 21(3), 2012, DOI 10.1177/0962280210394469.
- Cheesman, Kanefsky, Taylor. Where the Really Hard Problems Are. IJCAI-91 [proceedings URL to record; cited only as the cautionary phase-transition analogy].

---

## Appendix A: Consolidated give-up rules (quick reference)

| Phase | Give up / reroute when |
|-------|------------------------|
| 1 | Direct-hit prior art (KILL); unexplained correspondence mismatch on the canonical bank after one repair round and no accepted methods-only reroute (KILL); library not green (halt everything) |
| 2 | Unexplained mismatches > 2% after one debug round (C1/C2 demoted; methods-only reroute or KILL); obstruction uniformly 0% or 100% (strip cohomology layer); habitat table empty, i.e., the estimand is effectively unidentifiable whenever patterns conflict (KILL) |
| 2.5 | D1 demotion: priority set empty AND audit clean AND discordant family collapsed (C1/C2 -> motivation-only); D2 KILL of C1: any confirmed false RECOVERABLE in the audit (publish the counterexample either way); D3 theory halt: divergence proven impossible on this class and family failed (file memo as lemma, strip higher-obstruction vocabulary); D4 stratum failure: < 50 cyclic-realized instances honestly attempted OR cyclic-stratum agreement < 98% after one debug round |
| 2.75 (WP3.0 pivot gate, G2.6) | Fewer than 2 of {prevalence scan GO, scaling probe GO, signal-validity probe GO} — or 2 GOs without signal-validity among them — (TERMINATE original program; ship boundary paper from P2.5 assets: degeneracy result + <=0.33% audited bound + cyclic obstruction characterization with classical-blindness witness) |
| 3 | Fused estimator worse than the strongest simple baseline in ALL regimes including its design-favorable one (KILL unless G2.5b+G2.5c clean AND WP3.0c GO, default KILL); wins only a fragile < 5% sliver (INCREMENTAL-ONLY, terminate); spurious null-cell gains (halt, fix, rerun); benchmark run without the pre-committed written win-prediction (treated as INCREMENTAL-ONLY severity) |
| 4 | Diagnostic no power/localization edge at matched size (cut C4; KILL if C3 also gone); real-data failure (non-terminal: simulation-only paper) |
| 5 | Genuine counterexample to a targeted theorem (halt proofs, rerun owning gate); G5 INCREMENTAL-ONLY (terminate per portfolio bar) |

Overall terminal conditions, any phase: the target is shown unidentifiable/unrecoverable in every regime where the method could help; the proposed method is dominated by benchmark models in every tested regime including its design-favorable one; or verified prior art absorbs the contribution.
