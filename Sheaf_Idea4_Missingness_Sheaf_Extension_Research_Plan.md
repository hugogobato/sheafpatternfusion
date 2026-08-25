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
| C1 (T1) | The m-graph observed-data factorization (Mohan-Pearl-Tian) is equivalent to the section condition of the canonically built pattern sheaf; recoverability of phi equals constancy of phi on the fiber of global extensions | enabling/grounding | High | Low-medium (translation hazard) | Necessary: anchors all semantics | UNTESTED | Keep as grounding proposition; prove minimally in Phase 5 only after numerics confirm |
| C2 (T2) | On chains/trees of patterns, pairwise gluing suffices (closed form); on general posets there exist genuine higher obstructions (pairwise-consistent families that fail triple consistency under m-graph constraints) | contribution (theory layer) | Uncertain (Hazard A: obstruction may never fire) | Medium | Medium | UNTESTED | Keep through Phase 2; strip vocabulary if degenerate |
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
P1 De-risk & enable (G0 validity, G1 novelty, G2 implementability)      [ACTIVE]
   -> P2 Enumeration falsification of C1/C2 (G3a)                       [ACTIVE after P1]
   -> P3 Fusion benchmark C3 (G3b)                                      [ACTIVE after P1; parallel-safe with P2]
        -> P4 Diagnosis C4 + scaling + real-data smoke (G3c, G4a)       [ACTIVE after P3]
             -> P5 Evidence-earned theory + paper (G5, G6)              [DORMANT UNTIL P4 gate]
```

Parallelism note: P2 (representation theory falsification) and P3 (estimation benchmark) consume disjoint Phase 1 outputs and kill independently; running them concurrently wastes nothing if either dies. P4 waits on P3 (diagnosis is only worth building around a surviving estimator) and uses P2's obstruction findings as framing only.

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

### Phase 3: Robust fusion benchmark C3 (target: 3-4 weeks; starts after Phase 1, parallel with Phase 2)

**Purpose and scientific question.** Does the projection-based fused estimator beat honest strong baselines (complete-case, tuned MI, IPW, pattern-mixture) in a characterizable band, and does the predicted dominance boundary match the T3 bias analysis? Gate G3b. This is the primary methods bet.

**Prerequisites.** Phase 1 gates passed. (Does not consume Phase 2 outputs.)

#### WP3.1 Gaussian closed-form engine and Hazard E characterization
- Implement the fused estimator via sheaf-Laplacian pseudoinverse projection; derive and code the bias expression under pattern-specific shifts; **explicitly characterize the equivalence class**: state for which configurations the projection coincides with minimum-distance pooling of pattern moments, and identify what differs (restriction selection from the m-graph; weighting; behavior off the consistent manifold). Output `results/phase3/hazard_E_memo.md`. If the estimator is *always* exactly generic MD with no distinguishing behavior, record it: the methods novelty then rests entirely on the robustness band and C4.

#### WP3.2 Benchmark grid
- Estimators: fused projection; complete-case; IPW (missingness weights); MI (normal-model and MICE-style, tuned per protocol); pattern-mixture baseline; naive pooling; oracle. DGPs: (i) MAR null (all estimators tie within MC error); (ii) baseline-favorable (one pattern clean and dominant, CC/MI should win); (iii) MNAR pattern-shift sweep (the predicted dominance band); (iv) contaminated-pattern (single pattern's law shifted, others clean); (v) crossover grid: shift magnitude x n x dimension x pattern-count. Metrics: RMSE on E[Y] and ATE, coverage where intervals exist, worst-decile error across replicates; >= 100 replicates for decisive cells; frozen seed lists.
- Pre-registered thresholds: fused beats complete-case AND tuned MI simultaneously over >= 25% of the shift grid in the MNAR band; never worse than the best baseline by more than 10% anywhere on the grid; null-cell gaps within MC error; empirical dominance-boundary matches the WP3.1 bias-expression prediction within stated tolerance.

**Give-up rules (Phase 3).**
1. Fused estimator worse than the strongest simple baseline across ALL regimes including its design-favorable one (the user's headline kill condition): methods contribution dead -> KILL unless Phase 2 delivered a live correspondence contribution worth a standalone theory note (user decides; default KILL).
2. Wins confined to a fragile sliver (< 5% of the grid) or erased by modest misspecification/tuning sensitivity: INCREMENTAL-ONLY -> terminate per portfolio bar; archive artifacts, write a short diagnostic.
3. Estimator requires unavailable information (true shift magnitudes, true m-graph beyond stated assumptions) with no working data-driven replacement after one repair round: treat as fail rule 1 or 2 depending on severity.
4. Null cell shows spurious fused-estimator gains beyond MC error: implementation bug or leakage; halt, fix, rerun before reading any other cell.

Compute: thousands of small jobs; pilot first. Mostly local; reserve up to ~10 Colab notebooks for MI-heavy high-dimensional cells if projected > 2 h or > 4 GB.

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
| C3 fusion | projection removes incompatible component | Gaussian MAR/MNAR/contamination grids | RMSE vs oracle; dominance %; worst-decile | tuned MI, CC, IPW, PMM, pooling | no-projection control; equal-weight pooling | dominance >= 25% of MNAR band; never lose > 10%; null ties | uniform loss | `results/phase3/fusion.csv` |
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
  src/sheafpatternfusion/   poset.py sheaf.py laplacian.py radius.py fuse.py mdag_dgp.py lp_ground_truth.py gaussian_ground_truth.py
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

## 13. Immediate actions (stop at the next unresolved gate)

1. WP1.1: run the remaining query families; upgrade/downgrade the Section 4 table; draft `refs.bib`; set watch alerts. Output: `docs/evidence_register.md`.
2. WP1.2: implement the LP recoverability engine + Gaussian ground truth; transcribe the textbook bank; tests green. Output: `src/sheafpatternfusion/lp_ground_truth.py`, `tests/test_ground_truth.py`.
3. WP1.3: write `docs/formalization_v0.md` covering B1/B2 (about four pages, no proofs).
4. WP1.4: run the correspondence/content smoke test against the bank. Output: `docs/smoke_report.md`. This resolves Gate G0.
5. In parallel once WP1.3 lands: WP1.5 library core with property tests.

Nothing beyond Phase 1 is scheduled until G0 (and G1/G2) return GO; Phases 2-5 remain conditional branches.

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

---

## Appendix A: Consolidated give-up rules (quick reference)

| Phase | Give up / reroute when |
|-------|------------------------|
| 1 | Direct-hit prior art (KILL); unexplained correspondence mismatch on the canonical bank after one repair round and no accepted methods-only reroute (KILL); library not green (halt everything) |
| 2 | Unexplained mismatches > 2% after one debug round (C1/C2 demoted; methods-only reroute or KILL); obstruction uniformly 0% or 100% (strip cohomology layer); habitat table empty, i.e., the estimand is effectively unidentifiable whenever patterns conflict (KILL) |
| 3 | Fused estimator worse than the strongest simple baseline in ALL regimes including its design-favorable one (KILL unless a live Phase 2 theory contribution stands, default KILL); wins only a fragile < 5% sliver (INCREMENTAL-ONLY, terminate); spurious null-cell gains (halt, fix, rerun) |
| 4 | Diagnostic no power/localization edge at matched size (cut C4; KILL if C3 also gone); real-data failure (non-terminal: simulation-only paper) |
| 5 | Genuine counterexample to a targeted theorem (halt proofs, rerun owning gate); G5 INCREMENTAL-ONLY (terminate per portfolio bar) |

Overall terminal conditions, any phase: the target is shown unidentifiable/unrecoverable in every regime where the method could help; the proposed method is dominated by benchmark models in every tested regime including its design-favorable one; or verified prior art absorbs the contribution.
