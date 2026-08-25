# Phase 1 Gate Memo

Date: 2026-08-24. Scope: WP1.1 through WP1.5 of the research plan. All work local, well under compute caps.

## Gate verdicts

**G0 (fatal viability): GO.**
WP1.4 passed with full semantic agreement: engine verdicts 11/11 and sheaf-fiber verdicts 11/11 against expected labels on the transcription bank; slice validity 1.0; no unexplained mismatch (none required the single repair iteration the fail rule allows). Content question resolved both ways as informative: constructive H^1 = 0 for the linear mean-coordinate sheaf (200/200), certified higher obstruction for covariance-valued stalks via exact PSD certificate with a feasible control, provenance of the obstruction refined (moment geometry, not CI constraints). Evidence: `docs/smoke_report.md`, `results/phase1/*`.

**G1 (novelty): GO (standing, monthly watch).**
WP1.1 clean at arXiv level: six query families zero direct hits; two watch papers downgraded after full-text reading (Robinson et al.: missing-data = per-observation stalks plus radius minimization in DSEMs, no MCAR/MAR/MNAR semantics; Scott et al.: purely graphical cluster-abstraction recoverability). Reference corrections recorded (Mohan-Pearl JASA 2021 DOI ends 1874961; Little 1988 and RRZ 1994 DOIs located; Robinson 2017 Information Fusion verified). Residual: Semantic Scholar/Google Scholar sweep scheduled as first Phase 2 action. Evidence: `docs/evidence_register.md`, `refs.bib`.

**G2 (implementability): GO.**
WP1.5 green: 33 tests passing (`tests/test_library.py`, `tests/test_ground_truth.py`), covering poset order axioms, Hasse-vs-bruteforce, restriction functoriality, discrete section checks, CI constraint detection, Laplacian PSD + harmonic dimensions, hand-computed star example (analytic optimum), projection idempotence, single-pattern identity, localization on a redundant triangle, DGP-vs-analytics consistency, Gaussian closed form vs Monte Carlo with MC-error-scaled tolerances, and all bank verdicts. The formalization was sufficient to write Phases 1-2-facing code without further math decisions, and it absorbed three mechanical corrections without redesign (evidence that the abstraction is stable).

## Work package completion

| WP | Deliverables | Status |
|---|---|---|
| WP1.1 | `docs/evidence_register.md`, `refs.bib` | done |
| WP1.2 | `src/sheafpatternfusion/{mdag_dgp,lp_ground_truth,gaussian_ground_truth}.py`, `configs/examples/*.yaml` (8 instances), `tests/test_ground_truth.py` | done |
| WP1.3 | `docs/formalization_v0.md` (updated with dated corrections) | done |
| WP1.4 | `scripts/run_smoke.py`, `docs/smoke_report.md`, `results/phase1/*` | done, G0 resolved |
| WP1.5 | `src/sheafpatternfusion/{poset,sheaf,laplacian,radius,fuse}.py`, `tests/test_library.py` | done |

## Deviations from plan (documented per protocol)

1. Ground-truth engine is LP-plus-witnesses rather than pure LP: unrecoverability carries model-valid witness pairs from multistart root-jumping (and SLSQP), while the LP relaxation alone is reported honestly as VARIABLE_UNCONSTRAINED_ONLY because the relaxation exceeds the model class. This strengthens, not weakens, the certificates.
2. C1 restated before proof: stalks carry mass-carrying tables W_r = P(V_O, R=r); conditional-law families do not glue directly. The MCAR characterization lives on the population-marginal (marginal-sheaf) reading. Phase 5's Th-A must be stated in these terms.
3. Hazard A answer is a refinement, not a yes/no: linear sheaf obstruction-free (constructive), full-law sheaf obstructed by moment geometry independent of m-graph constraints, discrete zero-CI triangles escape via product law. WP2.3 should enumerate richer constraint patterns instead of hunting triangle-PSD shapes.
4. Two instance slots were redesigned after engines contradicted their initial labels (x5 twice, x8 once); full history in the smoke report. No label was adjusted to fit an engine output without a structural explanation.

## Carry-forward obligations for Phase 2

- Google Scholar / Semantic Scholar sweep first (WP1.1 residual).
- Bank grows by transcribed MPT13 figure-level instances when full texts are available locally; current labels are principle-level citations plus engine certification.
- Enumeration must include mechanism structures of the surprising identified classes (mutual selection, double self-censoring) to map the identification boundary, since Hazard N's habitat question depends on it.
- Environment frozen now: `requirements.txt` committed at repo root.
