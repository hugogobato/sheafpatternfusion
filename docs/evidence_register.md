# Evidence register — SheafPatternFusion (Phase 1, WP1.1)

Updated: 2026-08-24. Verification levels: **E1** = abstract/metadata inspected; **E2** = beyond abstract (full text or substantive sections inspected); **E3** = independently cross-verified (publisher/Crossref metadata matched to claimed content).

## A. New query families run today (arXiv API, abs-level, max_results=25, sorted by date desc)

| Query family | Hits | Direct hit? | Consequence |
|---|---|---|---|
| `abs:"imputation" AND abs:"sheaf"` | 0 | none | novelty gap holds |
| `abs:"cohomology" AND abs:"missing data"` | 0 | none | novelty gap holds |
| `abs:"topological" AND abs:"missing data"` | 0 | none | novelty gap holds |
| `abs:"consistency radius"` | 5 | none (physics radii x2; aggregation sheaves 2106.04445; KPR 2012.00120; Robinson 1805.08927 — all known/tool papers) | added 2106.04445 + 1805.08927 to watch list |
| `abs:"pattern mixture" AND abs:"sheaf"` | 0 | none | novelty gap holds |
| `abs:"missingness" AND abs:"poset"` | 0 | none | novelty gap holds |

No unexamined direct hit remains on arXiv. G1 status: clean at arXiv level. Semantic Scholar / Google Scholar sweeps deferred to Phase 2 opening (recorded as residual risk LOW-MEDIUM; two live watch items below).

## B. Load-bearing prior-art rows upgraded

| Source | Level | What was checked | Finding vs our claims |
|---|---|---|---|
| Robinson, Szulczewski, Thorson, arXiv:2511.04603 | **E2** (full HTML text read; all "missing"-context passages extracted) | Their Definition 14: missing data handled by adding one preorder element per observation above each variable; missing values simply excluded from the support of the initial assignment; radius minimization then infers missing observations, path coefficients, AR coefficients | Missing-data treatment = stalk-splitting + consistency-radius-minimization imputation inside DSEM netlist models. NO MCAR/MAR/MNAR semantics, no recoverability theory, no estimands, no pattern structure, no diagnostic calibration. Table 4 row stands; direct-hit risk unchanged (Medium, monitor monthly) |
| Scott, Valdano, Assaad, arXiv:2605.20943 | **E2** (full HTML text read; Theorem 1 statement + Section 5 headers extracted) | Theorem 1: necessary and sufficient graphical condition (no cluster variable adjacent to its R-vertex, nor connected through collider-only paths of missing/observed clusters) for recovering the joint over clusters; closed-form recovery expression via Markov blankets; macro causal effects section | Purely graphical recoverability under cluster-level abstractions (m-C-DMG / cm-C-DMG). No topology, no quantitative disagreement layer, no fusion estimator, no diagnostics. Confirms m-graph community expanding along abstraction axes, not algebraic ones |
| Idrissova, Rekik, arXiv:2508.09717 | E1 | Abstract re-fetched today | Vocabulary neighbor only ("missing modalities", deep reconstruction); no statistical identification content |

## C. Flagged references resolved (all now E3 via Crossref/publisher resolution)

| Reference | Status | Verified locator |
|---|---|---|
| Rubin 1976, Inference and missing data | VERIFIED | DOI 10.1093/biomet/63.3.581 (Crossref: Biometrika, 1976) |
| Little 1988, MCAR test | VERIFIED (plan had no DOI) | DOI 10.1080/01621459.1988.10478722 (JASA 83(404)) |
| Mohan-Pearl JASA 2021 | VERIFIED, **DOI digits CORRECTED** | Correct DOI: **10.1080/01621459.2021.1874961**. Plan's "…1872461" does not resolve; do not cite it |
| Mohan-Pearl-Tian 2013, NeurIPS | VERIFIED | https://papers.nips.cc/paper_files/paper/2013/hash/0ff8033cf9437c213ee13937b1c4c455-Abstract.html |
| Robinson 2017, Sheaves are the canonical data structure for sensor integration | VERIFIED (venue/volume confirmed) | Information Fusion; DOI 10.1016/j.inffus.2016.12.002 |
| Hansen-Ghrist spectral sheaves | VERIFIED | arXiv:1808.01513; journal DOI 10.1007/s41468-019-00038-7 |
| Kearney-Palmowski-Robinson | VERIFIED (E1 re-check today) | arXiv:2012.00120 |
| Daniels-Hogan 2008 book | VERIFIED | DOI 10.1201/9781420011180 |
| Robins-Rotnitzky-Zhao 1994 IPW lineage | VERIFIED (**DOI corrected**) | DOI 10.1080/01621459.1994.10476818 (JASA 89(427)) |
| Tsiatis 2006, Semiparametric Theory and Missing Data | VERIFIED | DOI 10.1007/0-387-37345-4 (semiparametric anchor; exact Shpitser-lineage paper still to be selected in Phase 2, see refs.bib comment) |
| van Buuren & Groothuis-Oudshoorn 2011 (mice) | VERIFIED | DOI 10.18637/jss.v045.i03 (JSS 45(3)) |

Bonus find (added to bibliography): Mohan & Pearl 2022, "Graphical Models for Recovering Probabilistic and Causal Queries from Missing Data", DOI 10.1145/3501714.3501739 (extended release of the UAI-magisterial tutorial material; useful as an accessible recoverability reference).

## D. Watch items (monthly alert obligation)

1. Robinson group (owns consistency-radius toolkit): arXiv listing alerts on author names + "consistency radius"; includes new finds 2106.04445 (aggregation sheaves), 1805.08927 (assignments to sheaves of pseudometric spaces — note: this is the canonical citation for the tau-consistency machinery we build on in B2; added to refs.bib).
2. m-graph community (owns semantics): Scott/Valdano/Assaad line, Mohan/Shpitser line, m-C-DMG successors.

## E. Pass/fail rule check (WP1.1)

Pass rule "no unexamined direct hit remains": **PASS at arXiv level** (six families zero hits; two watch papers downgraded from threat after full-text reading). Residual: Google Scholar/Semantic Scholar sweep scheduled as first action of Phase 2 (no KILL trigger fired).

G1 verdict after WP1.1: **GO** (provisional → standing, subject to monthly watch).

## F. Phase-2 opening sweep (WP1.1 residual, 2026-08-24)

| Channel | Query families | Result |
|---|---|---|
| Crossref REST API | `"cellular sheaf" missingness`; `sheaf "missing data" recoverability`; `"consistency radius" survey`; `"pattern poset" sheaf` | Zero direct hits across all four. Top-ranked items are generic MI literature, sheaf-theory texts, and unrelated homonyms (neutron-star radii, permutation-pattern posets). The only recoverability item is Mohan-Pearl-lineage SSRN 2343873, already the known incumbent semantics |
| Semantic Scholar API | (same families) | **Rate-limited (HTTP 429) on every attempt**, including spaced retries; not swept. Rescheduled; residual risk unchanged (LOW-MEDIUM) |
| arXiv API re-run | composite queries | API unreachable from current network (empty responses); Phase-1 six-family sweep stands as the arXiv evidence |

Conclusion: no new direct hits in anything accessible today; novelty gap holds at the level accessible from this environment. G1 remains **GO** on standing watch.
