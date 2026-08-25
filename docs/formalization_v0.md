# Formalization v0 — Pattern-Poset Sheaf, Consistency Radius, and Fusion Estimator

Project `SheafPatternFusion`, WP1.3 (enabling formalization only; no proofs). Status: frozen for Phase 1 implementation. Everything here is specified precisely enough to code; nothing here is claimed as a theorem. References: Mohan-Pearl-Tian 2013 (factorization semantics), Mohan-Pearl 2021 (recoverability), Hansen-Ghrist 2019 (sheaf Laplacian, Hodge decomposition), Robinson 2018 (assignments to sheaves of pseudometric spaces, tau-consistency, minimal radius).

## 0. Notation

Variables V = {1, ..., n}; in Phase 1 all variables are binary. A pattern is r in {0,1}^n with r_i = 1 meaning V_i is observed. O(r) = {i : r_i = 1} is the observed set and M(r) = V \ O(r) the missing set. The full data law P lives on (V_1,...,V_n,R_1,...,R_n); the observed law of pattern r is q_r(o) = P(V_{O(r)} = o | R = r), a probability table on {0,1}^{O(r)}. An m-graph G has vertices V ∪ R; each R_i may have arrows from variables (and, if MNAR, from its own variable) and no outgoing arrows except to its own variable's column convention (we use the standard m-graph reading: R_i is a child of its causes; V_i's distribution does not depend on any R_j). The missingness mechanism factorizes as P(R | V) = prod_i P(R_i | pa_G(R_i)).

The pattern poset: P_G = set of patterns realizable under G (positivity assumed), ordered by r ⊑ r' iff O(r) ⊆ O(r'). Joins are unions of observed sets when that union is itself a realized pattern; where a join is not realized, the poset simply lacks the element and no restriction between the two stalks exists. Phase 1 assumes the comparability graph used below is connected on realized patterns (stated assumption; empty-overlap instances are out of scope by scoping, not by failure).

## 1. B1: the pattern-poset sheaf of constrained law families

Stalks. CORRECTION (2026-08-24, discovered by the WP1.4 smoke test): stalk objects are the mass-carrying pattern tables W_r(o) = P(V_O(r) = o, R = r), not the normalized conditionals q_r(o) = W_r(o)/W_r(total). Reason: two patterns condition on different events, so conditional laws do NOT restrict to each other by plain marginalization; the mass-carrying tables do (completion-sum marginalization is linear and matches the MPT observed-data factorization cell-by-cell). Conditional laws remain available inside each stalk as W_r divided by its total mass. For a pattern r, the stalk F(r) is the set of mass-carrying tables whose normalized conditionals satisfy every conditional independence implied by G restricted to O(r). Two instantiations are coded:

1. Discrete instantiation: F(r) = mass-carrying tables W_r over {0,1}^{O(r)} whose normalized conditionals satisfy the declared CI constraints exactly, where each constraint is a list (X, Y, Z) of disjoint subsets of O(r) requiring X independent of Y given Z. Feasibility of an assignment against constraints is checked by exact evaluation on the normalized table.
2. Gaussian instantiation: F(r) = multivariate normal laws N(mu, Sigma) on R^{O(r)} such that Sigma restricted to pairs (i,j) with a declared marginal-CI constraint (X={i}, Y={j}, Z=empty) has zero covariance entry Sigma_ij, and conditional-CI constraints (X={i}, Y={j}, Z nonempty) have zero partial correlation rho_{ij·Z} computed from the submatrix on {i,j} ∪ Z.

Restriction maps. If r ⊑ r' then rho_{r'r}: F(r') -> F(r) is completion-sum marginalization of the mass-carrying tables onto O(r) (the map that sends W_{r'} to W_r of any full law inducing it). Marginalization preserves both instantiations' constraint classes (discrete: CI statements survive marginalization of unrelated coordinates only when the conditioning set is retained; therefore the stalk constraints of F(r) are required to be exactly those CIs of G whose variables all lie in O(r), which is what makes rho well defined as a map into F(r)). Functoriality (rho composes) holds by construction of marginalization.

Sections. A section s assigns to each realized pattern r an element s_r ∈ F(r) such that for every comparable pair r ⊑ r', rho_{r'r}(s_{r'}) = s_r. A family of observed laws (q_r)_r arises from a full-data law P factorizing w.r.t. G iff (q_r)_r is a section of this sheaf; this equivalence is the Phase-1 working statement of correspondence claim C1 and is checked mechanically in WP1.4, not proven.

Extension fiber and recoverability. Fix an assignment a = (a_r) with a_r ∈ F(r) (typically a_r = true observed law q_r, or an estimate). Ext(a) = set of global sections s with s|_r = a_r for all r (exact extension). For a functional phi of the full-data law (e.g., E[V_i], E[V_i | V_j = v], a contrast), define phi(s) by applying phi to the unique completion implicit in s (for discrete sections, phi reads off the glued table; for Gaussian sections, mu and Sigma). Then: "phi is recoverable at a" means phi is constant on Ext(a); "phi recoverable under G" means phi constant on Ext(a) for every realizable a. This is the operational form of C1's second half.

Degenerate-case handling. Single-pattern posets: Ext(a) contains the trivial section; fusion returns a_r unchanged and r* = 0. Empty overlaps: restrictions do not exist between incomparable-with-empty-intersection stalks; the construction requires the overlap graph connected, else report "fusion undefined" rather than silently pooling.

## 2. B2: assignment-sheaf consistency filtration (pseudometric variant)

Following Robinson 2018, equip each stalk with a pseudometric d_r (Phase 1: total variation on discrete tables; Euclidean distance on concatenated mean vectors for Gaussians). An assignment a = (a_r) is tau-consistent if for every face of the poset (every finite chain), the sum of edge distances along the face is ≤ tau. The minimal radius of a is

r*(a) = inf over global sections g of max over faces sum of d along face edges,

which in the vertex-poset, single-edge formulation reduces to r*(a) = min_g max_r d_r(a_r, g|_r) when distances are combined per-face additively; we implement the standard two-term version on Hasse edges: cost(g) = max over Hasse edges (r ⊑ r') of w_e · ||a_{r'} − rho_{r'r}(g_{r'})|| ... concretely we implement the quadratic functional below, which is what the Gaussian projection solves exactly:

J(g) = sum over Hasse edges e=(r,r') of w_e * || S_e g − a_e ||^2, minimized subject to g being a global coordinate vector.

Gaussian unconstrained case (the coded closed form): collect unknown global moments into x ∈ R^D (means of all variables; covariances handled by the same linear algebra after vectorization, Phase 3). Each pattern r contributes observation block b_r (its estimated mean vector on O(r)) and selection matrix S_r embedding O(r)-coordinates into R^D. Then J(x) = sum_r w_r ||S_r x − b_r||² and the minimizer is x* = (Σ_r w_r S_rᵀ S_r)^+ Σ_r w_r S_rᵀ b_r, i.e., a weighted least-squares projection computable via the sheaf Laplacian L = deltaᵀ W delta with delta the coboundary on the Hasse diagram (Hansen-Ghrist). With CI-constrained stalks, x* is additionally projected onto the constraint set by alternating projection (zero-out constrained covariances, re-project means); this alternating step is exact for marginal-CI (zero-covariance) constraints and iterative but monotone for conditional-CI constraints.

Definitions fixed for coding:

- r*_quad(a) = sqrt( J(x*) ) using the weights above. This is THE diagnostic statistic.
- Fused estimator: given estimated pattern summaries (b̂_r, weights ŵ_r = n_r / Σ n), compute x̂ = argmin J and return phi applied to x̂ (e.g., mean of Y = component μ_Y of x̂).
- Diagnostic test: bootstrap-resample within each pattern (stratified by pattern, B resamples), recompute r*_quad on each resample to get its null-ish sampling distribution; calibrated threshold τ̂_{1-α} = empirical (1-α) quantile under the MAR-null-fitted model (resample from the pooled/global-section fit rather than raw data, so resampling mimics noise around consistency, which is the null). Reject "patterns consistent" when r*_quad(data) > τ̂_{1-α}.
- Localization score for pattern r: contribution c_r = w_r ||S_r x̂ − b̂_r||² / Σ_r' w_r' ||S_r' x̂ − b̂_r'||², plus leave-one-out drop Δ_r = r*_quad(full) − r*_quad(without r). Contaminated pattern := argmax c_r (ties broken by Δ_r). Both are reported; top-1 accuracy is the Phase 4 metric.

## 3. What Phase 1 code must satisfy (acceptance mapping)

1. `poset.py`: build P_G from a list of patterns; comparability, joins where realized; Hasse diagram. Property: order is a partial order; Hasse covers verified against brute-force transitive reduction on small inputs.
2. `sheaf.py`: discrete and Gaussian stalks with CI-constraint lists; marginalization restrictions; section check; extension enumeration for small discrete cases (enumerate completions consistent with all restrictions). Properties: functoriality rho_{r''r'} ∘ rho_{r'r} = rho_{r''r}; restriction outputs remain in target stalk.
3. `laplacian.py`: assemble delta and L = deltaᵀW delta for the Gaussian mean-coordinate sheaf. Properties: L psd; harmonic subspace dimension = dim ker L equals number of connected components of the comparability graph; Hodge-style split verified numerically (im(Lᵀ) ⊥ ker L).
4. `radius.py`: r*_quad via pseudoinverse projection; bootstrap calibration wrapper. Property: hand-computed 3-pattern example matches analytic answer.
5. `fuse.py`: fused estimator + localization contributions. Properties: projection idempotence (fusing already-consistent input changes nothing beyond float tolerance); single-pattern identity.
6. Ground-truth engines (`lp_ground_truth.py`, `gaussian_ground_truth.py`, `mdag_dgp.py`): see WP1.2 spec in the research plan and module docstrings. Key honesty note: unrecoverability certificates come from the assumption-free LP relaxation (sound one way: a witness pair valid without mechanism assumptions remains valid under them); recoverability verdicts come from exact constructive formulas (MAR/MCAR identities) or exhaustive model-aware witness search failing to find a counterexample within budget, recorded as such.

## 4. Scope boundaries stated up front

No proofs in this document. Conditional-CI (partial-correlation) handling is iterative in B2's projection step; only zero-marginal-covariance constraints are exact there. The general-poset higher-obstruction question is deferred to WP1.4/WP2.3 as an empirical content check, not asserted either way here.
