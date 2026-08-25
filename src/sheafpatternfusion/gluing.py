"""Gluing and obstruction instruments (WP2.3).

Discrete layer: mass-carrying tables W_r on realized patterns; the section
condition is cover-agreement (linear marginalization); global completion is
the existence of ONE full table T >= 0 whose slice onto each O(r) equals
W_r. Stalk CI constraints restrict which family members are admissible but
do not enter the gluing maps, so obstruction EXISTENCE for a poset is a
property of the linear slice system alone; constraints only gate which
families arise as observed data. Both facts are exploited here:
  - scan_poset_discrete samples mutually-consistent families and certifies
    (in)completability by LP (HiGHS), hunting genuine higher obstructions
    (pairwise-consistent, globally infeasible);
  - acyclic overlap hypergraphs are predicted completable (Graham test).

Gaussian layer: covariance-valued stalks on antichain/cyclic posets carry
assigned pairwise correlations on unit margins; global completion is PSD
feasibility of the partial correlation matrix. min-eigenvalue maximization
over free entries yields exact certificates (Phase-1 style).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog, minimize


# --------------------------------------------------------------------------
# discrete marginal problem
# --------------------------------------------------------------------------

def full_cells(n_vars: int) -> list[tuple]:
    return list(itertools.product((0, 1), repeat=n_vars))


def slice_marginal(T: dict[tuple, float], r: tuple[int, ...]) -> dict[tuple, float]:
    """Marginalize a FULL-variable-keyed table T onto O(r)."""
    idx = [i for i in range(len(r)) if r[i] == 1]
    out: dict[tuple, float] = {}
    for v, c in T.items():
        key = tuple(v[i] for i in idx)
        out[key] = out.get(key, 0.0) + c
    return out


def slice_marginal_dense(tab: dict[tuple, float], r: tuple[int, ...],
                         keep_vars: tuple[int, ...]) -> dict[tuple, float]:
    """Marginalize a DENSE-keyed table (keys over O(r)) onto keep_vars."""
    pos = {i: k for k, i in enumerate(j for j in range(len(r)) if r[j] == 1)}
    out: dict[tuple, float] = {}
    for o, c in tab.items():
        key = tuple(o[pos[i]] for i in keep_vars)
        out[key] = out.get(key, 0.0) + c
    return out


def marginal_problem_lp(n_vars: int, family: dict[tuple, dict[tuple, float]]) -> dict:
    """Feasibility of T on {0,1}^n with slice(T, O(r)) == family[r] for all r.

    Returns {feasible, status}. LP variables: cells of T (2^n); equality rows:
    total mass plus one row per family cell."""
    cells = full_cells(n_vars)
    nc = len(cells)
    cell_index = {v: k for k, v in enumerate(cells)}
    rows, rhs = [], []
    rows.append(np.ones(nc))
    rhs.append(1.0)
    for r, tab in family.items():
        idx = [i for i in range(n_vars) if r[i] == 1]
        for o, mass in tab.items():
            row = np.zeros(nc)
            for v in cells:
                if tuple(v[i] for i in idx) == tuple(o):
                    row[cell_index[v]] = 1.0
            rows.append(row)
            rhs.append(float(mass))
    A_eq, b_eq = np.array(rows), np.array(rhs)
    bounds = [(0.0, 1.0)] * nc
    res = linprog(np.zeros(nc), A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs")
    return {"feasible": bool(res.status == 0), "status": res.status}


def sample_family(patterns: list[tuple[int, ...]], rng: np.random.Generator,
                  concentration: float = 1.0) -> dict[tuple, dict[tuple, float]]:
    fam = {}
    for r in patterns:
        k = sum(r)
        raw = rng.gamma(concentration, 1.0, size=2 ** k)
        raw = raw / raw.sum()
        keys = list(itertools.product((0, 1), repeat=k))
        fam[r] = {kk: float(v) for kk, v in zip(keys, raw)}
    return fam


def mutually_consistent(family: dict[tuple, dict[tuple, float]],
                        tol: float = 1e-10) -> bool:
    """Agreement of every pair of family tables on their shared observed set
    (as MASS tables, including totals on the overlap)."""
    pats = list(family.keys())
    for i, a in enumerate(pats):
        for b in pats[i + 1:]:
            shared = tuple(j for j in range(len(a)) if a[j] == 1 and b[j] == 1)
            if not shared:
                continue
            ma = slice_marginal_dense(family[a], a, shared)
            mb = slice_marginal_dense(family[b], b, shared)
            if set(ma) != set(mb):
                return False
            for k in ma:
                if abs(ma[k] - mb[k]) > tol:
                    return False
    return True


def cover_consistent(poset_covers: list[tuple[tuple, tuple]],
                     family: dict[tuple, dict[tuple, float]],
                     tol: float = 1e-10) -> bool:
    for small, big in poset_covers:
        if small not in family or big not in family:
            continue
        shared = tuple(j for j in range(len(big))
                       if big[j] == 1 and small[j] == 1)
        if not shared:
            continue
        pushed = slice_marginal_dense(family[big], big, shared)
        small_dense_vars = tuple(j for j in range(len(small)) if small[j] == 1)
        target = slice_marginal_dense(family[small], small, small_dense_vars)
        for o, c in pushed.items():
            if abs(c - target.get(o, 0.0)) > tol:
                return False
    return True


def sample_pair_family(patterns: list[tuple[int, ...]],
                       rng: np.random.Generator) -> dict[tuple, dict[tuple, float]]:
    """Constructive sampler of MUTUALLY CONSISTENT families on pair-stalk
    antichains: draw singleton margins once, realize each pair table inside
    its Fréchet bounds with a uniform association parameter. Every such
    family is mutually consistent by construction."""
    n_vars = len(patterns[0])
    m = rng.dirichlet(np.ones(2), size=n_vars)
    fam: dict[tuple, dict[tuple, float]] = {}
    for r in patterns:
        idx = [i for i in range(n_vars) if r[i] == 1]
        assert len(idx) == 2, f"pair-stalk sampler got pattern {r}"
        i, j = idx
        pi, pj = m[i][1], m[j][1]
        lo = max(0.0, pi + pj - 1.0)
        hi = min(pi, pj)
        span = max(hi - lo - 2e-6, 0.0)
        t = lo + 1e-6 + rng.random() * span
        cells = {(1, 1): t, (1, 0): pi - t, (0, 1): pj - t,
                 (0, 0): 1.0 - pi - pj + t}
        assert min(cells.values()) > -1e-12
        fam[tuple(r)] = cells
    return fam


def scan_poset_discrete(patterns: list[tuple[int, ...]], n_families: int = 40,
                        seed: int = 0, sampler: str = "pair") -> dict:
    """Sample mutually-consistent families, LP-test global completability,
    count genuine higher obstructions (mutually consistent yet globally
    infeasible)."""
    rng = np.random.default_rng(seed)
    n_vars = len(patterns[0])
    n_tested = 0
    n_feasible = 0
    witness = None
    for _ in range(n_families):
        if sampler == "pair":
            fam = sample_pair_family(sorted(patterns), rng)
        else:
            fam = sample_family(sorted(patterns), rng)
            if not mutually_consistent(fam):
                continue
        res = marginal_problem_lp(n_vars, fam)
        n_tested += 1
        if res["feasible"]:
            n_feasible += 1
        elif witness is None:
            witness = {"".join(map(str, r)): {",".join(map(str, o)): float(round(float(m), 6))
                                              for o, m in tab.items()}
                       for r, tab in fam.items()}
    return {
        "patterns": [list(p) for p in sorted(patterns)],
        "sampler": sampler,
        "n_families_tested": n_tested,
        "n_globally_feasible": n_feasible,
        "n_obstructed": n_tested - n_feasible,
        "witness": witness,
    }


# --------------------------------------------------------------------------
# Gaussian (covariance-stalk) PSD-completion certificates
# --------------------------------------------------------------------------

def psd_completion_min_eig(k: int, assigned: dict[tuple[int, int], float],
                           n_starts: int = 8, seed: int = 0) -> dict:
    """Max over free entries of the minimal eigenvalue of the correlation
    matrix with prescribed entries. Positive optimum => completable (explicit
    glue returned); negative even at optimum => certified obstruction."""
    free_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)
                  if (i, j) not in assigned]

    def min_eig_of(x_free):
        C = np.eye(k)
        for (i, j), rho in assigned.items():
            C[i, j] = C[j, i] = rho
        for (i, j), val in zip(free_pairs, x_free):
            C[i, j] = C[j, i] = val
        return float(np.min(np.linalg.eigvalsh(C))), C

    def obj(x):
        return -min_eig_of(x)[0]

    rng = np.random.default_rng(seed)
    best_val, best_C = -np.inf, None
    x_init = np.zeros(len(free_pairs))
    starts = [x_init]
    for _ in range(max(1, n_starts - 1)):
        starts.append(rng.uniform(-0.95, 0.95, size=len(free_pairs)))
    for x0 in starts:
        if len(free_pairs) == 0:
            val, C = min_eig_of(np.array([]))
        else:
            res = minimize(obj, x0, method="L-BFGS-B",
                           bounds=[(-0.999999, 0.999999)] * len(free_pairs),
                           options={"maxiter": 500})
            val, C = min_eig_of(res.x)
        if val > best_val:
            best_val, best_C = val, C
    return {
        "k": k,
        "assigned": {f"{i}-{j}": rho for (i, j), rho in assigned.items()},
        "optimal_min_eigenvalue": float(best_val),
        "completable": bool(best_val > 1e-9),
        "certificate_matrix": None if best_C is None else best_C.tolist(),
    }


def canonical_cycle_cases() -> list[dict]:
    """Deterministic WP2.3 witnesses/controls, expectations verified
    numerically (min-eigenvalue optima): constant and sign-alternating 4-cycles
    complete (min-eig = 1 - |c| at the symmetric optimum); obstruction needs
    incompatible path-implied correlations (mixed case), or the Phase-1
    triangle configuration."""
    cases = []

    def add(name, poset_desc, k, assigned, expect):
        res = psd_completion_min_eig(k, assigned, seed=3)
        res.update({"name": name, "poset": poset_desc,
                    "expected": expect,
                    "matches_expectation": (
                        (res["completable"] and expect == "GLUES") or
                        ((not res["completable"]) and expect == "OBSTRUCTED"))})
        cases.append(res)

    add("cycle4_const_0.9", "[[12],[23],[34],[14]]", 4,
        {(0, 1): 0.9, (1, 2): 0.9, (2, 3): 0.9, (0, 3): 0.9}, "GLUES")
    add("cycle4_const_0.5", "[[12],[23],[34],[14]]", 4,
        {(0, 1): 0.5, (1, 2): 0.5, (2, 3): 0.5, (0, 3): 0.5}, "GLUES")
    add("cycle4_alternating_0.9", "[[12],[23],[34],[14]]", 4,
        {(0, 1): 0.9, (1, 2): -0.9, (2, 3): 0.9, (0, 3): -0.9}, "GLUES")
    add("cycle4_mixed_witness", "[[12],[23],[34],[14]]", 4,
        {(0, 1): 0.95, (1, 2): 0.95, (2, 3): -0.95, (0, 3): 0.0}, "OBSTRUCTED")
    add("triangle_control_O2", "[[12],[13],[23]] with CI rho12=0", 3,
        {(0, 1): 0.0, (0, 2): 0.5, (1, 2): 0.5}, "GLUES")
    add("triangle_obstruction_O1", "[[12],[13],[23]] with CI rho12=0", 3,
        {(0, 1): 0.0, (0, 2): 0.9, (1, 2): 0.9}, "OBSTRUCTED")
    return cases
