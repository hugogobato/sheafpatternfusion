"""Phase 2 enumeration infrastructure (WP2.1).

Structure space: m-graphs on binary variables with topologically ordered
variable mechanisms (V_i may only depend on lower-indexed variables) and
missingness mechanisms P(R_i | pa) where pa may include any variable plus
R_i's own variable (self-censoring MNAR edge). A STRUCTURE is
(var_parents, r_parents); an INSTANCE adds seeded CPT parameters and targets.

Everything here is ground-truth-side or structure-side bookkeeping; no sheaf
machinery is used for verdicts (engine side stays assumption-clean).
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

import numpy as np

from .mdag_dgp import MDAG


# --------------------------------------------------------------------------
# structure generation
# --------------------------------------------------------------------------

def _subsets(pool: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [tuple(x for x in pool if (m >> pool.index(x)) & 1)
            for m in range(2 ** len(pool))]


def var_dags(n: int) -> list[dict[int, tuple[int, ...]]]:
    """All parent assignments respecting V_i depends on lower indices only."""
    blocks = [_subsets(tuple(range(i))) for i in range(n)]
    return [{i: c for i, c in enumerate(choice)}
            for choice in itertools.product(*blocks)]


def r_mechanisms(n: int) -> list[tuple[tuple[int, ...], ...]]:
    """All parent tuples (pa(R_0), ..., pa(R_{n-1})); pa(R_i) is a subset of
    the variable set, and containing i itself is the self-censoring edge."""
    block = _subsets(tuple(range(n)))
    return [tuple(choice) for choice in itertools.product(block, repeat=n)]


def all_structures(n: int) -> list[tuple[dict[int, tuple[int, ...]], tuple]]:
    vds = var_dags(n)
    rms = r_mechanisms(n)
    return [(vd, rm) for vd in vds for rm in rms]


def instantiate(structure, seed: int, fixed_cpt: list[dict] | None = None) -> MDAG:
    vp, rp = structure
    inst = MDAG(n_vars=len(vp), var_parents=dict(vp),
                r_parents={i: tuple(p) for i, p in enumerate(rp)})
    inst.validate_topological()
    rng = np.random.default_rng(seed)
    inst.random_fill(rng)
    for fx in fixed_cpt or []:
        table = inst.r_cpt if fx["kind"] == "r" else inst.var_cpt
        table[fx["node"]][tuple(fx["parents"])] = float(fx["p"])
    return inst


# --------------------------------------------------------------------------
# mechanism classification (per drawn instance; depends on realized patterns)
# --------------------------------------------------------------------------

def classify(inst: MDAG) -> dict:
    jt = inst.joint_table()
    patterns = inst.realized_patterns(jt=jt)
    always = [i for i in range(inst.n_vars) if all(r[i] == 1 for r in patterns)]
    never = [i for i in range(inst.n_vars) if all(r[i] == 0 for r in patterns)]
    has_self = any(i in inst.r_parents[i] for i in range(inst.n_vars))
    nonempty_pa = any(len(inst.r_parents[i]) > 0 for i in range(inst.n_vars))
    is_mcar = not nonempty_pa
    pa_all_observed = all(all(p in always for p in inst.r_parents[i])
                          for i in range(inst.n_vars))
    is_mar = (not is_mcar) and pa_all_observed
    if is_mcar:
        cls = "MCAR"
    elif is_mar:
        cls = "MAR"
    elif has_self:
        cls = "MNAR_self"
    else:
        cls = "MNAR_other"
    return {
        "mechanism_class": cls,
        "is_mcar": is_mcar,
        "is_mar": is_mar,
        "has_self_edge": has_self,
        "always_observed": tuple(always),
        "never_observed": tuple(never),
        "n_realized_patterns": len(patterns),
    }


def poset_shape(patterns: list[tuple[int, ...]]) -> str:
    """Coarse shape label keyed on the overlap hypergraph (the structure that
    governs gluing): chain / acyclic (Berge) / cyclic."""
    ps = sorted(patterns)
    if all(all(a[i] <= b[i] for i in range(len(a))) for a, b in zip(ps, ps[1:])):
        return "chain"
    sets = [frozenset(i for i in range(len(p)) if p[i] == 1) for p in ps]
    return "acyclic" if graham_acyclic(sets) else "cyclic"


def graham_acyclic(observed_sets: list[frozenset[int]]) -> bool:
    """Graham's algorithm: Berge-acyclicity of the overlap hypergraph. True
    posets (running-intersection property) are where pairwise gluing suffices
    classically; this is the WP2.3 readout key."""
    edges = [set(s) for s in observed_sets if s]
    while len(edges) > 1:
        # find an edge whose intersection with the union of others equals its
        # intersection with ONE other edge (a "leaf")
        found = False
        for k, e in enumerate(edges):
            others = [o for j, o in enumerate(edges) if j != k]
            rest = set().union(*others)
            shared_others = [(frozenset(e & o), o) for o in others if e & o]
            if not shared_others:
                edges.pop(k)
                found = True
                break
            # removable if all its elements shared with others live in one other edge
            touching = frozenset(e & rest)
            if any(touching <= o for _, o in shared_others):
                edges.pop(k)
                found = True
                break
            if not (e & rest):
                edges.pop(k)
                found = True
                break
        if not found:
            return False
    return True


# --------------------------------------------------------------------------
# implied slice CIs (structural, discovered across random parameter draws)
# --------------------------------------------------------------------------

def _ci_candidates(observed: tuple[int, ...]) -> list[tuple[tuple, tuple, tuple]]:
    idx = [i for i in range(len(observed)) if observed[i] == 1]
    out = []
    k = len(idx)
    for mask in range(3 ** k):
        assign = []
        m = mask
        for _ in range(k):
            assign.append(m % 3)
            m //= 3
        x = tuple(idx[j] for j in range(k) if assign[j] == 0)
        y = tuple(idx[j] for j in range(k) if assign[j] == 1)
        z = tuple(idx[j] for j in range(k) if assign[j] == 2)
        if x and y:
            out.append((x, y, z))
    return out


def discover_slice_cis(inst: MDAG, n_draws: int = 24, tol: float = 1e-7,
                       seed: int = 12345) -> dict[tuple, list[tuple]]:
    """CIs (X,Y,Z) that hold on pattern r's normalized slice for EVERY random
    parameter draw: structural consequences of the m-graph, mechanically
    verified. Returns {pattern: [(X,Y,Z), ...]}."""
    def holds(table_norm: dict[tuple, float], observed, x, y, z) -> bool:
        def pos(i):
            return sum(1 for j in range(len(observed)) if observed[j] == 1 and j < i)

        px, py, pz = [pos(i) for i in x], [pos(i) for i in y], [pos(i) for i in z]
        joint: dict = {}
        for o, c in table_norm.items():
            key = (tuple(o[i] for i in px), tuple(o[i] for i in py),
                   tuple(o[i] for i in pz))
            joint[key] = joint.get(key, 0.0) + c
        pzm = {}
        for (kx, ky, kz), c in joint.items():
            pzm[kz] = pzm.get(kz, 0.0) + c
        for kz, cz in pzm.items():
            mx, my, mxy = {}, {}, {}
            for (kx2, ky2, kz2), c2 in joint.items():
                if kz2 != kz:
                    continue
                mx[kx2] = mx.get(kx2, 0.0) + c2
                my[ky2] = my.get(ky2, 0.0) + c2
                mxy[(kx2, ky2)] = c2
            for (kx2, ky2), cxy in mxy.items():
                if abs(cxy - mx[kx2] * my[ky2] / cz) > tol * max(1e-8, cz):
                    return False
        return True

    patterns = inst.realized_patterns(jt=inst.joint_table())
    cands = {r: _ci_candidates(r) for r in patterns}
    alive = {r: list(cands[r]) for r in patterns}
    rng = np.random.default_rng(seed)
    for _ in range(n_draws):
        m = copy.deepcopy(inst)
        for i in range(m.n_vars):
            pa = m.var_parents[i]
            keys = list(itertools.product(*[(0, 1)] * len(pa))) if pa else [()]
            m.var_cpt[i] = {k: float(rng.uniform(0.15, 0.85)) for k in keys}
        for i in range(m.n_vars):
            pa = m.r_parents[i]
            keys = list(itertools.product(*[(0, 1)] * len(pa))) if pa else [()]
            m.r_cpt[i] = {k: float(rng.uniform(0.25, 0.75)) for k in keys}
        q = m.observed_laws()
        for r in patterns:
            if r not in q:
                alive[r] = []
                continue
            tab = q[r]
            alive[r] = [c for c in alive[r] if holds(tab, r, *c)]
        if all(not v for v in alive.values()):
            break
    return {r: alive[r] for r in patterns}


# --------------------------------------------------------------------------
# targets and conflict flags
# --------------------------------------------------------------------------

def pick_targets(inst: MDAG, max_targets: int = 2) -> list[tuple]:
    """Means of partially observed variables (deterministic order), falling
    back to variable 0 when everything is always/never observed."""
    jt = inst.joint_table()
    patterns = inst.realized_patterns(jt=jt)
    partial = [i for i in range(inst.n_vars)
               if any(r[i] == 0 for r in patterns) and any(r[i] == 1 for r in patterns)]
    chosen = partial[:max_targets]
    if not chosen:
        chosen = [0]
    return [("mean", j) for j in chosen]


def conflict_flags(inst: MDAG) -> dict:
    """Population-level pattern-conflict diagnostics.

    mcar_section_violation: two realized patterns observing the same variable
    disagree on the population-marginal law of that variable (the Phase-1
    marginal-sheaf section failure; MCAR-type ignorability characterization).
    """
    jt = inst.joint_table()
    q = inst.observed_laws(jt)
    violation = False
    max_gap = 0.0
    for i in range(inst.n_vars):
        dists = []
        for r, cells in q.items():
            if r[i] != 1:
                continue
            posn = sum(1 for j in range(inst.n_vars) if r[j] == 1 and j < i)
            mg: dict[tuple, float] = {}
            for o, c in cells.items():
                mg[o[posn]] = mg.get(o[posn], 0.0) + c
            dists.append(mg)
        for d in dists[1:]:
            gap = max(abs(d.get(k, 0.0) - dists[0].get(k, 0.0))
                      for k in set(d) | set(dists[0]))
            max_gap = max(max_gap, gap)
            if gap > 1e-9:
                violation = True
    return {
        "conflict_mcar_style": bool(violation),
        "max_cross_pattern_marginal_gap": float(max_gap),
    }


# --------------------------------------------------------------------------
# mandated named classes (gate-memo carry-forward obligation)
# --------------------------------------------------------------------------

def named_structures() -> dict[str, tuple[dict[int, tuple[int, ...]], tuple]]:
    """Mechanism families the Phase-1 memo requires in the enumeration:
    mutual selection, double self-censoring, mediated MNAR, plain self
    censoring, MAR anchor, MCAR reference."""
    two_var = {0: (), 1: (0,)}
    indep2 = {0: (), 1: ()}
    three_chain = {0: (), 1: (0,), 2: (1,)}
    out = {
        "mutual_selection": (indep2, ((), (1,), (0,))),
        "double_self_censor": (two_var, ((0,), (1,))),
        "self_censor_v1": (indep2, ((0,), ())),
        "self_censor_v2_chain": (two_var, ((), (1,))),
        "mediated_mnar": (two_var, ((1,), (0, 1))),
        "mnar_on_partial_cause": (two_var, ((), (0,))),
        "mar_textbook": (two_var, ((), (0,))),
        "mcar_reference": (two_var, ((), ())),
        "three_var_mixed": (three_chain, ((), (0,), (2,))),
        "three_var_double_self": (three_chain, ((0,), (1,), (2,))),
        "collider_selection": (indep2, ((), (0, 1))),
    }
    return out
