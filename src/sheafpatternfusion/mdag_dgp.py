"""m-graph data-generating process for discrete (binary) missing-data models.

Semantics follow Mohan-Pearl-Tian (2013) / Mohan-Pearl (2021): each binary
variable V_i has a structural mechanism P(v_i | parents); each missingness
indicator R_i has mechanism P(r_i | pa_G(R_i)) where pa_G(R_i) is a set of
VARIABLE indices and may include i itself (self-censoring MNAR edge).

Convention: r_i = 1 means V_i observed.

This module is ground-truth-side code only: it never uses sheaf machinery.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MDAG:
    n_vars: int
    var_parents: dict[int, tuple[int, ...]]
    r_parents: dict[int, tuple[int, ...]]  # may contain i itself (MNAR self-edge)
    var_cpt: dict[int, dict[tuple, float]] = field(default_factory=dict)
    r_cpt: dict[int, dict[tuple, float]] = field(default_factory=dict)

    def __post_init__(self):
        for i in range(self.n_vars):
            self.var_cpt.setdefault(i, {})
            self.r_cpt.setdefault(i, {})

    # ---------------- validation ----------------

    def validate_topological(self):
        """Variable mechanisms must depend on lower-indexed variables only;
        indicator mechanisms may depend on any variables (evaluation is
        conditional on the full v vector, so no ordering constraint applies)."""
        for i in range(self.n_vars):
            assert all(p < i for p in self.var_parents[i]), f"var parent order at {i}"

    def random_fill(self, rng: np.random.Generator):
        """Fill CPTs with uniform-random probabilities in [0.15, 0.85]."""
        for i in range(self.n_vars):
            pa = self.var_parents[i]
            keys = list(itertools.product(*[(0, 1)] * len(pa))) if pa else [()]
            self.var_cpt[i] = {k: rng.uniform(0.15, 0.85) for k in keys}
        for i in range(self.n_vars):
            pa = self.r_parents[i]
            keys = list(itertools.product(*[(0, 1)] * len(pa))) if pa else [()]
            self.r_cpt[i] = {k: rng.uniform(0.25, 0.75) for k in keys}

    # ---------------- exact laws ----------------

    def p_var(self, v: tuple[int, ...]) -> float:
        out = 1.0
        for i in range(self.n_vars):
            pa = tuple(v[p] for p in self.var_parents[i])
            p1 = self.var_cpt[i][pa]
            out *= p1 if v[i] == 1 else 1.0 - p1
        return out

    def p_r_given_v(self, r: tuple[int, ...], v: tuple[int, ...]) -> float:
        out = 1.0
        for i in range(self.n_vars):
            pa = tuple(v[p] for p in self.r_parents[i])  # includes i if self-edge
            q = self.r_cpt[i][pa]
            out *= q if r[i] == 1 else 1.0 - q
        return out

    def joint_table(self) -> dict[tuple[tuple, tuple], float]:
        """P(v, r) over all cells; entries may be zero via mechanisms."""
        out = {}
        for v in itertools.product((0, 1), repeat=self.n_vars):
            pv = self.p_var(v)
            if pv == 0.0:
                continue
            for r in itertools.product((0, 1), repeat=self.n_vars):
                out[(v, r)] = pv * self.p_r_given_v(r, v)
        return out

    def observed_laws(self, jt: dict | None = None) -> dict[tuple, dict[tuple, float]]:
        """q_r(o) = P(V_O=o | R=r) for every realized pattern r."""
        jt = self.joint_table() if jt is None else jt
        num: dict[tuple, dict[tuple, float]] = {}
        den: dict[tuple, float] = {}
        for (v, r), p in jt.items():
            o = tuple(v[i] for i in range(self.n_vars) if r[i] == 1)
            num.setdefault(r, {}).setdefault(o, 0.0)
            num[r][o] += p
            den[r] = den.get(r, 0.0) + p
        return {
            r: {o: c / den[r] for o, c in cells.items()}
            for r, cells in num.items()
            if den.get(r, 0.0) > 0.0
        }

    def realized_patterns(self, tol: float = 0.0, jt: dict | None = None) -> list[tuple]:
        jt = self.joint_table() if jt is None else jt
        den: dict[tuple, float] = {}
        for (v, r), p in jt.items():
            den[r] = den.get(r, 0.0) + p
        return sorted([r for r, p in den.items() if p > tol])

    # ---------------- sampling ----------------

    def sample(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw n individuals; returns (V[n,n], R[n,n]) with full truth."""
        rng = np.random.default_rng(seed)
        V = np.zeros((n, self.n_vars), dtype=np.int64)
        for row in range(n):
            v = []
            for i in range(self.n_vars):
                pa = tuple(v[p] for p in self.var_parents[i])
                v.append(int(rng.random() < self.var_cpt[i][pa]))
            V[row] = v
        R = np.zeros((n, self.n_vars), dtype=np.int64)
        for row in range(n):
            v = list(V[row])
            r = []
            for i in range(self.n_vars):
                pa = tuple(v[p] for p in self.r_parents[i])
                r.append(int(rng.random() < self.r_cpt[i][pa]))
            R[row] = r
        return V, R

    @staticmethod
    def empirical_observed_laws(V: np.ndarray, R: np.ndarray) -> dict[tuple, dict[tuple, float]]:
        """Empirical pattern-conditional laws from a masked dataset."""
        n, d = V.shape
        out: dict[tuple, dict[tuple, float]] = {}
        cnt: dict[tuple, int] = {}
        for k in range(n):
            r = tuple(int(x) for x in R[k])
            o = tuple(int(x) for x in V[k][np.array(r, dtype=bool)])
            out.setdefault(r, {}).setdefault(o, 0.0)
            out[r][o] += 1.0
            cnt[r] = cnt.get(r, 0) + 1
        return {r: {o: c / cnt[r] for o, c in cells.items()} for r, cells in out.items()}
