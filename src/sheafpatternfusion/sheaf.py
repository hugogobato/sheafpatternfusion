"""Sheaf structures: constrained stalks, marginalization restrictions,
section checks (discrete and Gaussian-mean instantiations)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .poset import PatternPoset, restriction_matrix


# --------------------------------------------------------------------------
# discrete stalks with conditional-independence constraint lists
# --------------------------------------------------------------------------

@dataclass
class CIConstraint:
    x: tuple[int, ...]
    y: tuple[int, ...]
    z: tuple[int, ...]  # disjoint index subsets within a pattern's observed set

    def holds(self, table: dict[tuple, float], observed: tuple[int, ...]) -> bool:
        def pos(i):
            return sum(1 for j in range(len(observed)) if observed[j] == 1 and j < i)

        px, py, pz = [pos(i) for i in self.x], [pos(i) for i in self.y], [pos(i) for i in self.z]
        joint: dict[tuple, float] = {}
        for o, c in table.items():
            key = (tuple(o[i] for i in px), tuple(o[i] for i in py), tuple(o[i] for i in pz))
            joint[key] = joint.get(key, 0.0) + c
        pz_marg: dict[tuple, float] = {}
        for (kx, ky, kz), c in joint.items():
            pz_marg[kz] = pz_marg.get(kz, 0.0) + c
        for kz, cz in pz_marg.items():
            if cz <= 0:
                continue
            mx: dict[tuple, float] = {}
            my: dict[tuple, float] = {}
            mxy: dict[tuple, float] = {}
            for (kx2, ky2, kz2), c2 in joint.items():
                if kz2 != kz:
                    continue
                mx[kx2] = mx.get(kx2, 0.0) + c2
                my[ky2] = my.get(ky2, 0.0) + c2
                mxy[(kx2, ky2)] = c2
            for (kx2, ky2), cxy in mxy.items():
                if abs(cxy - mx[kx2] * my[ky2] / cz) > 1e-8 * max(1e-8, cz):
                    return False
        return True


@dataclass
class DiscreteSheaf:
    poset: PatternPoset
    constraints: dict[tuple[int, ...], list[CIConstraint]] = field(default_factory=dict)

    def stalk_constraints(self, r):
        return self.constraints.get(r, [])

    def in_stalk(self, r, table) -> bool:
        """Membership on the MASS-CARRYING table W_r: constraints are checked
        on the normalized conditional (W_r / mass); zero-mass stalks allowed."""
        s = sum(table.values())
        if s <= 0:
            return all(v >= -1e-12 for v in table.values())
        if any(v < -1e-12 for v in table.values()):
            return False
        norm = {k: v / s for k, v in table.items()}
        return all(c.holds(norm, r) for c in self.stalk_constraints(r))

    def restrict(self, big, small, table) -> dict[tuple, float]:
        assert self.poset.leq(small, big)
        idx_big = [i for i in range(len(big)) if big[i] == 1]
        idx_small = [i for i in range(len(small)) if small[i] == 1]
        out: dict[tuple, float] = {}
        for o, c in table.items():
            key = tuple(o[idx_big.index(i)] for i in idx_small)
            out[key] = out.get(key, 0.0) + c
        return out

    def is_section(self, family: dict[tuple, dict[tuple, float]], tol=1e-9) -> bool:
        """A family of tables is a section iff every table is in its stalk and
        restrictions agree on overlaps."""
        for r, tab in family.items():
            if not self.in_stalk(r, tab):
                return False
        for small, big in self.poset.covers:
            if big not in family or small not in family:
                continue
            pushed = self.restrict(big, small, family[big])
            for o, c in pushed.items():
                if abs(c - family[small].get(o, 0.0)) > tol:
                    return False
        return True


def random_discrete_table(n_obs: int, rng: np.random.Generator) -> dict[tuple, float]:
    raw = rng.random((2,) * n_obs).ravel()
    raw = raw / raw.sum()
    keys = list(itertools.product((0, 1), repeat=n_obs))
    return {k: float(v) for k, v in zip(keys, raw)}


def collate_unconstrained(poset: PatternPoset, family: dict[tuple, dict[tuple, float]]) -> dict[tuple, float]:
    """Glue an unconstrained consistent family onto the maximal pattern: the
    collated table IS the top stalk element (Hazard-A check helper)."""
    top = max(poset.patterns, key=lambda p: sum(p))
    if top not in family:
        raise ValueError("family lacks the maximal pattern")
    return dict(family[top])


# --------------------------------------------------------------------------
# Gaussian mean-coordinate sheaf (linear-algebra instantiation)
# --------------------------------------------------------------------------

class GaussianMeanSheaf:
    """Vertex-space sheaf of mean vectors with selection restrictions.

    Vertex block for pattern r: R^{d_r}, d_r = |O(r)|. Restriction along a
    cover edge (small <= big): selection matrix. Assignments are stacked into
    one vector in the order of `self.poset.patterns`.
    """

    def __init__(self, poset: PatternPoset):
        self.poset = poset
        self.dims = [sum(p) for p in poset.patterns]
        self.offsets = np.concatenate([[0], np.cumsum(self.dims)])
        self.D = int(self.offsets[-1])

    def block(self, g: np.ndarray, r: tuple[int, ...]) -> np.ndarray:
        k = self.poset.patterns.index(r)
        return g[self.offsets[k]:self.offsets[k + 1]]

    def restriction(self, big, small) -> np.ndarray:
        return restriction_matrix(big, small)

    def edge_operator_blocks(self, weights: dict | None = None):
        """Return A (edge x vertex matrix) such that J(g)=||A g||^2 encodes
        sum_e w_e ||R_e g_head - g_tail||^2 over covers."""
        rows = []
        for e, (small, big) in enumerate(self.poset.covers):
            w = 1.0 if weights is None else float(weights.get((small, big), 1.0))
            k_s = self.poset.patterns.index(small)
            k_b = self.poset.patterns.index(big)
            d_s, d_b = self.dims[k_s], self.dims[k_b]
            R = self.restriction(big, small)
            row = np.zeros((d_s, self.D))
            row[:, self.offsets[k_s]:self.offsets[k_s + 1]] -= np.sqrt(w) * np.eye(d_s)
            row[:, self.offsets[k_b]:self.offsets[k_b + 1]] += np.sqrt(w) * R
            rows.append(row)
        if rows:
            return np.vstack(rows)
        return np.zeros((0, self.D))

    def laplacian(self, weights: dict | None = None) -> np.ndarray:
        A = self.edge_operator_blocks(weights)
        return A.T @ A

    def project(self, b: dict[tuple[int, ...], np.ndarray],
                weights: dict | None = None) -> tuple[np.ndarray, float]:
        """Weighted least-squares projection of per-pattern observations b_r
        onto global sections. Returns (stacked section, residual norm)."""
        rows, rhs, wlist = [], [], []
        for k, r in enumerate(self.poset.patterns):
            if r not in b:
                continue
            w = 1.0 if weights is None else float(weights.get(("obs", r), 1.0))
            eye = np.sqrt(w) * np.eye(self.dims[k])
            row = np.zeros((self.dims[k], self.D))
            row[:, self.offsets[k]:self.offsets[k + 1]] = eye
            rows.append(row)
            rhs.append(np.sqrt(w) * b[r])
            wlist.append(w)
        E = np.vstack(rows) if rows else np.zeros((0, self.D))
        e = np.concatenate(rhs) if rhs else np.zeros(0)
        A = self.edge_operator_blocks(weights)
        M = np.vstack([E, A])
        y = np.concatenate([e, np.zeros(A.shape[0])])
        g, *_ = np.linalg.lstsq(M, y, rcond=None)
        resid_sq = float(np.sum((M @ g - y) ** 2))
        return g, resid_sq
