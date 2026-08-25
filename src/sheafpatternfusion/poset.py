"""Pattern poset: patterns ordered by inclusion of observed sets."""
from __future__ import annotations

import itertools

import numpy as np


class PatternPoset:
    """Elements are bit tuples r (r_i=1 iff V_i observed); order is inclusion
    of observed sets. Provides covers (Hasse diagram) and comparability."""

    def __init__(self, patterns: list[tuple[int, ...]]):
        self.patterns = sorted(set(tuple(p) for p in patterns))
        if not self.patterns:
            raise ValueError("empty pattern set")
        self.n_vars = len(self.patterns[0])
        assert all(len(p) == self.n_vars for p in self.patterns)
        self._covers = self._build_covers()

    # ---------------- order structure ----------------

    def observed_set(self, r: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(i for i in range(self.n_vars) if r[i] == 1)

    def leq(self, a: tuple[int, ...], b: tuple[int, ...]) -> bool:
        """a <= b iff O(a) subset O(b)."""
        return all(a[i] <= b[i] for i in range(self.n_vars))

    def comparable_pairs(self) -> list[tuple[tuple, tuple]]:
        return [(a, b) for a in self.patterns for b in self.patterns
                if a != b and self.leq(a, b)]

    def _build_covers(self) -> list[tuple[tuple, tuple]]:
        covers = []
        for a, b in self.comparable_pairs():
            between = [c for c in self.patterns
                       if c != a and c != b and self.leq(a, c) and self.leq(c, b)]
            if not between:
                covers.append((a, b))
        return covers

    @property
    def covers(self):
        return list(self._covers)

    def overlap(self, a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(i for i in range(self.n_vars) if a[i] == 1 and b[i] == 1)

    def connected_components(self) -> int:
        """Components of the undirected graph on patterns with edges between
        patterns sharing at least one observed variable."""
        parent = {p: p for p in self.patterns}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, a in enumerate(self.patterns):
            for b in self.patterns[i + 1:]:
                if self.overlap(a, b):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
        return len({find(p) for p in self.patterns})

    def downset(self, r) -> list[tuple]:
        return [a for a in self.patterns if self.leq(a, r)]


def restriction_matrix(big: tuple[int, ...], small: tuple[int, ...]) -> np.ndarray:
    """Marginalization as a selection matrix R (d_small x d_big): the mean
    vector on O(small) equals R @ mean vector on O(big)."""
    idx_big = [i for i in range(len(big)) if big[i] == 1]
    idx_small = [i for i in range(len(small)) if small[i] == 1]
    R = np.zeros((len(idx_small), len(idx_big)))
    for a, i in enumerate(idx_small):
        R[a, idx_big.index(i)] = 1.0
    return R


def brute_force_transitive_reduction(order_pairs: set[tuple]) -> set[tuple]:
    """Reference implementation used to validate Hasse construction."""
    elems = {x for p in order_pairs for x in p}
    covers = set()
    for a, b in order_pairs:
        between = [c for c in elems
                   if c not in (a, b) and (a, c) in order_pairs and (c, b) in order_pairs]
        if not between:
            covers.add((a, b))
    return covers


def random_poset(n_vars: int, n_patterns: int, seed: int) -> PatternPoset:
    rng = np.random.default_rng(seed)
    pool = list(itertools.product((0, 1), repeat=n_vars))
    chosen = set()
    while len(chosen) < min(n_patterns, len(pool)):
        chosen.add(pool[int(rng.integers(0, len(pool)))])
    return PatternPoset(sorted(chosen))
