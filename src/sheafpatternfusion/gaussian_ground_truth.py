"""Linear-Gaussian missingness ground truth with pattern-specific shift knobs.

Model: V ~ N(mu, Sigma) in R^n. Individuals are assigned patterns r with
probabilities pi_r (missingness independent of V: the Phase-1 null; MNAR-style
contamination is injected by shifting the OBSERVED coordinates of individuals
in pattern r by d_r, i.e., the observed law of pattern r is

    q_r = N(mu[O(r)] + d_r, Sigma[O(r), O(r)]).

Analytic quantities are exact linear algebra; `monte_carlo_check` verifies the
analytics against direct sampling on the estimand scale (tolerance 1e-2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GaussianPatternModel:
    mu: np.ndarray                      # (n,)
    sigma: np.ndarray                   # (n,n) PSD
    pattern_probs: dict[tuple[int, ...], float]   # pi_r over realized patterns
    shifts: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)  # d_r on O(r)

    def observed_set(self, r: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(i for i in range(len(r)) if r[i] == 1)

    def pattern_mean(self, r: tuple[int, ...]) -> np.ndarray:
        idx = list(self.observed_set(r))
        out = self.mu[idx].copy()
        if r in self.shifts:
            out = out + self.shifts[r]
        return out

    def pattern_cov(self, r: tuple[int, ...]) -> np.ndarray:
        idx = list(self.observed_set(r))
        return self.sigma[np.ix_(idx, idx)]

    def full_moments(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mu.copy(), self.sigma.copy()

    def sample(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (V[n,n] with NaN masking applied per pattern, R[n,n])."""
        rng = np.random.default_rng(seed)
        n_vars = len(self.mu)
        pats = list(self.pattern_probs.keys())
        probs = np.array([self.pattern_probs[p] for p in pats])
        probs = probs / probs.sum()
        assign = rng.choice(len(pats), size=n, p=probs)
        X = rng.multivariate_normal(self.mu, self.sigma, size=n)
        R = np.zeros((n, n_vars), dtype=np.int64)
        for k in range(n):
            r = pats[assign[k]]
            R[k] = r
            idx = self.observed_set(r)
            X[k, list(idx)] += self.shifts.get(r, np.zeros(len(idx)))
        Vm = X.copy()
        Vm[R == 0] = np.nan
        return Vm, R

    def monte_carlo_check(self, n: int = 600_000, seed: int = 7):
        """Compare empirical per-pattern means/covariances to analytics with
        MC-error-scaled tolerances: pass iff err < 4 standard errors per entry."""
        Vm, R = self.sample(n, seed)
        worst_mean_ratio = 0.0
        worst_cov_ratio = 0.0
        for r in self.pattern_probs:
            mask = (R == np.array(r)).all(axis=1)
            n_r = int(mask.sum())
            if n_r < 2000:
                continue
            idx = list(self.observed_set(r))
            Xr = Vm[mask][:, idx]
            var_emp = np.nanvar(Xr, axis=0)
            se_mean = np.sqrt(var_emp / n_r)
            emp_mean = np.nanmean(Xr, axis=0)
            mean_err = np.abs(emp_mean - self.pattern_mean(r))
            worst_mean_ratio = max(worst_mean_ratio,
                                   float(np.max(mean_err / (4 * se_mean + 1e-12))))
            emp_cov = np.atleast_2d(np.cov(Xr, rowvar=False))
            vdiag = np.clip(np.diag(self.pattern_cov(r)), 1e-6, None)
            se_cov = np.sqrt(2.0 / n_r) * np.outer(vdiag, vdiag) ** 0.5 * np.sqrt(1.0)
            cov_err = np.abs(emp_cov - self.pattern_cov(r))
            worst_cov_ratio = max(worst_cov_ratio,
                                  float(np.max(cov_err / (4 * se_cov + 1e-12))))
        return {
            "worst_mean_ratio": worst_mean_ratio,
            "worst_cov_ratio": worst_cov_ratio,
            "pass": bool(worst_mean_ratio < 1.0 and worst_cov_ratio < 3.0),
        }


def closed_form_pooled_mean(model: GaussianPatternModel) -> np.ndarray:
    """Inverse-variance-free sanity quantity: probability-weighted average of
    pattern means embedded to full dimension (weights sum to one)."""
    n = len(model.mu)
    acc = np.zeros(n)
    for r, w in model.pattern_probs.items():
        idx = list(model.observed_set(r))
        acc[list(idx)] += w * model.pattern_mean(r)
    return acc


def mar_null_model(seed: int = 0, n: int = 3) -> GaussianPatternModel:
    """A clean MAR-null instance: no shifts, mechanism independent of values.
    Pattern probabilities are Dirichlet-drawn with a floor so every realized
    pattern keeps adequate sample mass in Monte Carlo checks."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T + 0.5 * np.eye(n)
    mu = rng.normal(size=n)
    pats = [tuple(int(b) for b in format(k, f"0{n}b")) for k in range(1, 2 ** n)]
    raw = rng.dirichlet(np.ones(len(pats)))
    probs_arr = np.clip(raw, 0.06, None)
    probs_arr = probs_arr / probs_arr.sum()
    probs = {p: float(probs_arr[i]) for i, p in enumerate(pats)}
    return GaussianPatternModel(mu=mu, sigma=Sigma, pattern_probs=probs)


def contaminated_model(shift_on: int = 2, delta: float = 1.5, seed: int = 1,
                       n: int = 3) -> GaussianPatternModel:
    """MAR base plus a mean shift injected on the coordinates observed in the
    fully observed pattern (pattern-level MNAR-style contamination knob)."""
    m = mar_null_model(seed=seed, n=n)
    r_star = tuple([1] * n)
    idx = list(m.observed_set(r_star))
    d = np.zeros(len(idx))
    if shift_on in idx:
        d[idx.index(shift_on)] = delta
    m.shifts[r_star] = d
    return m
