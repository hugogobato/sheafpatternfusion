"""Sheaf Laplacian assembly and spectral quantities (Hansen-Ghrist style),
specialized to the vertex-space Gaussian-mean sheaf."""
from __future__ import annotations

import numpy as np

from .poset import PatternPoset
from .sheaf import GaussianMeanSheaf


def assemble_laplacian(poset: PatternPoset, weights: dict | None = None) -> np.ndarray:
    """L = A^T A where A stacks sqrt(w_e) (R_e on head, -I on tail) per cover
    edge. PSD by construction; ker L = harmonic (consistent) sections when no
    observation terms are present."""
    return GaussianMeanSheaf(poset).laplacian(weights)


def spectrum(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (ascending) and eigenvectors of a symmetric psd matrix."""
    evals, evecs = np.linalg.eigh(0.5 * (L + L.T))
    return evals, evecs


def harmonic_dimension(L: np.ndarray, tol: float = 1e-9) -> int:
    evals, _ = spectrum(L)
    return int(np.sum(evals < tol))


def is_psd(L: np.ndarray, tol: float = 1e-10) -> bool:
    evals, _ = spectrum(L)
    return float(np.min(evals)) > -tol
