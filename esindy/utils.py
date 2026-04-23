"""
Utility functions for the E-SINDy active learning pipeline.

Covers:
- Derivative estimation from sparse/noisy data
- Data tensor management (revealed mask operations)
- Noise injection helpers
"""

import numpy as np
from typing import Optional


# Derivative estimation

def estimate_derivatives(
    X: np.ndarray,
    t: np.ndarray,
    method: str = "finite_difference",
) -> np.ndarray:
    """Estimate dX/dt from state data and timepoints.

    Parameters
    ----------
    X : array (n_timepoints, n_species)
        State matrix (may contain NaN for unrevealed entries).
    t : array (n_timepoints,)
        Timepoints corresponding to rows of X.
    method : str
        'finite_difference' — 2nd order central differences (default).
        'savgol' — Savitzky-Golay smoothed derivative.

    Returns
    -------
    X_dot : array (n_timepoints, n_species)
        Estimated derivatives. Boundary points use forward/backward diff.
        NaN entries in X propagate to NaN in X_dot.
    """
    if method == "finite_difference":
        return _fd_derivative(X, t)
    elif method == "savgol":
        return _savgol_derivative(X, t)
    else:
        raise ValueError(f"Unknown derivative method: {method}")


def _fd_derivative(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central finite differences with forward/backward at boundaries."""
    m, n = X.shape
    X_dot = np.full_like(X, np.nan)
    dt = np.diff(t)

    # Forward difference at t=0
    X_dot[0] = (X[1] - X[0]) / dt[0]

    # Central differences for interior points
    for i in range(1, m - 1):
        X_dot[i] = (X[i + 1] - X[i - 1]) / (t[i + 1] - t[i - 1])

    # Backward difference at t=end
    X_dot[-1] = (X[-1] - X[-2]) / dt[-1]

    return X_dot


def _savgol_derivative(
    X: np.ndarray, t: np.ndarray, window: int = 7, polyorder: int = 3
) -> np.ndarray:
    """Savitzky-Golay smoothed derivative estimation."""
    from scipy.signal import savgol_filter

    dt = np.mean(np.diff(t))  # assumes roughly uniform spacing
    m, n = X.shape
    X_dot = np.full_like(X, np.nan)

    for j in range(n):
        col = X[:, j]
        valid = ~np.isnan(col)
        if valid.sum() >= window:
            # Apply to valid contiguous blocks
            X_dot[valid, j] = savgol_filter(
                col[valid], window_length=min(window, valid.sum()),
                polyorder=polyorder, deriv=1, delta=float(dt)
            )
    return X_dot


# Data tensor operations

def get_revealed_data(pool: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract revealed (non-masked) data from the pool as SINDy-ready matrices.

    For each experiment, extracts contiguous revealed timepoint windows
    where ALL species in that window are revealed, constructs X and X_dot,
    and concatenates across experiments.

    Parameters
    ----------
    pool : dict from TargetSystem.generate_pool()

    Returns
    -------
    X : array (total_revealed_rows, n_species)
        Concatenated state data from all usable revealed windows.
    X_dot : array (total_revealed_rows, n_species)
        Corresponding derivative estimates.
    t_all : array (total_revealed_rows,)
        Corresponding timepoints (for reference).
    """
    X_full = pool["X_full"]        # (n_exp, n_t, n_species)
    revealed = pool["revealed"]     # (n_exp, n_t, n_species)
    t_eval = pool["t_eval"]

    n_exp, n_t, n_species = X_full.shape
    X_blocks = []
    Xdot_blocks = []
    t_blocks = []

    for i in range(n_exp):
        # A timepoint is "usable" if ALL species are revealed there
        usable = revealed[i].all(axis=1)  # (n_t,)

        # Find contiguous runs of usable timepoints
        runs = _find_contiguous_runs(usable)

        for start, end in runs:
            if end - start < 3:
                # Need at least 3 points for central finite diff
                continue

            X_window = X_full[i, start:end, :]
            t_window = t_eval[start:end]
            Xdot_window = estimate_derivatives(X_window, t_window)

            X_blocks.append(X_window)
            Xdot_blocks.append(Xdot_window)
            t_blocks.append(t_window)

    if len(X_blocks) == 0:
        return (
            np.empty((0, n_species)),
            np.empty((0, n_species)),
            np.empty((0,)),
        )

    return (
        np.vstack(X_blocks),
        np.vstack(Xdot_blocks),
        np.concatenate(t_blocks),
    )


def _find_contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True runs in a 1D boolean array.

    Returns list of (start, end) slices (end is exclusive).
    """
    runs = []
    in_run = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))
    return runs


def reveal_full_trajectories(
    pool: dict,
    experiment_indices: list[int],
) -> None:
    """Reveal all species at all timepoints for given experiments."""
    for idx in experiment_indices:
        pool["revealed"][idx, :, :] = True
