"""
Abstract base class for target dynamical systems.

Each target system acts as an oracle: given initial conditions and time points,
it returns (optionally noisy) trajectories by integrating the true ODE system.
It also exposes ground-truth metadata for evaluation (species names, true
coefficients, and the candidate function library expected by SINDy).
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp


class TargetSystem(ABC):
    """Oracle interface for a ground-truth ODE system."""

    # ------------------------------------------------------------------
    # Subclasses must implement these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def n_species(self) -> int:
        """Number of state variables (species)."""

    @property
    @abstractmethod
    def species_names(self) -> list[str]:
        """Human-readable names for each species."""

    @abstractmethod
    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        """Right-hand side of dx/dt = f(x). Used by solve_ivp."""

    @abstractmethod
    def get_true_coefficients(self) -> dict:
        """Return ground-truth info for evaluation.

        Returns a dict with at minimum:
            - 'active_terms': dict mapping species name -> list of
              (library_term_str, coefficient_value) for nonzero terms
            - 'params': dict of named parameter values
        """

    # ------------------------------------------------------------------
    # Shared simulation logic
    # ------------------------------------------------------------------

    def simulate(
        self,
        x0: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray,
        noise_level: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        **ivp_kwargs,
    ) -> np.ndarray:
        """Simulate a trajectory from initial condition x0.

        Parameters
        ----------
        x0 : array of shape (n_species,)
            Initial concentrations / state values.
        t_span : (t_start, t_end)
        t_eval : array of timepoints at which to return the solution.
        noise_level : float
            Standard deviation of additive Gaussian noise, scaled by the
            RMS of each species trajectory (0 = clean data).
        rng : numpy Generator, optional
            For reproducible noise.
        **ivp_kwargs : passed to scipy.integrate.solve_ivp

        Returns
        -------
        X : array of shape (len(t_eval), n_species)
            State matrix (rows = timepoints, columns = species).
        """
        defaults = dict(method="RK45", rtol=1e-10, atol=1e-12, dense_output=False)
        defaults.update(ivp_kwargs)

        sol = solve_ivp(self.rhs, 
                        t_span, x0, 
                        t_eval=t_eval,
                        method=ivp_kwargs.get("method", "RK45"),
                        rtol=ivp_kwargs.get("rtol", 1e-10),
                        atol=ivp_kwargs.get("atol", 1e-12),
                        dense_output=ivp_kwargs.get("dense_output", False), 
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        X = sol.y.T  # shape (m, n)

        if noise_level > 0:
            if rng is None:
                rng = np.random.default_rng()
            rms = np.sqrt(np.mean(X**2, axis=0))  # per-species RMS
            X = X + noise_level * rms * rng.standard_normal(X.shape)

        return X

    def generate_pool(
        self,
        ic_ranges: list[tuple[float, float]],
        n_experiments: int,
        t_eval: np.ndarray,
        noise_level: float = 0.0,
        seed: int = 42,
    ) -> dict:
        """Pre-generate a pool of trajectories for active learning.

        Parameters
        ----------
        ic_ranges : list of (low, high) per species for sampling ICs.
        n_experiments : number of experiments (distinct ICs).
        t_eval : shared timepoint array.
        noise_level : noise for all trajectories.
        seed : random seed.

        Returns
        -------
        pool : dict with keys:
            - 'X_full': array (n_experiments, n_timepoints, n_species)
            - 'ICs': array (n_experiments, n_species)
            - 't_eval': the timepoint array
            - 'revealed': bool array (n_experiments, n_timepoints, n_species)
              initialized to all False
        """
        rng = np.random.default_rng(seed)
        ic_array = np.asarray(ic_ranges)  # (n_species, 2)

        ICs = np.column_stack([
            rng.uniform(low, high, size=n_experiments)
            for low, high in ic_array
        ])

        n_t = len(t_eval)
        X_full = np.zeros((n_experiments, n_t, self.n_species))
        t_span = (t_eval[0], t_eval[-1])

        for i in range(n_experiments):
            X_full[i] = self.simulate(
                ICs[i], t_span, t_eval, noise_level=noise_level, rng=rng
            )

        return {
            "X_full": X_full,
            "ICs": ICs,
            "t_eval": t_eval,
            "revealed": np.zeros((n_experiments, n_t, self.n_species), dtype=bool),
        }
