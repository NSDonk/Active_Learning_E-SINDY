"""
E-SINDy: Ensemble Sparse Identification of Nonlinear Dynamics.

Wraps PySINDy to provide:
- Single SINDy model fitting (baseline)
- Ensemble (bagging/bragging) fitting with inclusion probabilities
- Ensemble forecasting with variance computation
"""

from typing import Optional, Literal
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class SINDyConfig:
    """Configuration for PySINDy model construction."""

    # Library settings
    poly_degree: int = 2
    include_interaction: bool = True
    include_bias: bool = True  # constant term in library

    # Optimizer settings
    optimizer: str = "STLSQ"  # or "SR3", "SSR"
    threshold: float = 0.1
    alpha: float = 0.05       # ridge regularization (for STRidge-like behavior)
    max_iter: int = 20

    # Differentiation
    diff_method: str = "finite_difference"  # or "smoothed_finite_difference"

    # Feature names (set by caller based on target system)
    feature_names: Optional[list[str]] = None

    def build_library(self) -> ps.PolynomialLibrary:
        return ps.PolynomialLibrary(
            degree=self.poly_degree,
            include_interaction=self.include_interaction,
            include_bias=self.include_bias,
        )

    def build_optimizer(self) -> ps.STLSQ:
        if self.optimizer == "STLSQ":
            return ps.STLSQ(
                threshold=self.threshold,
                alpha=self.alpha,
                max_iter=self.max_iter,
            )
        else:
            raise ValueError(f"Optimizer '{self.optimizer}' not yet supported")

    def build_differentiator(self):
        if self.diff_method == "finite_difference":
            return ps.FiniteDifference()
        elif self.diff_method == "smoothed_finite_difference":
            return ps.SmoothedFiniteDifference()
        else:
            raise ValueError(f"Unknown diff method: {self.diff_method}")


# ======================================================================
# Result containers
# ======================================================================

@dataclass
class SINDyResult:
    """Result from a single SINDy fit."""
    model: ps.SINDy
    coefficients: np.ndarray   # (n_features_in_library, n_species)
    feature_names: list[str]

    def print_model(self):
        self.model.print()


@dataclass
class ESINDyResult:
    """Result from ensemble SINDy fitting."""

    # Aggregated model
    coefficients: np.ndarray        # (n_library_terms, n_species) — aggregated
    feature_names: list[str]

    # Ensemble statistics
    all_coefficients: np.ndarray    # (n_bootstraps, n_library_terms, n_species)
    inclusion_probabilities: np.ndarray  # (n_library_terms, n_species)
    coefficient_std: np.ndarray     # (n_library_terms, n_species)

    # Config used
    config: SINDyConfig = field(default_factory=SINDyConfig)
    aggregation: str = "mean"

    def get_active_terms(self, threshold: float = 0.5) -> dict:
        """Return terms with inclusion probability above threshold, per species."""
        result = {}
        n_species = self.coefficients.shape[1]
        species_names = self.config.feature_names or [f"x{i}" for i in range(n_species)]

        for j in range(n_species):
            terms = []
            for i, fname in enumerate(self.feature_names):
                if self.inclusion_probabilities[i, j] >= threshold:
                    terms.append((
                        fname,
                        self.coefficients[i, j],
                        self.inclusion_probabilities[i, j],
                    ))
            result[species_names[j]] = terms
        return result


# ======================================================================
# Fitting functions
# ======================================================================

def fit_sindy(
    X: np.ndarray,
    t: np.ndarray,
    config: Optional[SINDyConfig] = None,
    X_dot: Optional[np.ndarray] = None,
) -> SINDyResult:
    """Fit a single SINDy model.

    Parameters
    ----------
    X : array (n_timepoints, n_species)
    t : array (n_timepoints,)
    config : SINDyConfig
    X_dot : optional pre-computed derivatives

    Returns
    -------
    SINDyResult
    """
    if config is None:
        config = SINDyConfig()

    model = ps.SINDy(
        feature_library=config.build_library(),
        optimizer=config.build_optimizer(),
        differentiation_method=config.build_differentiator(),
        feature_names=config.feature_names,
    )

    if X_dot is not None:
        model.fit(X, t=t, x_dot=X_dot)
    else:
        model.fit(X, t=t)

    coefs = model.coefficients()  # (n_species, n_library_terms)
    fnames = model.get_feature_names()

    return SINDyResult(
        model=model,
        coefficients=coefs.T,  # transpose to (n_library_terms, n_species)
        feature_names=fnames,
    )


def fit_esindy(
    X: np.ndarray,
    t: np.ndarray,
    config: Optional[SINDyConfig] = None,
    X_dot: Optional[np.ndarray] = None,
    n_bootstraps: int = 100,
    aggregation: Literal["mean", "median"] = "median",
    inclusion_threshold: float = 0.5,
    seed: int = 42,
) -> ESINDyResult:
    """Fit an ensemble of SINDy models via bagging.

    Parameters
    ----------
    X : array (n_timepoints, n_species)
    t : array (n_timepoints,)
    config : SINDyConfig
    X_dot : optional pre-computed derivatives
    n_bootstraps : number of bootstrap samples
    aggregation : 'mean' (bagging) or 'median' (bragging)
    inclusion_threshold : threshold for zeroing low-probability terms
    seed : random seed

    Returns
    -------
    ESINDyResult
    """
    if config is None:
        config = SINDyConfig()

    rng = np.random.default_rng(seed)
    m = X.shape[0]

    # First fit to get library size and feature names
    ref_result = fit_sindy(X, t, config, X_dot)
    n_lib = len(ref_result.feature_names)
    n_species = X.shape[1]

    all_coefs = np.zeros((n_bootstraps, n_lib, n_species))

    for b in range(n_bootstraps):
        # Bootstrap: sample m rows with replacement
        idx = rng.choice(m, size=m, replace=True)
        X_b = X[idx]
        t_b = t[idx]
        sort_order = np.argsort(t_b)
        X_b = X_b[sort_order]
        t_b = t_b[sort_order]

        # Remove duplicate timepoints (can cause issues)
        unique_mask = np.concatenate([[True], np.diff(t_b) > 0])
        X_b = X_b[unique_mask]
        t_b = t_b[unique_mask]

        if len(t_b) < 5:
            continue

        Xdot_b = None
        if X_dot is not None:
            Xdot_b = X_dot[idx][sort_order][unique_mask]

        try:
            result = fit_sindy(X_b, t_b, config, Xdot_b)
            all_coefs[b] = result.coefficients
        except Exception:
            # Some bootstraps may fail — skip them
            continue

    # Compute inclusion probabilities
    nonzero = all_coefs != 0  # (n_bootstraps, n_lib, n_species)
    inclusion_probs = nonzero.mean(axis=0)  # (n_lib, n_species)

    # Aggregate coefficients
    if aggregation == "mean":
        agg_coefs = all_coefs.mean(axis=0)
    else:  # median
        agg_coefs = np.median(all_coefs, axis=0)

    # Threshold by inclusion probability
    agg_coefs[inclusion_probs < inclusion_threshold] = 0.0

    coef_std = all_coefs.std(axis=0)

    return ESINDyResult(
        coefficients=agg_coefs,
        feature_names=ref_result.feature_names,
        all_coefficients=all_coefs,
        inclusion_probabilities=inclusion_probs,
        coefficient_std=coef_std,
        config=config,
        aggregation=aggregation,
    )


# ======================================================================
# Ensemble forecasting
# ======================================================================

def ensemble_forecast(
    esindy_result: ESINDyResult,
    x0: np.ndarray,
    t_eval: np.ndarray,
    n_models: int = 50,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ensemble forecasts and compute variance.

    Draws n_models coefficient sets from the bootstrap ensemble,
    forward-simulates each, and returns mean + variance trajectories.

    Parameters
    ----------
    esindy_result : ESINDyResult
    x0 : initial condition (n_species,)
    t_eval : timepoints for simulation
    n_models : how many ensemble members to simulate
    seed : random seed for drawing models

    Returns
    -------
    mean_traj : array (n_timepoints, n_species)
    var_traj : array (n_timepoints, n_species)
    """
    rng = np.random.default_rng(seed)
    all_coefs = esindy_result.all_coefficients  # (n_bootstraps, n_lib, n_species)
    n_bootstraps = all_coefs.shape[0]

    config = esindy_result.config
    library = config.build_library()

    # Collect valid forecasts
    trajectories = []
    indices = rng.choice(n_bootstraps, size=n_models, replace=True)

    for idx in indices:
        coef_matrix = all_coefs[idx]  # (n_lib, n_species)

        def rhs(t, x, coefs=coef_matrix):
            # Build library from current state
            x_row = x.reshape(1, -1)
            lib_features = library.fit_transform(x_row)  # (1, n_lib)
            return (lib_features @ coefs).flatten()

        try:
            sol = solve_ivp(
                rhs,
                (t_eval[0], t_eval[-1]),
                x0,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-8,
                atol=1e-10,
                max_step=0.1,
            )
            if sol.success:
                trajectories.append(sol.y.T)
        except Exception:
            continue

    if len(trajectories) == 0:
        n_t = len(t_eval)
        n_s = len(x0)
        return np.full((n_t, n_s), np.nan), np.full((n_t, n_s), np.nan)

    traj_array = np.stack(trajectories)  # (n_valid, n_t, n_species)
    mean_traj = traj_array.mean(axis=0)
    var_traj = traj_array.var(axis=0)

    return mean_traj, var_traj
