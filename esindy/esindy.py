"""
E-SINDy: Ensemble Sparse Identification of Nonlinear Dynamics

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
from .SINDy_PI import SINDyPI
from pysindy.optimizers import EnsembleOptimizer



# Configuration
@dataclass
class SINDyConfig:
    """Configuration for PySINDy model construction."""

    # Library selection
    library_type: str = "polynomial"   # or "pde"
    
    # Polynomial library setting
    poly_degree: int = 2
    include_interaction: bool = True
    include_bias: bool = False  # constant term in library

    # PDE / SINDyPI library settings
    library_functions: Optional[ps.CustomLibrary] = None
    function_names: Optional[list[str]] = None
    derivative_order: int = 1
    implicit_terms: bool = False
    temporal_grid: Optional[np.ndarray] = None
    
    # Optimizer settings
    optimizer: str = "STLSQ"  # or "SINDyPI" "SR3", "SSR"
    threshold: float = 0.05   
    alpha: float = 0.05       # ridge regularization
    max_iter: int = 20
    use_ensemble: bool = False
    n_models: int = 100
    replace: bool = True

    # Differentiation
    diff_method: str = "smoothed_finite_difference"  # or "_finite_difference"
    drop_endpoints: bool = True  # whether to drop endpoints after differentiation (can help with noise)
    
    # Input feature names for the state variables
    feature_names: Optional[list[str]] = None
    
    def build_library(self):
        if self.library_type == "polynomial":
            return ps.PolynomialLibrary(
                degree=self.poly_degree,
                include_interaction=self.include_interaction,
                include_bias=self.include_bias,
            )
        elif self.library_type == "pde":
            if self.library_functions is None:
                raise ValueError("For library_type='pde', provide library_functions and optionally function_names.")

            if self.implicit_terms and self.temporal_grid is None:
                raise ValueError("temporal_grid must be provided when implicit_terms=True.")
            
            # Ideally user passes  a feature-library object to be used directly
            if hasattr(self.library_functions, "fit") and hasattr(self.library_functions, "transform"):
                base_lib = self.library_functions
            
            #  but if they pass a list of functions instead we can wrap it in a CustomLibrary for them
            else:
                base_lib = ps.CustomLibrary(
                    library_functions=self.library_functions,
                    function_names=self.function_names,
                )
                
            return ps.PDELibrary(
                function_library=base_lib,
                derivative_order=self.derivative_order,
                include_bias=self.include_bias,
                implicit_terms=self.implicit_terms,
                temporal_grid=self.temporal_grid,
                include_interaction=self.include_interaction,
            )
        else:
            raise ValueError(f"Unknown library_type: {self.library_type}")

    def build_optimizer(self):
        if self.optimizer == "STLSQ":
            base_opt = ps.STLSQ(
                threshold=self.threshold,
                alpha=self.alpha,
                max_iter=self.max_iter,
            )
        elif self.optimizer == "SINDyPI":
            base_opt = SINDyPI(
                threshold=self.threshold,
                alpha=self.alpha,
                max_iter=self.max_iter,
            )
        else:
            raise ValueError(f"Optimizer '{self.optimizer}' not yet supported")

        if self.use_ensemble:
            return ps.EnsembleOptimizer(
                opt=base_opt,
                bagging=True,
                n_models=self.n_models,
                replace=self.replace,
            )
        return base_opt
    
    def build_differentiator(self):
        if self.diff_method == "finite_difference":
            return ps.FiniteDifference(drop_endpoints=self.drop_endpoints)
        elif self.diff_method == "smoothed_finite_difference":
            return ps.SmoothedFiniteDifference(drop_endpoints=self.drop_endpoints)
        else:
            raise ValueError(f"Unknown diff method: {self.diff_method}")


# Result containers
@dataclass
class SINDyResult:
    """Result from a single SINDy fit."""
    model: ps.SINDy
    coefficients: np.ndarray   # (n_features_in_library, n_species)


@dataclass
class ESINDyResult:
    """Result from ensemble SINDy fitting."""

    # Aggregated model
    coefficients: np.ndarray        # (n_library_terms, n_species) — aggregated

    # Ensemble statistics
    all_coefficients: np.ndarray    # (n_bootstraps, n_library_terms, n_species)
    inclusion_probabilities: np.ndarray  # (n_library_terms, n_species)
    coefficient_std: np.ndarray     # (n_library_terms, n_species)

    # Config used
    config: SINDyConfig = field(default_factory=SINDyConfig)
    aggregation: str = "mean"

# Fitting functions

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
    )

    # bare minimum fit requires time; keywords only if provided by user
    fit_kwargs = {
        "t": t,
        "feature_names": config.feature_names,
    }

    # if we have precomputed derivatives pass them to avoid redundancy
    if X_dot is not None:
        fit_kwargs["x_dot"] = X_dot

    model.fit(X, **fit_kwargs)

    coefs = model.coefficients()  # (n_species, n_library_terms)

    return SINDyResult(
        model=model,
        coefficients=coefs.T,  # transpose to (n_library_terms, n_species)
    )


def fit_esindy(
    X: np.ndarray,
    t: np.ndarray,
    config: Optional[SINDyConfig] = None,
    aggregation: Literal["mean", "median"] = "median",
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

    if not config.use_ensemble:
        raise ValueError("config.use_ensemble must be True to use fit_esindy")

    result = fit_sindy(X, t, config)
    ensemble_opt = result.model.optimizer
    assert isinstance(ensemble_opt, EnsembleOptimizer)      
    coef_list = np.array(ensemble_opt.coef_list)  # (n_models, n_species, n_lib)
    coef_list = coef_list.transpose(0, 2, 1)                # (n_models, n_lib, n_species)

    inclusion_probs = (coef_list != 0).mean(axis=0)         # (n_lib, n_species)
    coef_std = coef_list.std(axis=0)

    if aggregation == "mean":
        agg_coefs = coef_list.mean(axis=0)
    else:
        agg_coefs = np.median(coef_list, axis=0)

    return ESINDyResult(
        coefficients=agg_coefs,
        all_coefficients=coef_list,
        inclusion_probabilities=inclusion_probs,
        coefficient_std=coef_std,
        config=config,
        aggregation=aggregation,
    )


# Ensemble forecasting

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
                method="LSODA",
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

