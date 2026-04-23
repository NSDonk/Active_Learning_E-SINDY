from typing import Optional, Literal
from dataclasses import dataclass, field

import numpy as np
import pysindy as ps
from .SINDy_configs import SINDyConfig
from pysindy.optimizers import EnsembleOptimizer

# Result containers
@dataclass
class SINDyResult:
    """Result from a single SINDy fit."""
    model: ps.SINDy
    coefficients: np.ndarray   # (n_features_in_library, n_species)
    config: SINDyConfig = field(default_factory=SINDyConfig)


@dataclass
class ESINDyResult:
    """Result from ensemble SINDy fitting."""

    # Aggregated model
    coefficients: np.ndarray        # (n_library_terms, n_species) — aggregated
    model: ps.SINDy

    # Ensemble statistics
    all_coefficients: np.ndarray    # (n_bootstraps, n_library_terms, n_species)
    inclusion_probabilities: np.ndarray  # (n_library_terms, n_species)
    coefficient_std: np.ndarray     # (n_library_terms, n_species)

    # Config used
    config: SINDyConfig = field(default_factory=SINDyConfig)
    aggregation: str = "mean"
    

# Fitting functions

def fit_sindy(
    X:  np.ndarray | list[np.ndarray],
    t:  np.ndarray | list[np.ndarray],
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
        config=config,
    )


def fit_esindy(
    X:  np.ndarray | list[np.ndarray],
    t:  np.ndarray | list[np.ndarray],
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
    aggregation : 'mean' (bagging) or 'median' (bragging)
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
        model=result.model,
        config=config,
        aggregation=aggregation,
    )