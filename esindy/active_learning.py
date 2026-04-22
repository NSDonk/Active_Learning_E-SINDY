"""
Active learning loop for E-SINDy.

Implements the main AL loop and query strategies:
- Random: uniformly random reveals
- Forecast disagreement: reveal where ensemble variance is highest

The loop operates on a pre-generated pool (data tensor with revealed mask)
and iteratively reveals measurements, refits E-SINDy, and logs metrics.
"""

from typing import Optional 
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm
import sympy as sp
from .SINDy_configs import SINDyConfig
from .fit import (
    ESINDyResult,
    SINDyResult,
    fit_sindy,
    fit_esindy,   
)
from .SINDyPI_solve import simulate_sindy_pi, ensemble_forecast_sindy_pi
from .utils import get_revealed_data
from .target_systems.hpt_axis import HPTAxis
from scipy.integrate import solve_ivp

@dataclass
class ALStepLog:
    """Log entry for a single active learning step."""
    step: int
    query: tuple  # (experiment, species, time_start)
    coefficients: Optional[np.ndarray] = None
    inclusion_probabilities: Optional[np.ndarray] = None
    coefficient_std: Optional[np.ndarray] = None


@dataclass
class ALResult:
    """Full result of an active learning run."""
    logs: list[ALStepLog] = field(default_factory=list)
    final_model: Optional[ESINDyResult] = None
    final_sindy_model: Optional[SINDyResult] = None
    strategy: str = ""
    config: Optional[SINDyConfig] = None

# Query strategies

def random_query(
    pool: dict,
    esindy_result: Optional[ESINDyResult],
    window_size: int,
    rng: np.random.Generator,
    **kwargs,
) -> tuple[int, int, int]:
    """Randomly select an unrevealed (experiment, species, time_start) to query.

    Returns
    -------
    (experiment_idx, species_idx, time_start_idx)
    """
    revealed = pool["revealed"]  # (n_exp, n_t, n_species)
    n_exp, n_t, n_species = revealed.shape

    # Find candidate (exp, species, time_start) that are not fully revealed
    candidates = []
    for e in range(n_exp):
        for s in range(n_species):
            for t_start in range(0, n_t - window_size + 1, window_size):
                window = revealed[e, t_start:t_start + window_size, s]
                if not window.all():
                    candidates.append((e, s, t_start))

    if len(candidates) == 0:
        raise RuntimeError("No unrevealed entries remaining in pool.")

    idx = rng.integers(len(candidates))
    return candidates[idx]

# for generating initial condition pools
def generate_ic_pool(
    ranges: dict,
    n_candidates: int = 50,
    seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(n_candidates):
        ic = {k: float(rng.uniform(low, high)) for k, (low, high) in ranges.items()}
        pool.append(ic)
    return pool

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

def trajectory_uncertainty(
    X_full: np.ndarray,
    esindy_result: ESINDyResult,
) -> tuple[float, np.ndarray]:
    """Compute ensemble variance across full trajectory.
    
    Returns
    -------
    total_variance : scalar score for IC selection
    mask : bool array (n_t,) — timepoints above mean variance
    t_queried, X_queried handled by caller using mask
    """
    config = esindy_result.config
    library = config.build_library()
    all_coefs = esindy_result.all_coefficients
    n_models = all_coefs.shape[0]

    lib_features = np.asarray(library.fit_transform(X_full))  # (n_t, n_lib)

    preds = np.array([
        (lib_features @ all_coefs[i])
        for i in range(n_models)
    ])  # (n_models, n_t, n_lib or n_species)

    var_per_timepoint = preds.var(axis=0).sum(axis=1)  # (n_t,)
    mask = var_per_timepoint > var_per_timepoint.mean()
    
    # guard: if mask keeps too few points, fall back to top 20%
    min_points = max(20, int(len(X_full) * 0.2))
    if mask.sum() < min_points:
        threshold = np.percentile(var_per_timepoint, 80)
        mask = var_per_timepoint >= threshold

    return float(var_per_timepoint.sum()), mask

def inclusion_probs_converged(
    history: list[np.ndarray],
    window: int = 3,
    tol: float = 0.01,
) -> bool:
    """Check if inclusion probabilities have stabilized over last `window` iterations."""
    if len(history) < window:
        return False
    recent = np.array(history[-window:])  # (window, n_lib, n_species)
    return float(np.std(recent, axis=0).mean()) < tol


# Registry of available strategies
QUERY_STRATEGIES = {
    "random": random_query,
    "trajectory_uncertainty": trajectory_uncertainty,
}

# AL loop
def active_learning_loop(
    runner,                          # biomolecular_controllers runner
    params: dict | None,             # HPTAxis has default params if None
    ic_pool: list[dict],             # pool of candidate initial conditions
    n_init: int = 3,                 # initial random labeled set size
    n_queries: int = 10,             # number of active learning iterations
    t_span: tuple = (0, 12e6),
    points: int = 1000,
    n_test: int = 5,                   # ICs held out from ic_pool for final eval
    eval_every: int = 10,
    config: Optional[SINDyConfig] = None,
    seed: int = 42,
) -> dict:

    converter = HPTAxis() # for converting HPT sim output to SINDy format
    rng = np.random.default_rng(seed)
    
    # carve out test set before anything else
    test_idx = rng.choice(len(ic_pool), size=n_test, replace=False)
    test_pool = [ic_pool[i] for i in test_idx]
    train_pool = [ic for i, ic in enumerate(ic_pool) if i not in test_idx]
    
    # Step 1: initial random labeled set
    init_idx = rng.choice(len(train_pool), size=n_init, replace=False)
    labeled = [train_pool[i] for i in init_idx]
    remaining_pool = [ic for i, ic in enumerate(train_pool) if i not in init_idx]
    
    # simulate and concatenate initial trajectories
    X_labeled_list = []
    holdouts = []  # list of (X_holdout, t_holdout) per labeled trajectory
    t_list = []
    

    for ic in labeled:
        s = runner.run_deterministic('HPT_full', t_span=t_span, points=points, params=params, ic=ic)
        X_full, t_full = converter.hpt_to_sindy(s)
        X_labeled_list.append(X_full)
        holdouts.append((X_full, t_full))
        t_list.append(t_full) # shared timepoints for train portion   
    
    results = []
    prob_history = []
    scores_history = []
    for q in range(n_queries):
        # Step 2: fit E-SINDy or SINDy on current labeled set
        if config.use_ensemble:
            e_result = fit_esindy(X_labeled_list, t_list, config)
        else:
            sindy_result = fit_sindy(X_labeled_list, t_list, config)
            
        results.append(e_result)
        prob_history.append(e_result.inclusion_probabilities)
        
        if inclusion_probs_converged(prob_history):
            print(f"Converged at iteration {q+1}")
            break
        
        # Step 3: compute highest uncertainty IC
        scores = []
        trajectories = []
        for ic in remaining_pool:
            s = runner.run_deterministic('HPT_full', t_span=t_span, points=points, params=params, ic=ic)
            X_full, t_full = converter.hpt_to_sindy(s)
            score, mask = trajectory_uncertainty(X_full, e_result)
            scores.append(score)
            trajectories.append((X_full, t_full, mask))

        # Step 4: query highest variance IC, keep only informative timepoints
        best_idx = np.argmax(scores)
        best_ic = remaining_pool.pop(best_idx)
        X_full_new, t_full_new, mask = trajectories[best_idx]

        X_labeled_list.append(X_full_new[mask])
        t_list.append(t_full_new[mask])
        holdouts.append((X_full_new[~mask], t_full_new[~mask]))
        scores_history.append(scores[best_idx])
        print(f"Query {q+1}: score={scores[best_idx]:.4f}, kept {mask.sum()}/{len(mask)} timepoints")  
        
        converged = inclusion_probs_converged(prob_history)

        if converged:
            break
        
        
        if (q + 1) % eval_every == 0:
            print(f"  [ensemble_forecast] Phase 1: solving sympy for {config.n_models} members...") # type: ignore
            if config is None:
                config = SINDyConfig()
                
            X_holdout, t_holdout = holdouts[-1]  # most recent trajectory's holdout
            x0_holdout = X_holdout[0]
            
            X_train_concat = np.concatenate(X_labeled_list, axis=0)
            t_train_concat = np.concatenate(t_list, axis=0)
            
            species_names = config.feature_names or e_result.model.feature_names or [f'x{i}' for i in range(e_result.model.n_features_in_)]

            if config.optimizer == "SINDyPI":
                mean_traj, var_traj = ensemble_forecast_sindy_pi(
                    e_result,
                    e_result.model,
                    X_train_concat, t_train_concat,
                    species_names,
                    x0_holdout, t_holdout,
                )
            else:
                mean_traj, var_traj = ensemble_forecast(
                    e_result, x0_holdout, t_holdout,
                )
            rmse = np.sqrt(np.mean((X_holdout - mean_traj)**2))
            print(f"Query {q+1}: RMSE={rmse:.4f}, mean_var={var_traj.mean():.4f}")
           
    
    return {
            'results': results,           # list of ESINDyResult per iteration
            'test_pool': test_pool,       # held-out ICs for final evaluation
            'prob_history': prob_history,
            'scores_history': scores_history,
        }


