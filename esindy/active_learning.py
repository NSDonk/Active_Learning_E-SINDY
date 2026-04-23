"""
Active learning loop for E-SINDy.

Implements the main AL loop and query strategies:
- Random: uniformly random reveals
- Forecast disagreement: reveal where ensemble variance is highest

The loop operates on a pre-generated pool (data tensor with revealed mask)
and iteratively reveals measurements, refits E-SINDy, and logs metrics.
"""

from typing import Optional, Callable 
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
from .target_systems.base import TargetSystem
from pysindy.utils import AxesArray, comprehend_axes

# Helper functions! 

# to convert simulation outputs into SINDy ready format
def to_sindy(species_order, sim_result) -> tuple[np.ndarray, np.ndarray]:
        '''
        Convert HPT simulation output to format suitable for SINDy.
        '''
        # time vector
        t = sim_result['time']
        # preserve species order for consistency with config.feature_names and interpretability
        X = np.column_stack([sim_result['states'][s] for s in species_order])
        return X, t

# waaaay too many nested loops, just going to use this to dispatch the uncertainty metrics   
def uncertainty_fn(config: SINDyConfig):
    """
    Route to the right uncertainty function based on optimizer + ensemble flag.
    """
    is_pi = config.optimizer == "SINDyPI"
    if config.use_ensemble:
        return trajectory_uncertainty_ensemble_pi if is_pi else trajectory_uncertainty_ensemble
    else:
        return trajectory_uncertainty_single_pi if is_pi else trajectory_uncertainty_single
   
# Query strategies
def query_random(available_traj, rng):
    # uniform random selection
    return rng.integers(len(available_traj))

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

def trajectory_uncertainty_single(
    available_traj: np.ndarray,
    sindy_result: SINDyResult,
    points: int,
) -> tuple[float, np.ndarray]:
    """Per-timepoint residual between predicted and finite-difference derivatives.
    
    For the single-model case: score reflects where the fitted model disagrees
    with derivatives estimated directly from the trajectory data. High residuals
    indicate regions where the model's learned dynamics fail to explain the
    observed data — those are the informative points to query.
    
    Returns
    total_score : scalar score for IC selection (sum of per-timepoint residuals)
    mask : bool array (n_t,) — timepoints above mean residual
    """
    model = sindy_result.model

    # wrap for pysindy differentiation
    x_wrapped = AxesArray(available_traj, comprehend_axes(available_traj))
    
    # finite-difference derivatives separate independent of the fitted model
    x_dot_fd = np.asarray(model.differentiation_method(x_wrapped))
    
    # model-predicted derivatives
    x_dot_pred = np.asarray(model.predict(available_traj))
    
    # drop endpoints where finite-difference is unreliable
    x_dot_fd = x_dot_fd[1:-1]
    x_dot_pred = x_dot_pred[1:-1]

    # residual magnitude per timepoint, summed across species
    residual_per_timepoint = np.abs(x_dot_pred - x_dot_fd).sum(axis=1)  # (n_t - 2,)

    mask_interior = residual_per_timepoint > residual_per_timepoint.mean()

    # pad mask back to original length 
    mask = np.zeros(available_traj.shape[0], dtype=bool)
    mask[1:-1] = mask_interior

    # guard: if mask keeps too few points, fall back to top 20%
    min_points = max(20, int(points * 0.2))
    if mask.sum() < min_points:
        threshold = np.percentile(residual_per_timepoint, 80)
        mask_interior = residual_per_timepoint >= threshold
        mask = np.zeros(available_traj.shape[0], dtype=bool)
        mask[1:-1] = mask_interior

    return float(residual_per_timepoint.sum()), mask

def trajectory_uncertainty_ensemble(
    available_traj: np.ndarray,
    esindy_result: ESINDyResult,
    points: int,
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

    lib_features = np.asarray(library.fit_transform(available_traj))  # (n_t, n_lib)

    # predictions involve matrix multiplicationof each models predicted sparse coefficients
    # with the available data transformed by the library features
    preds = np.array([
        (lib_features @ all_coefs[i])
        for i in range(n_models)
    ])  # (n_models, n_t, n_lib)

    var_per_timepoint = preds.var(axis=0).sum(axis=1)  # (n_t,)
    mask = var_per_timepoint > var_per_timepoint.mean()
    
    # guard: if mask keeps too few points, fall back to top 20%
    min_points = max(20, int(points * 0.2))
    if mask.sum() < min_points:
        threshold = np.percentile(var_per_timepoint, 80)
        mask = var_per_timepoint >= threshold

    return float(var_per_timepoint.sum()), mask

def trajectory_uncertainty_ensemble_pi(
    available_traj: np.ndarray,
    esindy_result: ESINDyResult,
    points: int,
) -> tuple[float, np.ndarray]:
    """
    Ensemble SINDy-PI: variance across library terms.
    Variance is taken across models, summed across library terms and timepoints.
    """
    config = esindy_result.config
    library = config.build_library()
    all_coefs = esindy_result.all_coefficients  # (n_models, n_lib, n_lib) for PI
    n_models = all_coefs.shape[0]

    lib_features = np.asarray(library.fit_transform(available_traj))  # (n_t, n_lib)
    n_lib = lib_features.shape[1]

    # reconstruct each targeted library term per model
    # reconstruction for candidate j: drop col j from lib_features, multiply by sparse coefficient vector (with 0 at j)
    preds = np.array([
        lib_features @ all_coefs[i]  # (n_t,n_lib)
        for i in range(n_models)
    ])  # (n_models, n_t, n_lib)

    var_per_timepoint = preds.var(axis=0).sum(axis=1)  # (n_t,)
    mask = var_per_timepoint > var_per_timepoint.mean()

    min_points = max(20, int(points * 0.2))
    if mask.sum() < min_points:
        threshold = np.percentile(var_per_timepoint, 80)
        mask = var_per_timepoint >= threshold

    return float(var_per_timepoint.sum()), mask


def trajectory_uncertainty_single_pi(
    available_traj: np.ndarray,
    sindy_result: SINDyResult,
    points: int,
) -> tuple[float, np.ndarray]:
    """Per-timepoint reconstruction residual for single-model SINDy-PI.
    
    For each candidate library term that the model solved for, SINDy-PI
    learned an implicit equation candidate_j = candidate_library_{-j} x sparse_coefficient_vector_j. 
    The residual candidate_j - candidate_library_{-j} x sparse_coefficient_vector_j.
    measures how well the fitted implicit equation holds on new data 
    regions where residuals are large are where the model's implicit relationships fail.
    
    Returns
    -------
    total_score : scalar score for IC selection
    mask : bool array (n_t,) — timepoints above mean residual
    """
    config = sindy_result.config
    library = config.build_library()
    coefs = sindy_result.coefficients  # (n_lib, n_lib)

    lib_features = np.asarray(library.fit_transform(available_traj))  # (n_t, n_lib)
    n_lib = lib_features.shape[1]

    # identify candidates that were actually solved (non-zero coefficient row)
    active = np.array([not np.all(coefs[j] == 0) for j in range(n_lib)])
    if not active.any():
        # no candidates solved — fall back to uniform score, full-trajectory mask
        n_t = available_traj.shape[0]
        return 0.0, np.ones(n_t, dtype=bool)

    # for each active candidate j:
    # LHS = candidate_j(x(t)) = lib_features[:, j]
    # RHS = candidate_library_{-j}(x(t)) · sparse_coefficient_vector_j = lib_features @ coefs[:, j]   
    # (since sparse_coefficient_vector_j has 0 at position j)
    # residual = |LHS - RHS|
    reconstructions = lib_features @ coefs[:, active]  # (n_t, n_active)
    targets = lib_features[:, active]                  # (n_t, n_active)
    residuals = np.abs(targets - reconstructions)      # (n_t, n_active)

    score_per_timepoint = residuals.sum(axis=1)  # (n_t,)

    mask = score_per_timepoint > score_per_timepoint.mean()

    # guard: if mask keeps too few points, fall back to top 20%
    min_points = max(20, int(points * 0.2))
    if mask.sum() < min_points:
        threshold = np.percentile(score_per_timepoint, 80)
        mask = score_per_timepoint >= threshold

    return float(score_per_timepoint.sum()), mask

        
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


def active_learning_loop(
    target_sys: TargetSystem,
    params: Optional[dict] | None,   
    ic_pool: list[dict],             # pool of candidate initial conditions
    config: SINDyConfig,
    mode: str,                       # trajectory_uncertainty or random_query
    t_span: np.ndarray,
    masking: bool = False,
    n_init: int = 3,                 # initial random labeled set size
    n_queries: int = 10,             # number of active learning iterations
    n_test: int = 5,                   # ICs held out from ic_pool for final eval
    seed: int = 42,
) -> dict:
            
    rng = np.random.default_rng(seed)
    
    # carve out test set before anything else
    test_idx = rng.choice(len(ic_pool), size=n_test, replace=False)
    test_pool = [ic_pool[i] for i in test_idx]
    train_pool = [ic for i, ic in enumerate(ic_pool) if i not in test_idx]
    
    # initial random labeled set
    init_idx = rng.choice(len(train_pool), size=n_init, replace=False)
    init_labeled = [train_pool[i] for i in init_idx]
    remaining_pool = [ic for i, ic in enumerate(train_pool) if i not in init_idx]
    
    # simulate and concatenate the trajectories and time vectors in remaining pool
    available_traj = []
    available_time = []
    for i in remaining_pool:
        X_full = target_sys.simulate(y0=np.array(i), t_span=(t_span[0], t_span[-1]), t_eval=t_span) 
        available_traj.append(X_full)
        available_time.append(t_span)

    X_labeled_list = []
    holdouts = []  # list of (X_holdout, t_holdout) per labeled trajectory
    t_list = []
    
    # simulate and concatenate initial trajectories
    for ic in init_labeled:
        X_full = target_sys.simulate(y0=np.array(ic), t_span=(t_span[0], t_span[-1]), t_eval=t_span) 
        X_labeled_list.append(X_full)
        holdouts.append((X_full, t_span))
        t_list.append(t_span) # shared timepoints for train portion

    final_results = []
    prob_history = []
    scores_history = []
    for q in range(n_queries): 
        
        if len(remaining_pool) == 0: # if for some reason we have allowed more queries than available trajectories
            break
        
        # fit E-SINDy or SINDy on current labeled set
        if mode == 'trajectory_uncertainty':
            if config.use_ensemble:
                result = fit_esindy(X_labeled_list, t_list, config)
                final_results.append(result)
                prob_history.append(result.inclusion_probabilities)
            else:
                result = fit_sindy(X_labeled_list, t_list, config)
                final_results.append(result)
                
            uncertainty_function = uncertainty_fn(config)
                
            # compute highest uncertainty IC
            # store scores and mask for holdout time points 
            scores = []
            masks = []
            for traj in available_traj:
                s, m = uncertainty_function(traj, result, points=len(traj))
                scores.append(s)
                masks.append(m)

            max_uncertainty = np.argmax(scores) # idx of the ic for which the std was highest -> most disagreement 
            X_full, t_full = available_traj[max_uncertainty], available_time[max_uncertainty]
            mask = masks[max_uncertainty]
            
            # from the query for the highest variance IC, testing whether 
            # keeping only "informative" timepoints improves accuracy / convergence
            if not masking:
                X_labeled_list.append(X_full) # append all the timepoints from IC trajectory for which models were most uncertain
                t_list.append(t_full)
            else:
                X_labeled_list.append(X_full[mask]) # only append the specific timepoints for which the models were most uncertain
                t_list.append(t_full[mask])
                holdouts.append((X_full[~mask], t_full[~mask]))
            
            # Per iteration reporting
            scores_history.append(scores[max_uncertainty])
            print(f"Query {q+1}: score={scores[max_uncertainty]:.4f}, kept {mask.sum()}/{len(mask)} timepoints")
            
            # Remove queried point from pool of trajectories & times
            available_traj.pop(max_uncertainty)
            available_time.pop(max_uncertainty)
            remaining_pool.pop(max_uncertainty)
            
            # check to see if the model converged in which case break
            if inclusion_probs_converged(prob_history):
                print(f"Converged at iteration {q+1}")
                break
        
            
        else: # random_query
            if config.use_ensemble:
                result = fit_esindy(X_labeled_list, t_list, config)
                final_results.append(result)
                prob_history.append(result.inclusion_probabilities)
                
            else:
                result = fit_sindy(X_labeled_list, t_list, config)
                final_results.append(result)
            
            uncertainty_function = uncertainty_fn(config)
                
            # compute highest uncertainty IC
            # store scores for reporting
            
            # random query
            rand_idx = query_random(available_traj, rng)
            
            s, _ = uncertainty_function(available_traj[rand_idx], result, points=len(available_traj[rand_idx]))
                            
            X_labeled_list.append(available_traj[rand_idx]) 
            t_list.append(available_time[rand_idx])
            
            # Per iteration reporting
            scores_history.append(s)
            print(f"Query {q+1}: score={s:.4f}")
            
            # Remove queried point from pool of trajectories
            available_traj.pop(rand_idx)  
            available_time.pop(rand_idx)
            remaining_pool.pop(rand_idx) 
            
            # check to see if the model converged in which case break
            if inclusion_probs_converged(prob_history):
                print(f"Converged at iteration {q+1}")
                break   

            
    return {
            'results': final_results,           # list of ESINDy|SINDy Result per iteration
            'X_train': X_labeled_list,
            't_train': t_list,
            'test_pool': test_pool,       # held-out ICs for final evaluation
            'prob_history': prob_history,
            'scores_history': scores_history,
        }




# # AL loop HPT
# def HPT_active_learning_loop(
#     runner: Optional[Callable],      # biomolecular_controllers runner
#     target_sys: Optional[Callable],
#     params: Optional[dict] | None,   # HPTAxis has default params if None
#     ic_pool: list[dict],             # pool of candidate initial conditions
#     n_init: int = 3,                 # initial random labeled set size
#     n_queries: int = 10,             # number of active learning iterations
#     t_span: tuple = (0, 12e6),
#     points: int = 1000,
#     n_test: int = 5,                   # ICs held out from ic_pool for final eval
#     eval_every: int = 10,
#     config: Optional[SINDyConfig] = None,
#     seed: int = 42,
# ) -> dict:

#     if runner:
#         converter = HPTAxis() # for converting HPT sim output to SINDy format
        
#     rng = np.random.default_rng(seed)
    
#     # carve out test set before anything else
#     test_idx = rng.choice(len(ic_pool), size=n_test, replace=False)
#     test_pool = [ic_pool[i] for i in test_idx]
#     train_pool = [ic for i, ic in enumerate(ic_pool) if i not in test_idx]
    
#     # Step 1: initial random labeled set
#     init_idx = rng.choice(len(train_pool), size=n_init, replace=False)
#     labeled = [train_pool[i] for i in init_idx]
#     remaining_pool = [ic for i, ic in enumerate(train_pool) if i not in init_idx]
    
#     # simulate and concatenate initial trajectories
#     X_labeled_list = []
#     holdouts = []  # list of (X_holdout, t_holdout) per labeled trajectory
#     t_list = []
    

#     for ic in labeled:
#         if runner:
#             s = runner.run_deterministic('HPT_full', t_span=t_span, points=points, params=params, ic=ic)
#             X_full, t_full = converter.hpt_to_sindy(s)
#             X_labeled_list.append(X_full)
#             holdouts.append((X_full, t_full))
#             t_list.append(t_full) # shared timepoints for train portion
#         else: 
            
           
    
#     results = []
#     prob_history = []
#     scores_history = []
#     for q in range(n_queries):
#         # Step 2: fit E-SINDy or SINDy on current labeled set
#         if config.use_ensemble:
#             e_result = fit_esindy(X_labeled_list, t_list, config)
#         else:
#             sindy_result = fit_sindy(X_labeled_list, t_list, config)
            
#         results.append(e_result)
#         prob_history.append(e_result.inclusion_probabilities)
        
#         if inclusion_probs_converged(prob_history):
#             print(f"Converged at iteration {q+1}")
#             break
        
#         # Step 3: compute highest uncertainty IC
#         scores = []
#         trajectories = []
#         for ic in remaining_pool:
#             s = runner.run_deterministic('HPT_full', t_span=t_span, points=points, params=params, ic=ic)
#             X_full, t_full = converter.hpt_to_sindy(s)
#             score, mask = trajectory_uncertainty(X_full, e_result)
#             scores.append(score)
#             trajectories.append((X_full, t_full, mask))

#         # Step 4: query highest variance IC, keep only informative timepoints
#         best_idx = np.argmax(scores)
#         best_ic = remaining_pool.pop(best_idx)
#         X_full_new, t_full_new, mask = trajectories[best_idx]

#         X_labeled_list.append(X_full_new[mask])
#         t_list.append(t_full_new[mask])
#         holdouts.append((X_full_new[~mask], t_full_new[~mask]))
#         scores_history.append(scores[best_idx])
#         print(f"Query {q+1}: score={scores[best_idx]:.4f}, kept {mask.sum()}/{len(mask)} timepoints")  
        
#         converged = inclusion_probs_converged(prob_history)

#         if converged:
#             break

#     return {
#             'results': results,           # list of ESINDyResult per iteration
#             'test_pool': test_pool,       # held-out ICs for final evaluation
#             'prob_history': prob_history,
#             'scores_history': scores_history,
#         }