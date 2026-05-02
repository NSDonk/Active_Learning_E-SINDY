"""
Active learning loop for E-SINDy.

Implements the main AL loop and 4 query strategies:
- Random: uniformly random selection (baseline)
- Query by committee disagreement (Ensemble SINDy): variance across bootstrap models' predicted derivatives at each timepoint
- Derivative residual (single SINDy model): |predicted difference - finite-difference derivative| per timepoint 
- Implicit residual (single SINDyPI model): per-candidate leave-one-out reconstruction error of the implicit equations
  *There's also an ensemble variant of the implicit residual for SINDy-PI*

The loop operates on a pre-generated pool of initial conditions, trajectories are simulated up front, then iteratively selected, 
appended to the labeled set, and the model is refit. Inclusion-probability stability is used as a stopping rule for ensemble runs.

"""
from typing import Optional, Callable

import numpy as np
from tqdm import tqdm
import joblib
from pathlib import Path
from datetime import datetime
import warnings
from scipy.integrate import solve_ivp
from pysindy.utils import AxesArray, comprehend_axes

from .SINDy_configs import SINDyConfig
from .target_systems.base import TargetSystem
from .fit import (
    ESINDyResult,
    SINDyResult,
    fit_sindy,
    fit_esindy,   
)

# Helper functions! 
# Generating initial condition pools
def generate_ic_pool(ranges: dict[str, tuple[float, float]], n_candidates: int = 50, seed: int = 626,) -> list[np.ndarray]:
    ''' 
    Returns a list of initial conditions from which to generate the pool of trajectories
    for active learning query selection. Note: user must pass ranges as a dictionary with
    species in the same order as config.feature_names.
    '''
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(n_candidates):
        ic = np.array([rng.uniform(low, high) for low, high in ranges.values()])
        pool.append(ic)
    return pool

def save_run(run: dict, target_sys: TargetSystem, config: SINDyConfig, mode: str, out_dir: str | Path = "al_runs", tag: str | None = None, ) -> Path:
    """
    Save the data needed to plot/analyze an active learning run.
    Drops the live pysindy model objects which hold unpicklable lambas from the CustomLibrary,
    keeps coefficients, features, histories, training data and the IC pool. If the live model
    is needed later then it can be refit  from the save X_train/t_train.

    Parameters
    ----------
    run : dict
        The dict returned by active_learning_loop.
    target_sys, config, mode
        Persisted alongside the results so plotting code can reconstruct
        what was run without consulting external state.
    out_dir : path
        Directory to write into. Created if missing.
    tag : optional string
        Whatever additional naming convention you like. If None, a timestamp
        is used.

    Returns
    -------
    Path to the written file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # just storing the name of the target system
    sys_name = type(target_sys).__name__
    optim = config.optimizer
    ens = "ens" if config.use_ensemble else "single"
    stamp = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    # construct the file name
    fname = f"{sys_name}_{optim}_{ens}_{mode}_{stamp}.joblib"
    path = out_dir / fname
    
    payload = {
        "results": [
            {"coefficients": r.model.coefficients(),
             "feature_names": r.model.get_feature_names()}
            for r in run["results"]
        ],
        "X_train": run["X_train"],
        "t_train": run["t_train"],
        "test_pool": run["test_pool"],
        "unqueried_ics": run["unqueried_ics "],
        "prob_history": run["prob_history"],
        "coefs_history": run.get("coefs_history", []),
        "scores_history": run["scores_history"],
        "target_sys": sys_name,
        "mode": mode,
    }
    joblib.dump(payload, path, compress=3)
    return path


def load_run(path: str | Path) -> dict:
    """Load a previously-saved AL run. Returns the full payload dict."""
    return joblib.load(path)


# Ensemble forecasting
def ensemble_forecast(
    esindy_result: ESINDyResult,
    x0: np.ndarray,
    t_eval: np.ndarray,
    n_models: int = 50,
    seed: int = 626,
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

# I had way too many nested loops, decided it was better to dispatch the uncertainty metrics   
def get_uncertainty_fn(config: SINDyConfig) -> Callable:
    """
    Route to the right uncertainty function based on optimizer + ensemble flag.
    
    Ensemble branches: rank trajectories by committee disagreement (variance 
    across bootstrap models' library-space predictions, summed over timepoints)
    
    Single model branches: rank trajectories by residual error between predicted
    derivative and approximated ground truth. For STLSQ it's x_predicted - x_finite_difference
    and for SINDyPI, it's per candidate implicit eq reconstruction error
    """
    is_pi = config.optimizer == "SINDyPI"
    if config.use_ensemble:
        return trajectory_uncertainty_ensemble_pi if is_pi else trajectory_uncertainty_ensemble
    else:
        return trajectory_uncertainty_single_pi if is_pi else trajectory_uncertainty_single
   
# Query strategies
def query_random(n: int, rng:np.random.Generator) -> int:
    '''
    Random Query Selection: uniformly sample a random integer from among the 
    length of the available remaining trajectories. Assumes n passed in equals
    len(available_trajectories).
    '''
    # uniform random selection
    return int(rng.integers(n))

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

def trajectory_uncertainty_single_pi(
    available_traj: np.ndarray,
    sindy_result: SINDyResult,
    points: int,
) -> tuple[float, np.ndarray]:
    """Per-timepoint reconstruction residual for single-model SINDy-PI.
    
    For each candidate library term that the model solved for, SINDy-PI
    learned an implicit equation candidate_j = candidate_library_{-j} x sparse_coefficient_vector_j. 
    The residual candidate_j - candidate_library_{-j} x sparse_coefficient_vector_j
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

def coefs_converged(
    coefs_history: list[np.ndarray],
    window: int = 3,
    support_threshold: float = 1e-6,
) -> bool:
    """
    Single-model convergence criterion: stop when the set of active library terms per species 
    has been identical for the last `window` iterations.

    Parameters
    ----------
    coefs_history : list of (n_lib, n_species) coefficient arrays
    window : how many recent iterations must agree
    support_threshold : magnitude below which a coefficient is treated as zero

    Returns
    -------
    True if support has been stable across the window.
    """
    if len(coefs_history) < window:
        return False
    
    # build list of coefficient matrices to be compared
    recent_supports = [(np.abs(coef) > support_threshold) for coef in coefs_history[-window:]]
    
    # pairwise comparison between 'oldest' and other coefficients
    return all(np.array_equal(recent_supports[0], s) for s in recent_supports[1:])

def active_learning_loop(
    target_sys: TargetSystem,     # Repressilator, Lotka Volterra etc
    params: Optional[dict],       # legacy from diff simulation, ignore
    ic_pool:  list[np.ndarray],   # pool of candidate initial conditions
    config: SINDyConfig,          # defines PySindy Model Construction
    mode: str,                    # query select strategy: 'trajectory_uncertainty' or 'random_query'
    t_span: np.ndarray,           # simulated trajectory runtime
    masking: bool = False,        # whether to mask 'uninformative' timepoints
    n_init: int = 3,              # initial random labeled set size
    n_queries: int = 10,          # defines upper limit on # of active learning iterations
    n_test: int = 5,              # ICs held out from ic_pool for final eval
    seed: int = 626,
) -> dict:
    
    """
    Pool-based active learning loop for SINDy / E-SINDy / SINDy-PI / E-SINDy-PI.

    Parameters
    ----------
    target_sys : TargetSystem
        Oracle ODE system; must implement .simulate(y0, t_span, t_eval).
    ic_pool : list of arrays
        Can pass the output of generate_ic_pool here
        Candidate initial conditions. n_test are reserved for evaluation,
        n_init seed the labeled set, the remainder form query pool.
    config : SINDyConfig
        User defined optimizer (STLSQ vs SINDy-PI), ensembling, function library, finite-differentiation method, etc.
    mode : str
        'trajectory_uncertainty' selects the highest-ranking IC each round
        'random_query' selects uniformly at random.
    t_span : np.ndarray
        Time grid used for every simulation.
    masking : bool
        If True, append only the timepoints flagged as informative by the
        uncertainty function (top-residual or top-variance points).
    n_init, n_queries, n_test : int
    seed : int
        Controls test/init splits and any RNG-using selection step.

    Returns
    -------
    dict with keys:
        'results' : list of (E)SINDyResult, one per iteration
        'X_train' : list of trajectories used to fit
        't_train' : list of time vectors used to fit
        'test_pool' : ICs held out for evaluation
        'prob_history' : list of inclusion-prob arrays (ensemble runs only)
        'scores_history' : list of scalar scores logged per query
    """
            
    rng = np.random.default_rng(seed)
    
    # carve out test set before anything else
    test_idx = rng.choice(len(ic_pool), size=n_test, replace=False)
    test_pool = [ic_pool[i] for i in test_idx]
    train_pool = [ic for i, ic in enumerate(ic_pool) if i not in test_idx]
    
    # initial random labeled set
    init_idx = rng.choice(len(train_pool), size=n_init, replace=False)
    init_labeled = [train_pool[i] for i in init_idx]
    remaining_pool = [ic for i, ic in enumerate(train_pool) if i not in init_idx]
    
    # simulate the remaining trajectories to be queried up front to build pool
    available_traj = []
    available_time = []
    available_ics = list(remaining_pool) # we'll return at the end to investigate unqueried ICs
    
    for ic in remaining_pool:
        X = target_sys.simulate(y0=np.array(ic), t_span=(t_span[0], t_span[-1]), t_eval=t_span) 
        available_traj.append(X)
        available_time.append(t_span)

    # instantiate lists for storing labeled trajectories and their timepoints
    X_labeled_list = []
    t_list = []
    
    # simulate and concatenate the initial trajectories, # defined by n_init parameter
    for ic in init_labeled:
        X = target_sys.simulate(y0=np.array(ic), t_span=(t_span[0], t_span[-1]), t_eval=t_span) 
        X_labeled_list.append(X)
        t_list.append(t_span) # shared timepoints for train portion

    # instantiate lists for collecting: 
    final_results = []  # stores fitted model at every iteration
    prob_history = []   # store per iteration inclusion probabilities for ensemble model covergence criteria and visualization 
    coefs_history = []  # store per iteration regression coefficients for single model covergence criteria and visualization 
    scores_history = [] # stores per iteration value of uncertainty metric
    
    uncertainty_function = get_uncertainty_fn(config)    
     
    with warnings.catch_warnings():
        # AxesArray labels 2 axes when SINDy-PI transiently re-wraps a 1D array
        # during library evaluation. Metadata-only, no numerical impact.
        warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysindy")
        
        pbar = tqdm(range(n_queries), desc=f"AL ({mode})", unit="query")
        for q in pbar:
            
            if len(available_traj) == 0: # if for some reason we have allowed more queries than available trajectories
                pbar.write(f"Pool exhausted at iteration {q}")
                break
        
            # fit E-SINDy or SINDy on current labeled set
            if config.use_ensemble:
                result = fit_esindy(X_labeled_list, t_list, config)
                prob_history.append(result.inclusion_probabilities)
                coefs_history.append(result.model.coefficients()) 
            else:
                result = fit_sindy(X_labeled_list, t_list, config)
                coefs_history.append(result.model.coefficients())  # (n_lib, n_species)
                
            # store fitted model
            final_results.append(result)    
            
            # temp storage uncertainty scores + masks, no need to rewrite lists each iteration of AL loop
            best_score = -np.inf
            best_idx = 0
            best_mask = None
            
            # select highest uncertainty IC from among available trajectories based on mode
            if mode == "trajectory_uncertainty":    
                for i, traj in enumerate(available_traj):
                    s, m = uncertainty_function(traj, result, points=len(traj))
                    if s > best_score:
                        best_score = s
                        best_idx= i
                        best_mask = m
            
            else:  # random_query
                best_idx = query_random(len(available_traj), rng)
                best_score, best_mask = uncertainty_function(available_traj[best_idx], result, points=len(available_traj[best_idx]))

            # append (optionally masked trajectory to labeled set)    
            X_full, t_full = available_traj[best_idx], available_time[best_idx]
                
            # when masking = true, keeping only "informative" timepoints  to see if it improves accuracy / convergence
            if masking:
                X_labeled_list.append(X_full[best_mask]) # only append the specific timepoints for which the models were most uncertain 
                t_list.append(t_full[best_mask])
            else:
                X_labeled_list.append(X_full) # append all the timepoints from IC trajectory for which models were most uncertain
                t_list.append(t_full)
                
            # Per iteration reporting
            scores_history.append(best_score)
            kept = best_mask.sum() if (masking and best_mask is not None) else len(t_full) # little janky but I can't stand pylance warnings so it stands
            pbar.set_postfix(score=f"{best_score:.2e}", kept=f"{kept}/{len(t_full)}")
                
            # Remove queried point from pool of trajectories & times
            available_traj.pop(best_idx)
            available_time.pop(best_idx)
            available_ics.pop(best_idx)

                            
            # check to see if the model converged (currently only relevant for ensemble)in which case break
            if config.use_ensemble: 
                if inclusion_probs_converged(prob_history):
                    pbar.write(f"Converged (inclusion probs) at iteration {q+1}")
                    break 
            else: 
                if coefs_converged(coefs_history):
                    pbar.write(f"Converged (coefficient support) at iteration {q+1}")
                    break
                
    return {
            'results': final_results,           # list of ESINDy|SINDy Result per iteration
            'X_train': X_labeled_list,
            't_train': t_list,
            'test_pool': test_pool,       # held-out ICs for final evaluation
            'prob_history': prob_history,
            'coefs_history': coefs_history,
            'scores_history': scores_history,
            'unqueried_ics': available_ics,
        }