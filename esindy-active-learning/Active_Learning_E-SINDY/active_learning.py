"""
Active learning loop for E-SINDy.

Implements the main AL loop and query strategies:
- Random: uniformly random reveals
- Forecast disagreement: reveal where ensemble variance is highest

The loop operates on a pre-generated pool (data tensor with revealed mask)
and iteratively reveals measurements, refits E-SINDy, and logs metrics.
"""

from typing import Callable, Optional, Literal
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from .esindy import (
    ESINDyResult,
    SINDyConfig,
    fit_sindy,
    fit_esindy,
    ensemble_forecast,
    SINDyResult,
)
from .utils import get_revealed_data, reveal_window, count_revealed


# ======================================================================
# Query strategies
# ======================================================================

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


def forecast_disagreement_query(
    pool: dict,
    esindy_result: ESINDyResult,
    window_size: int,
    rng: np.random.Generator,
    n_forecast_models: int = 30,
    **kwargs,
) -> tuple[int, int, int]:
    """Select query based on maximum ensemble forecast variance.

    For each experiment, forward-simulate the ensemble from the IC,
    compute per-(species, timepoint) variance, and return the unrevealed
    window with highest total variance.

    Returns
    -------
    (experiment_idx, species_idx, time_start_idx)
    """
    revealed = pool["revealed"]
    n_exp, n_t, n_species = revealed.shape
    t_eval = pool["t_eval"]
    ICs = pool["ICs"]

    best_score = -np.inf
    best_query = None

    for e in range(n_exp):
        # Forward-simulate ensemble from this experiment's IC
        _, var_traj = ensemble_forecast(
            esindy_result,
            ICs[e],
            t_eval,
            n_models=n_forecast_models,
            seed=int(rng.integers(1e6)),
        )

        # var_traj shape: (n_t, n_species)
        if np.all(np.isnan(var_traj)):
            continue

        # Score each unrevealed window
        for s in range(n_species):
            for t_start in range(0, n_t - window_size + 1, window_size):
                window_mask = revealed[e, t_start:t_start + window_size, s]
                if window_mask.all():
                    continue  # already revealed

                # Score = total variance in this window for this species
                score = np.nansum(var_traj[t_start:t_start + window_size, s])

                if score > best_score:
                    best_score = score
                    best_query = (e, s, t_start)

    if best_query is None:
        # Fallback to random if all forecasts failed
        return random_query(pool, esindy_result, window_size, rng)

    return best_query


# Registry of available strategies
QUERY_STRATEGIES = {
    "random": random_query,
    "forecast_disagreement": forecast_disagreement_query,
}


# ======================================================================
# AL loop
# ======================================================================

@dataclass
class ALStepLog:
    """Log entry for a single active learning step."""
    step: int
    n_revealed: int
    fraction_revealed: float
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


def run_active_learning(
    pool: dict,
    strategy: str = "forecast_disagreement",
    learner: Literal["sindy", "esindy"] = "esindy",
    config: Optional[SINDyConfig] = None,
    n_initial_experiments: int = 2,
    n_queries: int = 50,
    window_size: int = 10,
    n_bootstraps: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> ALResult:
    """Run the active learning loop.

    Parameters
    ----------
    pool : dict from TargetSystem.generate_pool()
    strategy : key in QUERY_STRATEGIES
    learner : 'sindy' for single model baseline, 'esindy' for ensemble
    config : SINDy configuration
    n_initial_experiments : number of full trajectories revealed at init
    n_queries : number of AL iterations
    window_size : temporal window size per query
    n_bootstraps : for E-SINDy ensemble
    seed : random seed
    verbose : show progress bar

    Returns
    -------
    ALResult with per-step logs and final model.
    """
    if config is None:
        config = SINDyConfig()

    if strategy not in QUERY_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Available: {list(QUERY_STRATEGIES.keys())}"
        )

    query_fn = QUERY_STRATEGIES[strategy]
    rng = np.random.default_rng(seed)
    result = ALResult(strategy=strategy, config=config)

    n_exp = pool["X_full"].shape[0]

    # ------------------------------------------------------------------
    # Initialization: reveal a few full trajectories
    # ------------------------------------------------------------------
    init_indices = rng.choice(n_exp, size=n_initial_experiments, replace=False)
    for idx in init_indices:
        pool["revealed"][idx, :, :] = True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    iterator = range(n_queries)
    if verbose:
        iterator = tqdm(iterator, desc=f"AL [{strategy}]")

    current_esindy = None

    for step in iterator:
        # Fit model on current revealed data
        X, X_dot, t = get_revealed_data(pool)

        if X.shape[0] < 10:
            # Not enough data yet — do a random query
            query = random_query(pool, None, window_size, rng)
        else:
            if learner == "esindy":
                try:
                    current_esindy = fit_esindy(
                        X, t, config=config, X_dot=X_dot,
                        n_bootstraps=n_bootstraps, seed=int(rng.integers(1e6)),
                    )
                except Exception as e:
                    if verbose:
                        tqdm.write(f"  Step {step}: fit failed ({e}), random query")
                    query = random_query(pool, None, window_size, rng)
                    reveal_window(pool, *query, window_size)
                    continue
            else:
                # Single SINDy — no uncertainty, always random query
                try:
                    fit_sindy(X, t, config=config, X_dot=X_dot)
                except Exception:
                    pass

            # Select next query
            if learner == "esindy" and strategy != "random" and current_esindy is not None:
                query = query_fn(
                    pool, current_esindy, window_size, rng,
                )
            else:
                query = random_query(pool, current_esindy, window_size, rng)

        # Reveal the queried window
        reveal_window(pool, *query, window_size)

        # Log
        n_rev = count_revealed(pool)
        frac = pool["revealed"].mean()

        log_entry = ALStepLog(
            step=step,
            n_revealed=n_rev,
            fraction_revealed=frac,
            query=query,
        )

        if current_esindy is not None:
            log_entry.coefficients = current_esindy.coefficients.copy()
            log_entry.inclusion_probabilities = current_esindy.inclusion_probabilities.copy()
            log_entry.coefficient_std = current_esindy.coefficient_std.copy()

        result.logs.append(log_entry)

        if verbose:
            iterator.set_postfix(
                revealed=f"{frac:.1%}",
                n_pts=n_rev,
            )

    # Final fit
    X, X_dot, t = get_revealed_data(pool)
    if X.shape[0] >= 10:
        if learner == "esindy":
            result.final_model = fit_esindy(
                X, t, config=config, X_dot=X_dot,
                n_bootstraps=n_bootstraps, seed=seed,
            )
        result.final_sindy_model = fit_sindy(X, t, config=config, X_dot=X_dot)

    return result
