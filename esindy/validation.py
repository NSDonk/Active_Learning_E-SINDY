"""
Post-hoc validation for active learning results.

Evaluates fitted SINDy / E-SINDy / SINDy-PI models on held-out test initial
conditions. Reuses SINDy-PI symbolic machinery from SINDyPI_solve.
"""

from typing import Optional, Union
import numpy as np
from pysindy.utils import AxesArray, comprehend_axes

from .fit import ESINDyResult, SINDyResult
from .target_systems.base import TargetSystem
from .active_learning import to_sindy, ensemble_forecast
from .SINDyPI_solve import (
    simulate_sindy_pi,
    ensemble_forecast_sindy_pi,
    select_best_sindy_pi_model,
)

Result = Union[SINDyResult, ESINDyResult]


def _derivative_rmse(
    result: Result,
    X_true: np.ndarray,
    t_true: np.ndarray,
    X_train: np.ndarray,
    t_train: np.ndarray,
) -> np.ndarray:
    """Per-species RMSE between predicted and finite-difference derivatives.

    For SINDy-PI, selects the best candidate per species using training data
    (via select_best_sindy_pi_model) before computing test-set predictions.
    """
    model = result.model
    config = result.config
    n_states = X_true.shape[1]

    x_wrapped = AxesArray(X_true, comprehend_axes(X_true))
    X_dot_true = np.asarray(model.differentiation_method(x_wrapped, t=t_true)[1:-1])
    preds_all = np.asarray(model.predict(X_true))[1:-1]

    if config.optimizer == "SINDyPI":
        # select on TRAIN data, evaluate on TEST data
        best = select_best_sindy_pi_model(model, X_train, t_train)
        preds = np.full_like(X_dot_true, np.nan)
        for k in range(n_states):
            if k in best:
                best_idx, _ = best[k]
                preds[:, k] = preds_all[:, best_idx]
        if np.isnan(preds).any():
            return np.full(n_states, np.nan)
    else:
        preds = preds_all  # already (n_t, n_states)

    return np.sqrt(np.mean((preds - X_dot_true) ** 2, axis=0))


def _forecast_rmse(
    result: Result,
    X_true: np.ndarray,
    t_true: np.ndarray,
    X_train: np.ndarray,
    t_train: np.ndarray,
    species_names: list[str],
) -> np.ndarray:
    """Per-species RMSE of forward-simulated trajectory vs. true."""
    config = result.config
    x0 = X_true[0]
    n_states = X_true.shape[1]
    nan_result = np.full(n_states, np.nan)

    try:
        if config.optimizer == "SINDyPI" and config.use_ensemble:
            mean_traj, _ = ensemble_forecast_sindy_pi(
                result, result.model,
                X_train, t_train,
                species_names, x0, t_true,
            )
        elif config.optimizer == "SINDyPI":
            out = simulate_sindy_pi(
                result.model, X_train, t_train,
                species_names, x0, t_true,
            )
            mean_traj = out['trajectory']
        elif config.use_ensemble:
            mean_traj, _ = ensemble_forecast(result, x0, t_true)
        else:
            mean_traj = result.model.simulate(x0, t_true)
    except Exception:
        return nan_result

    if mean_traj is None or np.any(np.isnan(mean_traj)):
        return nan_result

    return np.sqrt(np.mean((mean_traj - X_true) ** 2, axis=0))


def evaluate_results(
    results: list[Result],
    test_pool: list[tuple],
    target_sys: TargetSystem,
    t_span: np.ndarray,
    X_train: np.ndarray,
    t_train: np.ndarray,
    params: Optional[dict] = None,
    feature_names: Optional[list[str]] = None,
) -> dict:
    """Evaluate each iteration's fitted model on held-out test ICs.

    Parameters
    ----------
    results : list of fitted results (one per AL iteration)
    test_pool : held-out initial conditions (list of tuples)
    target_sys : ground-truth system for oracle simulation
    t_span : timepoints for oracle simulation
    X_train, t_train : concatenated training data from the final AL iteration,
        used for SINDy-PI best-candidate selection. Pass the same data used
        to fit results[-1].
    feature_names : species ordering; defaults to config.feature_names
    """
    if feature_names is None:
        feature_names = results[0].config.feature_names

    # Oracle trajectories — simulate once up front
    test_trajs = []
    for ic in test_pool:
        sim = target_sys.simulate(
            y0=np.array(ic),
            t_span=(t_span[0], t_span[-1]),
            t_eval=t_span,
        )
        test_trajs.append((sim, t_span))

    n_iter, n_test = len(results), len(test_pool)
    n_species = test_trajs[0][0].shape[1]

    deriv_rmse = np.full((n_iter, n_test, n_species), np.nan)
    fc_rmse = np.full((n_iter, n_test, n_species), np.nan)

    for i, result in enumerate(results):
        for j, (X_true, t_true) in enumerate(test_trajs):
            try:
                deriv_rmse[i, j] = _derivative_rmse(
                    result, X_true, t_true, X_train, t_train,
                )
            except Exception as e:
                print(f"[iter {i}, test {j}] derivative RMSE failed: {e}")
            try:
                fc_rmse[i, j] = _forecast_rmse(
                    result, X_true, t_true, X_train, t_train,
                    feature_names,
                )
            except Exception as e:
                print(f"[iter {i}, test {j}] forecast RMSE failed: {e}")

    return {
        "derivative_rmse": deriv_rmse,
        "forecast_rmse": fc_rmse,
        "mean_derivative_rmse": np.nanmean(deriv_rmse, axis=1),
        "mean_forecast_rmse": np.nanmean(fc_rmse, axis=1),
    }