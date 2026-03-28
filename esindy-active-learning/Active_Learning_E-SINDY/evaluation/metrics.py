"""
Evaluation metrics for comparing learned models against ground truth.

Metrics:
1. Coefficient recovery error (relative L2 norm)
2. Term accuracy (correct nonzero/zero identification)
3. Trajectory RMSE on held-out experiments
"""

import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps

from ..esindy import ESINDyResult, SINDyResult, SINDyConfig


def coefficient_error(
    true_coefs: np.ndarray,
    learned_coefs: np.ndarray,
) -> float:
    """Relative L2 coefficient error: ||true - learned||_2 / ||true||_2.

    Same as Eq. 11 in the E-SINDy paper.
    """
    norm_true = np.linalg.norm(true_coefs)
    if norm_true == 0:
        return np.linalg.norm(learned_coefs)
    return np.linalg.norm(true_coefs - learned_coefs) / norm_true


def term_accuracy(
    true_coefs: np.ndarray,
    learned_coefs: np.ndarray,
) -> dict:
    """Evaluate structural recovery: did we get the right nonzero terms?

    Returns
    -------
    dict with:
        'success_rate': fraction of terms correctly identified (zero or nonzero)
        'true_positives': correctly identified nonzero terms
        'false_positives': incorrectly included terms
        'false_negatives': missed true terms
        'precision': TP / (TP + FP)
        'recall': TP / (TP + FN)
    """
    true_nonzero = true_coefs != 0
    learned_nonzero = learned_coefs != 0

    tp = np.sum(true_nonzero & learned_nonzero)
    fp = np.sum(~true_nonzero & learned_nonzero)
    fn = np.sum(true_nonzero & ~learned_nonzero)
    tn = np.sum(~true_nonzero & ~learned_nonzero)

    total = true_coefs.size
    success_rate = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "success_rate": success_rate,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": precision,
        "recall": recall,
    }


def trajectory_rmse(
    target_system,
    learned_result,
    test_ICs: np.ndarray,
    t_eval: np.ndarray,
    config: SINDyConfig = None,
) -> np.ndarray:
    """Compute RMSE of learned model trajectories vs. ground truth.

    Parameters
    ----------
    target_system : TargetSystem instance (oracle)
    learned_result : ESINDyResult or SINDyResult
    test_ICs : array (n_test, n_species) — held-out initial conditions
    t_eval : timepoints for simulation
    config : SINDy config (needed for library construction)

    Returns
    -------
    rmse_per_ic : array (n_test,) — RMSE for each test IC, averaged over
        species and timepoints.
    """
    if config is None:
        config = SINDyConfig()

    n_test = test_ICs.shape[0]
    rmse_per_ic = np.full(n_test, np.nan)

    # Get coefficients
    if isinstance(learned_result, ESINDyResult):
        coefs = learned_result.coefficients
    elif isinstance(learned_result, SINDyResult):
        coefs = learned_result.coefficients
    else:
        raise TypeError(f"Unexpected result type: {type(learned_result)}")

    library = config.build_library()

    for i, x0 in enumerate(test_ICs):
        # Ground truth trajectory
        true_traj = target_system.simulate(x0, (t_eval[0], t_eval[-1]), t_eval)

        # Learned model trajectory
        def rhs(t, x, c=coefs):
            x_row = x.reshape(1, -1)
            lib = library.fit_transform(x_row)
            return (lib @ c).flatten()

        try:
            sol = solve_ivp(
                rhs,
                (t_eval[0], t_eval[-1]),
                x0,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-8,
                atol=1e-10,
            )
            if sol.success:
                pred_traj = sol.y.T
                rmse_per_ic[i] = np.sqrt(np.mean((true_traj - pred_traj) ** 2))
        except Exception:
            continue

    return rmse_per_ic


def evaluate_al_run(
    logs: list,
    true_coefs: np.ndarray,
) -> dict:
    """Extract learning curves from AL logs.

    Parameters
    ----------
    logs : list of ALStepLog
    true_coefs : ground truth coefficient matrix (n_lib, n_species)

    Returns
    -------
    dict with arrays indexed by step:
        'n_revealed': total measurements at each step
        'coef_error': relative coefficient error at each step
        'success_rate': term identification accuracy at each step
    """
    n_steps = len(logs)
    n_revealed = np.zeros(n_steps)
    coef_errors = np.full(n_steps, np.nan)
    success_rates = np.full(n_steps, np.nan)

    for i, log in enumerate(logs):
        n_revealed[i] = log.n_revealed

        if log.coefficients is not None:
            coef_errors[i] = coefficient_error(true_coefs, log.coefficients)
            ta = term_accuracy(true_coefs, log.coefficients)
            success_rates[i] = ta["success_rate"]

    return {
        "n_revealed": n_revealed,
        "coef_error": coef_errors,
        "success_rate": success_rates,
    }
