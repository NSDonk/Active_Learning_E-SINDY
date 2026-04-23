"""
Plotting utilities for AL results.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    results_by_condition: dict,
    metric: str = "forecast_rmse",
    title: str = "",
    ax=None,
):
    """Plot learning curves (metric vs AL iteration) across conditions.

    Parameters
    ----------
    results_by_condition : dict
        Keys are condition labels (e.g., 'AL + SINDy', 'Random + SINDy').
        Values are dicts from evaluate_results containing 'mean_derivative_rmse'
        and 'mean_forecast_rmse' arrays of shape (n_iterations, n_species).
    metric : 'derivative_rmse' or 'forecast_rmse'
    title : plot title
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    metric_key = f"mean_{metric}"

    for label, eval_out in results_by_condition.items():
        curve = np.nanmean(eval_out[metric_key], axis=1)  # mean across species
        iterations = np.arange(1, len(curve) + 1)
        ax.plot(iterations, curve, marker='o', label=label, linewidth=2)

    ax.set_xlabel("Active learning iteration")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    return ax


def plot_inclusion_probabilities(
    prob_histories: dict,
    species_idx: int = 0,
    top_k_terms: int = 8,
    feature_names: list = None,
    title: str = "",
    ax=None,
):
    """Plot inclusion probabilities over AL iterations for the top-K terms.

    Parameters
    ----------
    prob_histories : dict
        Keys are condition labels. Values are lists of (n_lib, n_species) arrays,
        one per AL iteration.
    species_idx : which species' equation to plot (pick one for clarity)
    top_k_terms : how many library terms to show (picked by final probability)
    feature_names : library term names for legend
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # pick colors per condition, linestyles per term
    condition_colors = plt.cm.tab10.colors
    linestyles = ['-', '--', ':', '-.']

    for c_idx, (label, history) in enumerate(prob_histories.items()):
        if len(history) == 0:
            continue
        probs_over_time = np.stack([p[:, species_idx] for p in history])  # (n_iter, n_lib)
        # top-K by final iteration
        top_terms = np.argsort(probs_over_time[-1])[-top_k_terms:][::-1]
        iterations = np.arange(1, len(history) + 1)
        for t_idx, term in enumerate(top_terms):
            name = feature_names[term] if feature_names else f"term{term}"
            ax.plot(
                iterations, probs_over_time[:, term],
                color=condition_colors[c_idx],
                linestyle=linestyles[t_idx % len(linestyles)],
                alpha=0.7,
                label=f"{label}: {name}" if t_idx == 0 else None,
            )

    ax.set_xlabel("Active learning iteration")
    ax.set_ylabel("Inclusion probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_trajectory_comparison(
    X_true: np.ndarray,
    predictions: dict,
    t: np.ndarray,
    feature_names: list,
    title: str = "",
    figsize: tuple = (10, 3),
):
    """Compare true vs predicted trajectories across conditions.

    Parameters
    ----------
    X_true : array (n_t, n_species)
    predictions : dict mapping condition label -> predicted trajectory array
    t : time vector
    """
    n_species = X_true.shape[1]
    fig, axes = plt.subplots(1, n_species, figsize=figsize, sharey=True)
    if n_species == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for i, ax in enumerate(axes):
        ax.plot(t, X_true[:, i], 'k-', linewidth=2, label='true', alpha=0.8)
        for j, (label, X_pred) in enumerate(predictions.items()):
            if X_pred is None or np.any(np.isnan(X_pred)):
                continue
            ax.plot(t, X_pred[:, i], '--', color=colors[j + 1], label=label, alpha=0.8)
        ax.set_title(feature_names[i])
        ax.set_xlabel("t")
        if i == 0:
            ax.set_ylabel("concentration")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes