"""
Visualization utilities for E-SINDy active learning experiments.

Provides plotting functions for:
- Learning curves (error vs. measurements)
- Coefficient convergence
- Inclusion probability heatmaps
- Trajectory comparison (true vs. predicted)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from typing import Optional


def plot_learning_curves(
    results: dict[str, dict],
    title: str = "Active Learning Comparison",
    figsize: tuple = (12, 5),
) -> Figure:
    """Plot coefficient error and success rate vs. number of measurements.

    Parameters
    ----------
    results : dict mapping strategy_name -> output of evaluate_al_run()
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for name, data in results.items():
        axes[0].plot(data["n_revealed"], data["coef_error"], label=name)
        axes[1].plot(data["n_revealed"], data["success_rate"], label=name)

    axes[0].set_xlabel("# Measurements Revealed")
    axes[0].set_ylabel("Relative Coefficient Error")
    axes[0].set_title("Coefficient Recovery")
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].set_xlabel("# Measurements Revealed")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Term Identification Accuracy")
    axes[1].legend()
    axes[1].set_ylim([0, 1.05])

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_inclusion_probabilities(
    esindy_result,
    species_names: Optional[list[str]] = None,
    figsize: tuple = (8, 6),
) -> Figure:
    """Heatmap of inclusion probabilities.

    Parameters
    ----------
    esindy_result : ESINDyResult
    species_names : labels for columns
    """
    ip = esindy_result.inclusion_probabilities
    fnames = esindy_result.feature_names
    n_lib, n_species = ip.shape

    if species_names is None:
        species_names = [f"x{i}" for i in range(n_species)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(ip, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n_species))
    ax.set_xticklabels(species_names)
    ax.set_yticks(range(n_lib))
    ax.set_yticklabels(fnames, fontsize=8)
    ax.set_xlabel("Species (ẋ equation)")
    ax.set_ylabel("Library Term")
    ax.set_title("Inclusion Probabilities")

    # Annotate cells
    for i in range(n_lib):
        for j in range(n_species):
            ax.text(j, i, f"{ip[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_coefficient_convergence(
    logs: list,
    true_coefs: np.ndarray,
    feature_names: list[str],
    species_names: Optional[list[str]] = None,
    max_terms: int = 6,
    figsize: tuple = (14, 8),
) -> Figure:
    """Plot convergence of active coefficients over AL steps.

    Parameters
    ----------
    logs : list of ALStepLog
    true_coefs : (n_lib, n_species)
    feature_names : library term names
    species_names : species labels
    max_terms : max number of active terms to plot per species
    """
    n_lib, n_species = true_coefs.shape
    if species_names is None:
        species_names = [f"x{i}" for i in range(n_species)]

    # Identify active terms
    active_mask = true_coefs != 0

    fig, axes = plt.subplots(n_species, 1, figsize=figsize, sharex=True)
    if n_species == 1:
        axes = [axes]

    axes = np.asarray(axes)
    steps = [log.step for log in logs if log.coefficients is not None]

    for j in range(n_species):
        ax = axes[j]
        active_indices = np.where(active_mask[:, j])[0][:max_terms]

        for idx in active_indices:
            true_val = true_coefs[idx, j]
            learned_vals = [
                log.coefficients[idx, j]
                for log in logs
                if log.coefficients is not None
            ]
            ax.plot(steps, learned_vals, label=f"{feature_names[idx]}")
            ax.axhline(true_val, linestyle="--", alpha=0.5, color="gray")

        ax.set_ylabel(f"d({species_names[j]})/dt\nCoefficients")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("AL Step")
    fig.suptitle("Coefficient Convergence Over Active Learning")
    fig.tight_layout()
    return fig


def plot_trajectories(
    t: np.ndarray,
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    species_names: Optional[list[str]] = None,
    title: str = "Trajectory Comparison",
    figsize: tuple = (12, 6),
) -> Figure:
    """Plot true vs. predicted trajectories.

    Parameters
    ----------
    t : (n_timepoints,)
    true_traj : (n_timepoints, n_species)
    pred_traj : (n_timepoints, n_species)
    """
    n_species = true_traj.shape[1]
    if species_names is None:
        species_names = [f"x{i}" for i in range(n_species)]

    fig, axes = plt.subplots(n_species, 1, figsize=figsize, sharex=True)
    axes = np.asarray(axes)
    for j in range(n_species):
        axes[j].plot(t, true_traj[:, j], "k-", label="True", linewidth=1.5)
        axes[j].plot(t, pred_traj[:, j], "r--", label="Predicted", linewidth=1.5)
        axes[j].set_ylabel(species_names[j])
        axes[j].legend(fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
