"""
SINDyPI_solve.py

Utilities for solving implicit SINDy-PI equations symbolically and simulating forward.

For each species k, SINDy-PI returns n_lib candidate models (one per library term used
as LHS). This module:
  1. Selects the best candidate model per species via derivative MSE
  2. Reconstructs the implicit equation symbolically via SymPy
  3. Solves algebraically for dx_k/dt
  4. Combines all species into a single RHS for solve_ivp
  5. Simulates forward and returns predicted trajectory

Supports arbitrary numbers of species and library terms.
"""

import re
import warnings
from typing import Optional

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from pysindy.utils import AxesArray, comprehend_axes
from joblib import Parallel, delayed
from typing import cast, Optional
import time # for diagnostics


# Symbol map construction
def build_symbol_map(feature_names: list[str], species_names: list[str]) -> dict[str, sp.Symbol]:
    """Build a sympy symbol map from feature and species names.

    Handles both state variable symbols (e.g. 'x0', 'T_mass') and their
    time derivatives (e.g. 'x0_t', 'T_mass_t') as they appear in PDE library
    feature names.

    Parameters
    ----------
    feature_names : list of str
        Feature names from model.get_feature_names()
    species_names : list of str
        State variable names e.g. ['x1', 'x2', 'x3', 'T_mass', 'P_mass']

    Returns
    -------
    symbol_map : dict mapping string token -> sympy Symbol
        Sorted by token length descending so longer tokens match first.
    """
    symbol_map = {}

    for name in species_names:
        # state variable symbol
        symbol_map[name] = sp.Symbol(name)
        # time derivative symbol — PDELibrary appends '_t'
        deriv_name = f"{name}_t"
        symbol_map[deriv_name] = sp.Symbol(deriv_name)

    # sort by length descending — critical so 'T_mass_t' matches before 'T_mass'
    symbol_map = dict(sorted(symbol_map.items(), key=lambda x: len(x[0]), reverse=True))
    return symbol_map


# Feature name -> SymPy expression
def name_to_sympy(name: str, symbol_map: dict[str, sp.Symbol]) -> sp.Expr:
    """Parse a PDE library feature name string into a SymPy expression.

    Handles products of tokens with optional integer exponents e.g.:
        'x0'           -> x0
        'x0^2'         -> x0**2
        'x0x0_t'       -> x0 * x0_t
        'T_massx0_t'   -> T_mass * x0_t
        'x0^2x0_t'     -> x0**2 * x0_t
        '1'            -> 1

    Parameters
    ----------
    name : str
        Feature name from model.get_feature_names()
    symbol_map : dict
        Output of build_symbol_map() — keys sorted longest first.

    Returns
    -------
    sympy expression
    """
    if name == '1':
        return sp.Integer(1)

    tokens = list(symbol_map.keys())  # already sorted longest first
    expr = sp.Integer(1)
    name_work = name

    while name_work:
        matched = False
        for token in tokens:
            if name_work.startswith(token):
                sym = symbol_map[token]
                rest = name_work[len(token):]
                # check for integer exponent immediately after token
                exp_match = re.match(r'^\^(\d+)', rest)
                if exp_match:
                    expr *= sym ** int(exp_match.group(1))
                    name_work = rest[exp_match.end():]
                else:
                    expr *= sym
                    name_work = rest
                matched = True
                break
        if not matched:
            raise ValueError(
                f"Could not parse token in feature name '{name}' at '{name_work}'. "
                f"Check that symbol_map covers all state variables."
            )
    return expr

# For precomputing all unique sympy solutions before the parallel loop 
# cache them in a regular dict, then passing 
# the pre-solved lambdified functions to the workers
def coef_cache_key(coefs: np.ndarray, precision: int = 4) -> tuple:
    """Hashable key for a coefficient vector."""
    return tuple(np.round(coefs, precision))

def build_feature_map(
    feature_names: list[str],
    symbol_map: dict[str, sp.Symbol],
) -> dict[str, sp.Expr]:
    """Map all feature name strings to SymPy expressions.

    Parameters
    ----------
    feature_names : list of str
    symbol_map : dict from build_symbol_map()

    Returns
    -------
    dict mapping feature_name -> sympy expression
    """
    return {name: name_to_sympy(name, symbol_map) for name in feature_names}

# Best model selection per species (derivative MSE)
def select_best_sindy_pi_model(
    model,
    X: np.ndarray,
    t: np.ndarray,
) -> dict[int, tuple[int, np.ndarray]]:
    """Select the best implicit candidate model per species via derivative MSE.

    For each species k, SINDy-PI's coefficient matrix has n_lib rows
    (one candidate model per library term used as LHS). This function
    evaluates each candidate's derivative predictions against the true
    derivatives and returns the best one per species.

    Parameters
    ----------
    model : fitted ps.SINDy model with SINDyPI optimizer
    X : array (n_timepoints, n_species)
        Training state data
    t : array (n_timepoints,)
        Timepoints

    Returns
    -------
    best : dict mapping species_idx -> (best_model_idx, best_coefs)
        best_coefs shape: (n_lib,)
    """
    # wrap X for PySINDy's AxesArray differentiation
    x_wrapped = AxesArray(X, comprehend_axes(X))

    # compute true derivatives — drop endpoints (boundary artifacts)
    X_dot_true = np.asarray(
        model.differentiation_method(x_wrapped, t=t)
    )[1:-1]                                        # (n_timepoints-2, n_species)

    # predicted derivatives from each candidate model
    X_dot_pred_all = np.asarray(model.predict(X))[1:-1]  # (n_timepoints-2, n_lib)

    n_species = X.shape[1]
    n_lib = model.coefficients().shape[0]           # n_lib (rows of coef matrix)

    best: dict[int, tuple[int, np.ndarray]] = {}

    for k in range(n_species):
        best_error = np.inf
        best_idx = None

        for j in range(n_lib):
            coefs = model.coefficients()[j, :]
            if np.all(coefs == 0):
                continue

            X_dot_pred = X_dot_pred_all[:, j]       # predictions for candidate j

            # normalized error against species k's true derivative
            true_k = X_dot_true[:, k].flatten()
            norm = np.linalg.norm(true_k)
            if norm < 1e-12:
                continue

            error = np.linalg.norm(true_k - X_dot_pred) / norm

            if error < best_error:
                best_error = error
                best_idx = j

        if best_idx is not None:
            best[k] = (best_idx, model.coefficients()[best_idx, :])

    return best


# Implicit equation solve per species

def solve_species_equation(
    best_coefs: np.ndarray,
    feature_names: list[str],
    feature_map: dict[str, sp.Expr],
    deriv_symbol: sp.Symbol,
    coef_precision: int = 4,
) -> Optional[sp.Expr]:
    """Solve one species' implicit equation for its time derivative.

    Given the coefficient vector xi_j for the best candidate model of
    species k, reconstructs:

        deriv_symbol = sum_i (coef_i * feature_i)

    and solves for deriv_symbol.

    Parameters
    ----------
    best_coefs : array (n_lib,)
    feature_names : list of str
    feature_map : dict from build_feature_map()
    deriv_symbol : sympy Symbol for dx_k/dt e.g. sp.Symbol('x1_t')
    coef_precision : decimal places to round coefficients before solve

    Returns
    -------
    solution : sympy expression for dx_k/dt, or None if solve fails
    """
    coefs_rounded = np.round(best_coefs, coef_precision)

    expr = sum(
        float(coef) * feature_map[name] # type: ignore
        for name, coef in zip(feature_names, coefs_rounded)
        if abs(coef) > 1e-10
    )

    eq = sp.Eq(deriv_symbol, expr) 
    
    try:
        start = time.time()
        solutions = sp.solve(eq, deriv_symbol)
        elapsed = time.time() - start
        if elapsed > 2.0:
            warnings.warn(f"sp.solve took {elapsed:.1f}s for {deriv_symbol} — consider simplifying library")
        if len(solutions) == 0:
            warnings.warn(f"No solution found for {deriv_symbol}")
            return None
        return solutions[0]
    except Exception as e:
        warnings.warn(f"SymPy solve failed for {deriv_symbol}: {e}")
        return None


# Full pipeline: select + solve + simulate

def simulate_sindy_pi(
    model,
    X_train: np.ndarray,
    t_train: np.ndarray,
    species_names: list[str],
    x0: np.ndarray,
    t_eval: np.ndarray,
    coef_precision: int = 4,
    ivp_method: str = 'LSODA',
) -> dict:
    """Full SINDy-PI pipeline: select best models, solve implicitly, simulate forward.

    Parameters
    ----------
    model : fitted ps.SINDy with SINDyPI optimizer
    X_train : array (n_timepoints, n_species) — training data for model selection
    t_train : array (n_timepoints,) — training timepoints
    species_names : list of str e.g. ['x1', 'x2', 'x3', 'T_mass', 'P_mass']
    x0 : array (n_species,) — initial condition for forward simulation
    t_eval : array — timepoints for forward simulation
    coef_precision : rounding precision for sympy solve
    ivp_method : solver for solve_ivp

    Returns
    -------
    dict with keys:
        'trajectory'  : array (n_t, n_species) or None if simulation failed
        'solutions'   : dict species_idx -> sympy expression for dx_k/dt
        'best_models' : dict species_idx -> (best_model_idx, best_coefs)
        'success'     : bool
    """
    feature_names = model.get_feature_names()
    symbol_map = build_symbol_map(feature_names, species_names)
    feature_map = build_feature_map(feature_names, symbol_map)

    # Step 1: select best candidate model per species
    best_models = select_best_sindy_pi_model(model, X_train, t_train)

    # Step 2: solve implicit equation per species
    solutions = {}
    rhs_functions = {}

    for k, species in enumerate(species_names):
        if k not in best_models:
            warnings.warn(f"No valid model found for species {species}, using zero RHS")
            rhs_functions[k] = lambda x, _k=k: 0.0
            continue

        _, best_coefs = best_models[k]
        deriv_symbol = symbol_map[f"{species}_t"]

        solution = solve_species_equation(
            best_coefs, feature_names, feature_map,
            deriv_symbol, coef_precision,
        )

        if solution is None:
            warnings.warn(f"Solve failed for species {species}, using zero RHS")
            rhs_functions[k] = lambda x, _k=k: 0.0
            continue

        solutions[k] = solution

        # lambdify over all state variables
        state_symbols = [symbol_map[s] for s in species_names]
        rhs_k = sp.lambdify(state_symbols, solution, 'numpy')
        rhs_functions[k] = rhs_k

    # Step 3: combine into single RHS for solve_ivp
    def rhs_combined(t, x):
        return [rhs_functions[k](*x) for k in range(len(species_names))]

    # Step 4: forward simulate
    try:
        sol = solve_ivp(
            rhs_combined,
            (t_eval[0], t_eval[-1]),
            x0,
            t_eval=t_eval,
            method=ivp_method,
            rtol=1e-8,
            atol=1e-10,
        )
        trajectory = sol.y.T if sol.success else None
        success = sol.success
        if not sol.success:
            warnings.warn(f"solve_ivp failed: {sol.message}")
    except Exception as e:
        warnings.warn(f"Forward simulation failed: {e}")
        trajectory = None
        success = False

    return {
        'trajectory': trajectory,
        'solutions': solutions,
        'best_models': best_models,
        'success': success,
    }


# Ensemble forecast for SINDy-PI (validation only)

def ensemble_forecast_sindy_pi(
    esindy_result,
    model,
    X_train: np.ndarray,
    t_train: np.ndarray,
    species_names: list[str],
    x0: np.ndarray,
    t_eval: np.ndarray,
    n_models: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    all_coefs = esindy_result.all_coefficients
    n_bootstraps = all_coefs.shape[0]

    feature_names = model.get_feature_names()
    symbol_map = build_symbol_map(feature_names, species_names)
    feature_map = build_feature_map(feature_names, symbol_map)

    # precompute true derivatives once
    x_wrapped = AxesArray(X_train, comprehend_axes(X_train))
    X_dot_true = np.asarray(
        model.differentiation_method(x_wrapped, t=t_train)
    )[1:-1]
    X_dot_pred_all = np.asarray(model.predict(X_train))[1:-1]

    n_species = len(species_names)
    n_lib = all_coefs.shape[1]
    indices = rng.choice(n_bootstraps, size=n_models, replace=True)

    # ── Phase 1: sequential — find best_j per species per member, solve sympy, cache ──
    solution_cache: dict[tuple, Optional[sp.Expr]] = {}
    member_rhs: list[Optional[dict]] = []  # rhs_functions per member

    for idx in indices:
        coef_matrix = all_coefs[idx]
        rhs_functions = {}
        valid = True
        
        print(f"  [ensemble_forecast] Phase 1: solving sympy for {n_models} members...") # PRINT STATEMENT

        for k, species in enumerate(species_names):
            
            print(f"    member {i+1}/{n_models} - cache size: {len(solution_cache)}") # PRINT STATEMENT
            
            best_error = np.inf
            best_j = None

            for j in range(n_lib):
                coefs_j = coef_matrix[j, :]
                if np.all(coefs_j == 0):
                    continue
                X_dot_pred = X_dot_pred_all[:, j]
                true_k = X_dot_true[:, k].flatten()
                norm = np.linalg.norm(true_k)
                if norm < 1e-12:
                    continue
                error = np.linalg.norm(true_k - X_dot_pred) / norm
                if error < best_error:
                    best_error = error
                    best_j = j

            if best_j is None:
                valid = False
                break

            key = (k, coef_cache_key(coef_matrix[best_j, :]))
            if key not in solution_cache:
                deriv_symbol = symbol_map[f"{species}_t"]
                solution_cache[key] = solve_species_equation(
                    coef_matrix[best_j, :], feature_names,
                    feature_map, deriv_symbol,
                )

            solution = solution_cache[key]
            if solution is None:
                valid = False
                break

            state_symbols = [symbol_map[s] for s in species_names]
            rhs_functions[k] = sp.lambdify(state_symbols, solution, 'numpy')
        
        print(f"  [ensemble_forecast] Phase 1 complete. Cache size: {len(solution_cache)}") # PRINT STATEMENT

        member_rhs.append(rhs_functions if valid else None)

    print(f"  [ensemble_forecast] Phase 2: running {len(member_rhs)} solve_ivp in parallel...") # PRINT STATEMENT
    # ── Phase 2: parallel — solve_ivp per member ──
    def _run_ivp(rhs_functions: Optional[dict]) -> Optional[np.ndarray]:
        if rhs_functions is None:
            return None
        def rhs_combined(t, x, _fns=rhs_functions):
            return [_fns[k](*x) for k in range(n_species)]
        try:
            sol = solve_ivp(
                rhs_combined,
                (t_eval[0], t_eval[-1]),
                x0, t_eval=t_eval,
                method='LSODA', rtol=1e-8, atol=1e-10,
            )
            return sol.y.T if sol.success else None
        except Exception:
            return None

    results_parallel = cast(
        list[Optional[np.ndarray]],
        Parallel(n_jobs=-1)(
            delayed(_run_ivp)(rhs) for rhs in member_rhs
        )
    )

    trajectories = [t for t in results_parallel if t is not None]

    if len(trajectories) == 0:
        n_t = len(t_eval)
        return np.full((n_t, n_species), np.nan), np.full((n_t, n_species), np.nan)

    traj_array = np.stack(trajectories)
    return traj_array.mean(axis=0), traj_array.var(axis=0)