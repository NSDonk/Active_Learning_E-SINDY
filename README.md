# Active Learning for Sparse Identification of Nonlinear Dynamics (E-SINDy-PI)

An active learning framework for data-efficient discovery of governing equations 
of nonlinear dynamical systems, combining Ensemble SINDy (E-SINDy) with a custom 
parallel implicit SINDy-PI optimizer.

**Collaborators**: Nate Odonkor, William Liu — Carnegie Mellon University, 
Computational Biology Department

## Overview

Sparse Identification of Nonlinear Dynamics (SINDy) discovers governing ODEs 
from time-series data via sparse regression over a library of candidate functions. 
This project extends SINDy in two directions:

1. **Custom SINDy-PI implementation** — a parallel implicit SINDy variant that 
   recovers rational function nonlinearities (e.g. Michaelis-Menten kinetics) 
   unattainable by standard polynomial SINDy. Inheriting from STLSQ rather than SR3,
   enabling hard thresholding and avoiding CVXPY overhead.

3. **E-SINDy-PI** — wraps the custom SINDy-PI optimizer inside PySINDy's 
   `EnsembleOptimizer` to produce bootstrap ensembles of implicit sparse models, 
   enabling inclusion probability-based uncertainty quantification over candidate 
   library terms.

4. **Active learning loop** — queries the most informative initial conditions 
   from a candidate pool using ensemble derivative variance as the uncertainty 
   metric. Within each queried trajectory, only the highest-variance timepoints 
   are retained for training, naturally filtering out uninformative steady-state 
   portions without manual truncation.

## Methods

- **Library**: PySINDy `PDELibrary` with custom candidate functions including 
  polynomial and implicit terms
- **Optimizer**: Custom `SINDyPI(STLSQ)` with joblib parallelization over 
  candidate library terms
- **Ensemble**: `EnsembleOptimizer(SINDyPI)` with bagging over trajectories
- **Query strategy**: Trajectory uncertainty — ensemble derivative variance 
  summed over timepoints and species
- **Convergence**: Inclusion probability stabilization (ensemble) or RMSE 
  plateau (single model)
- **Validation**: Symbolic reconstruction via SymPy + forward simulation with 
  `solve_ivp`
- **Target systems**: Lotka-Volterra, repressilator, Michaelis-Menten enzyme 
  kinetics, hypothalamic-pituitary-thyroid (HPT) axis

## Key Contributions

- STLSQ-based SINDy-PI that matches the original Kaheman et al. (2020) 
  formulation more closely than PySINDy's SR3-based implementation
- Trajectory-level active learning query strategy that jointly selects 
  informative initial conditions and timepoints
- Modular config-driven pipeline supporting vanilla SINDy, SINDy-PI, 
  E-SINDy, and E-SINDy-PI under a unified interface

## References

- Brunton et al. (2016). Discovering governing equations from data. *PNAS*
- Kaheman et al. (2020). SINDy-PI. *Proc. Royal Society A*
- Fasel et al. (2022). Ensemble-SINDy. *Proc. Royal Society A*
