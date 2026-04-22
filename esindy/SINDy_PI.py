import warnings

import numpy as np
from pysindy.optimizers.stlsq import STLSQ
from joblib import Parallel, delayed
from typing import cast 


class SINDyPI(STLSQ):
    """
    SINDy-PI optimizer for implicit SINDy problems, as described in
    "SINDy-PI: A robust algorithm for parallel implicit sparse identification of
    nonlinear dynamical systems" (https://arxiv.org/abs/2206.07787).
    
    This implementation solves each of the n models separately, which can be
    parallelized with joblib. The user can also specify a subset of the models
    to solve with the parameter model_subset.

    Parameters
    ----------
    regularizer : string, optional (default 'l1')
        Regularization function to use. Currently implemented options
        are 'l1' (l1 norm), 'weighted_l1' (weighted l1 norm), l2, and
        'weighted_l2' (weighted l2 norm)
        
    threshold : float, optional (default 0.1)
        Minimum magnitude for a coefficient in the weight vector.
        Coefficients with magnitude below the threshold are set
        to zero.

    alpha : float, optional (default 0.05)
        Optional L2 (ridge) regularization on the weight vector.

    model_subset : np.ndarray, shape(n_models), optional (default None)
        List of indices to compute models for. If list is not provided,
        the default is to compute SINDy-PI models for all possible
        candidate functions. This can take a long time for 4D systems
        or larger.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Regularized weight vector(s). This is the v in the objective
        function.

    unbias: bool
        Required to be false, maintained for supertype compatibility
    """
    def __init__(self, 
                 threshold=0.1, 
                 alpha=0.05, 
                 max_iter=20, 
                 normalize_columns=False, 
                 model_subset=None
            ):
        
        super().__init__( 
            threshold=threshold,
            alpha=alpha,
            max_iter=max_iter,
            normalize_columns=normalize_columns,
            unbias=False,
        )

        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize_columns = normalize_columns
        self.model_subset = model_subset
        
        if self.model_subset is not None:
            if not isinstance(self.model_subset, list):
                raise ValueError("Model subset must be in list format.")
            subset_integers = [
                model_ind for model_ind in self.model_subset if isinstance(model_ind, int)
            ]
            if subset_integers != self.model_subset:
                raise ValueError("Model subset list must consist of integers.")

        self.model_subset = self.model_subset

    def _solve_candidate(self, x, i):
        """Solve for xi_j given candidate index i."""
        
        # "known" target vector θ_j, shape (n_samples, 1)
        lhs = x[:, i:i+1] 
        
        # Θ(X, Ẋ|θ_j), shape (n_samples, n_features-1)
        rhs = np.delete(x, i, axis=1)
         
        # create a fresh STLSQ instance to avoid shared state
        solver = STLSQ(
            threshold=self.threshold,
            alpha=self.alpha,
            max_iter=self.max_iter,
            normalize_columns=self.normalize_columns,
            unbias=False,
        )
        solver.fit(rhs, lhs)
        return i, np.insert(solver.coef_.flatten(), i, 0.0)
    
    def _update_parallel_coef_constraints(self, x):
        """
        Solves each of the model fits separately, which can in principle be
        parallelized. Unfortunately most parallel Python packages don't give
        huge speedups. Instead, we allow the user to only solve a subset of
        the models with the parameter model_subset.
        """
        n_features = x.shape[1]
        xi_final = np.zeros((n_features, n_features))

        # Todo: parallelize this for loop with Multiprocessing/joblib
        if self.model_subset is None:
            self.model_subset = range(n_features)
        elif np.max(np.abs(self.model_subset)) >= n_features:
            raise ValueError(
                "A value in model_subset is larger than the number "
                "of features in the candidate library"
            )
            
        results = cast(list[tuple[int, np.ndarray]], Parallel(n_jobs=-1)(
            delayed(self._solve_candidate)(x, i) for i in self.model_subset
            ))

        for i, xi_j in results:
            xi_final[:, i] = xi_j
        
        return xi_final

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the SINDy-PI
        optimization problem, using CVXPY.
        """
        coef = self._update_parallel_coef_constraints(x)
        self.coef_ = coef.T
