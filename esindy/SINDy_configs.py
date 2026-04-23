from typing import Optional
from dataclasses import dataclass

import numpy as np
import pysindy as ps
from .SINDy_PI import SINDyPI



# Configuration
@dataclass
class SINDyConfig:
    """Configuration for PySINDy model construction."""

    # Library selection
    library_type: str = "polynomial"   # or "pde"
    
    # Polynomial library setting
    poly_degree: int = 2
    include_interaction: bool = True
    include_bias: bool = False  # constant term in library

    # PDE / SINDyPI library settings
    library_functions: Optional[ps.CustomLibrary] = None
    function_names: Optional[list[str]] = None
    derivative_order: int = 1
    implicit_terms: bool = False
    temporal_grid: Optional[np.ndarray] = None
    
    # Optimizer settings
    optimizer: str = "STLSQ"  # or "SINDyPI" "SR3", "SSR"
    threshold: float = 0.05   
    alpha: float = 0.05       # ridge regularization
    max_iter: int = 20
    use_ensemble: bool = False
    n_models: int = 20
    replace: bool = True
    normalize_columns: bool = True

    # Differentiation
    diff_method: str = "smoothed_finite_difference"  # or "_finite_difference"
    drop_endpoints: bool = True  # whether to drop endpoints after differentiation (can help with noise)
    
    # Input feature names for the state variables
    feature_names: Optional[list[str]] = None
    
    
    
    def build_library(self):
        if self.library_type == "polynomial":
            return ps.PolynomialLibrary(
                degree=self.poly_degree,
                include_interaction=self.include_interaction,
                include_bias=self.include_bias,
            )
        elif self.library_type == "pde":
            if self.library_functions is None:
                raise ValueError("For library_type='pde', provide library_functions and optionally function_names.")

            if self.implicit_terms and self.temporal_grid is None:
                raise ValueError("temporal_grid must be provided when implicit_terms=True.")
            
            # Ideally user passes  a feature-library object to be used directly
            if hasattr(self.library_functions, "fit") and hasattr(self.library_functions, "transform"):
                base_lib = self.library_functions
            
            #  but if they pass a list of functions instead we can wrap it in a CustomLibrary for them
            else:
                base_lib = ps.CustomLibrary(
                    library_functions=self.library_functions,
                    function_names=self.function_names,
                )
                
            return ps.PDELibrary(
                function_library=base_lib,
                derivative_order=self.derivative_order,
                include_bias=self.include_bias,
                implicit_terms=self.implicit_terms,
                temporal_grid=self.temporal_grid,
                include_interaction=self.include_interaction,
            )
        else:
            raise ValueError(f"Unknown library_type: {self.library_type}")

    def build_optimizer(self):
        if self.optimizer == "STLSQ":
            base_opt = ps.STLSQ(
                threshold=self.threshold,
                alpha=self.alpha,
                max_iter=self.max_iter,
                normalize_columns=self.normalize_columns
            )
        elif self.optimizer == "SINDyPI":
            base_opt = SINDyPI(
                threshold=self.threshold,
                alpha=self.alpha,
                max_iter=self.max_iter,
                normalize_columns=self.normalize_columns
            )
        else:
            raise ValueError(f"Optimizer '{self.optimizer}' not yet supported")

        if self.use_ensemble:
            return ps.EnsembleOptimizer(
                opt=base_opt,
                bagging=True,
                n_models=self.n_models,
                replace=self.replace,
            )
        return base_opt
    
    def build_differentiator(self):
        if self.diff_method == "finite_difference":
            return ps.FiniteDifference(drop_endpoints=self.drop_endpoints)
        elif self.diff_method == "smoothed_finite_difference":
            return ps.SmoothedFiniteDifference(drop_endpoints=self.drop_endpoints)
        else:
            raise ValueError(f"Unknown diff method: {self.diff_method}")
