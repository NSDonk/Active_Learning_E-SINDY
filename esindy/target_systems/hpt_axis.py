# type: ignore[import]
import biomolecular_controllers as bc
from biomolecular_controllers.simulation import SimulationRunner
from biomolecular_controllers.model_library import DEFAULT_PARAMS, DEFAULT_INITIAL_CONDITIONS
from esindy.target_systems.base import TargetSystem
import numpy as np

class HPTAxis(TargetSystem):

    def __init__(self, params=None, noise_level=0.0):
        self.runner = SimulationRunner()
        self.params = params  # None uses defaults
        self.noise_level = noise_level

    @property
    def n_species(self) -> int:
        return 5

    @property
    def species_names(self) -> list[str]:
        return ['x1', 'x2', 'x3', 'T_mass', 'P_mass']

    def rhs(self, t, x):
        # not implemented — HPT runs via Tellurium
        raise NotImplementedError("HPTAxis uses Tellurium, call simulate() directly")

    def get_true_coefficients(self) -> dict:
        # MM structure known from Antimony — document it here
        return {}
    

    def hpt_to_sindy(self, sim_result) -> tuple[np.ndarray, np.ndarray]:
        '''
        Convert HPT simulation output to format suitable for SINDy.
        '''
        # time vector
        t = sim_result['time']
        # preserve species order for consistency with config.feature_names and interpretability
        species_order = ['x1', 'x2', 'x3', 'P_mass', 'T_mass'] # trh, tsh, t4, pituitary_mass, thyroid_mass
        X = np.column_stack([sim_result['states'][s] for s in species_order])
        return X, t
    
    def simulate(self, y0, t_span, t_eval, noise_level=0.0, rng=None, **kwargs):
        # override of base class, we'll use SimulationRunner instead of solve_ivp
        
        # convert y0 array to IC dict
        ic = dict(zip(self.species_names, y0))
        
        result = self.runner.run_deterministic(
            'HPT_full',
            t_span=t_span,
            points=len(t_eval),
            params=self.params,
            ic=ic,
        )
        
        X, _ = self.hpt_to_sindy(result)
        
        if noise_level > 0:
            if rng is None:
                rng = np.random.default_rng()
            rms = np.sqrt(np.mean(X**2, axis=0))
            X = X + noise_level * rms * rng.standard_normal(X.shape)
        
        return X

    