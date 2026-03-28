"""
Lotka-Volterra predator-prey system.

    dx/dt =  alpha * x  -  beta * x * y
    dy/dt = -delta * y  +  gamma * x * y

Default parameters from classical formulation.
"""

import numpy as np
from .base import TargetSystem


class LotkaVolterra(TargetSystem):

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        delta: float = 1.5,
        gamma: float = 0.075,
    ):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma

    @property
    def n_species(self) -> int:
        return 2

    @property
    def species_names(self) -> list[str]:
        return ["prey", "predator"]

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        prey, pred = x
        dprey = self.alpha * prey - self.beta * prey * pred
        dpred = -self.delta * pred + self.gamma * prey * pred
        return np.array([dprey, dpred])

    def get_true_coefficients(self) -> dict:
        return {
            "params": {
                "alpha": self.alpha,
                "beta": self.beta,
                "delta": self.delta,
                "gamma": self.gamma,
            },
            "active_terms": {
                "prey": [
                    ("prey", self.alpha),
                    ("prey * predator", -self.beta),
                ],
                "predator": [
                    ("predator", -self.delta),
                    ("prey * predator", self.gamma),
                ],
            },
        }

    @staticmethod
    def default_ic_ranges() -> list[tuple[float, float]]:
        """Reasonable IC ranges for pool generation."""
        return [(10.0, 50.0), (5.0, 25.0)]

    @staticmethod
    def default_t_eval(n_points: int = 200) -> np.ndarray:
        return np.linspace(0, 30, n_points)
