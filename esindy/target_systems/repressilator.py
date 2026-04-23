"""
Repressilator: synthetic oscillatory network of three transcriptional
repressors (Elowitz & Leibler, Nature 2000).

    dm_i/dt = -m_i + alpha / (1 + p_j^n) + alpha_0
    dp_i/dt = -beta * (p_i - m_i)

where (i, j) cycles through (1,3), (2,1), (3,2) — each mRNA is repressed
by the protein product of the previous gene in the cycle.

State vector: [m1, m2, m3, p1, p2, p3]

Note: The repressilator dynamics include Hill-function nonlinearities,
which are *not* polynomial. This means SINDy's standard polynomial library
won't naturally capture these terms — the candidate library will need to
be augmented or the system approximated. This is part of the challenge
your project aims to investigate.
"""

import numpy as np
from .base import TargetSystem


class Repressilator(TargetSystem):

    def __init__(
        self,
        alpha: float = 216.0,
        alpha_0: float = 0.216, 
        beta: float = 101.0, # protein half life
        n: float = 2.0, # hill coefficient
    ):
        self.alpha = alpha      # maximal transcription rate
        self.alpha_0 = alpha_0  # leaky transcription rate
        self.beta = beta        # protein/mRNA lifetime ratio
        self.n = n              # Hill coefficient

    @property
    def n_species(self) -> int:
        return 6

    @property
    def species_names(self) -> list[str]:
        return ["m1", "m2", "m3", "p1", "p2", "p3"]

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        m1, m2, m3, p1, p2, p3 = x

        # Repression pairs: m1 repressed by p3, m2 by p1, m3 by p2
        hill = lambda p: self.alpha / (1.0 + p**self.n) + self.alpha_0

        dm1 = -m1 + hill(p3)
        dm2 = -m2 + hill(p1)
        dm3 = -m3 + hill(p2)
        dp1 = -self.beta * (p1 - m1)
        dp2 = -self.beta * (p2 - m2)
        dp3 = -self.beta * (p3 - m3)

        return np.array([dm1, dm2, dm3, dp1, dp2, dp3])

    def get_true_coefficients(self) -> dict:
        """For the repressilator, 'true coefficients' are the parameters
        rather than sparse polynomial terms, since the dynamics aren't
        polynomial. Evaluation will need to compare predicted trajectories
        rather than exact term recovery."""
        return {
            "params": {
                "alpha": self.alpha,
                "alpha_0": self.alpha_0,
                "beta": self.beta,
                "n": self.n,
            },
            
        }

    @staticmethod
    def default_ic_ranges() -> list[tuple[float, float]]:
        """Reasonable IC ranges. mRNA and protein levels."""
        return [
            (0.0, 10.0),  # m1
            (0.0, 10.0),  # m2
            (0.0, 10.0),  # m3
            (0.0, 10.0),  # p1
            (0.0, 10.0),  # p2
            (0.0, 10.0),  # p3
        ]

    @staticmethod
    def default_t_eval(n_points: int = 500) -> np.ndarray:
        return np.linspace(0, 50, n_points)
