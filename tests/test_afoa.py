import numpy as np
from src.optim.afoa import AFOA, AFOAConfig


def test_afoa_sphere_maximization():
    opt = AFOA(np.array([-2.0, -2.0]), np.array([2.0, 2.0]), AFOAConfig(population_size=10, max_iterations=10, seed=1))
    result = opt.optimize(lambda x: -float(np.square(x).sum()))
    assert result["best_position"].shape == (2,)
    assert np.isfinite(result["best_fitness"])
