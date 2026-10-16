import itertools

import numpy as np

import ufedmm


def test_mean_force_fit_uses_each_variables_own_sigma():
    model = ufedmm.AlanineDipeptideModel()
    sigmas = [0.3, 0.6]
    s_phi = ufedmm.DynamicalVariable(
        "s_phi", -3, 3, 50, 300, model.phi, 1000, periodic=False
    )
    s_psi = ufedmm.DynamicalVariable(
        "s_psi", -3, 3, 50, 300, model.psi, 1000, periodic=False
    )
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300)
    analyzer = ufedmm.FreeEnergyAnalyzer(ufed, dataframe=None)

    # Mean forces generated from a known Gaussian RBF expansion on the same grid
    # must be reproduced exactly by the fitted model.
    grid = np.array(list(itertools.product([-0.5, 0.0, 0.5], repeat=2)))
    weights = np.random.default_rng(0).normal(size=len(grid))
    variances = np.square(sigmas)

    def true_force(x):
        d = x - grid
        k = weights * np.exp(-0.5 * np.sum(d**2 / variances, axis=1))
        return np.sum(k[:, None] * d / variances, axis=0)

    forces = np.array([true_force(x) for x in grid])
    centers = [grid[:, 0], grid[:, 1]]
    mean_forces = [forces[:, 0], forces[:, 1]]
    _, mean_force = analyzer.mean_force_free_energy(centers, mean_forces, sigmas)
    fitted = np.array([np.asarray(mean_force(*x), dtype=float) for x in grid])
    assert np.allclose(fitted, forces, atol=1e-6)
