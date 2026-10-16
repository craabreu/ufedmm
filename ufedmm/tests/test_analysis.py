import itertools

import numpy as np
import pandas as pd
import pytest

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


def test_mean_force_fit_with_periodic_variables():
    model = ufedmm.AlanineDipeptideModel()
    sigmas = [0.5, 0.3]
    periods = [2 * np.pi, 6.0]
    s_phi = ufedmm.DynamicalVariable("s_phi", -np.pi, np.pi, 50, 300, model.phi, 1000)
    s_psi = ufedmm.DynamicalVariable("s_psi", -3, 3, 50, 300, model.psi, 1000)
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300)
    analyzer = ufedmm.FreeEnergyAnalyzer(ufed, dataframe=None)

    # Von Mises kernels with distinct periods and widths, on an asymmetric grid so
    # that the fit is well determined.
    grid = np.array(list(itertools.product([-2.0, -0.3, 1.1], [-0.6, 0.1, 0.7])))
    weights = np.random.default_rng(1).normal(size=len(grid))
    variances = np.square(sigmas)
    factors = 2 * np.pi / np.array(periods)

    def true_force(x):
        d = x - grid
        exponent = np.sum((np.cos(factors * d) - 1) / (factors**2 * variances), axis=1)
        k = weights * np.exp(exponent)
        return np.sum(k[:, None] * np.sin(factors * d) / (factors * variances), axis=0)

    forces = np.array([true_force(x) for x in grid])
    centers = [grid[:, 0], grid[:, 1]]
    mean_forces = [forces[:, 0], forces[:, 1]]
    _, mean_force = analyzer.mean_force_free_energy(centers, mean_forces, sigmas)
    fitted = mean_force(grid[:, 0], grid[:, 1])
    assert fitted.shape == (len(grid), 2)
    assert np.allclose(fitted, forces, atol=1e-6)


def test_metadynamics_bias_free_energy_can_be_built_repeatedly():
    model = ufedmm.AlanineDipeptideModel()
    sigmas = [0.4, 0.7]
    s_phi = ufedmm.DynamicalVariable(
        "s_phi", -np.pi, np.pi, 50, 300, model.phi, 1000, sigma=sigmas[0]
    )
    s_psi = ufedmm.DynamicalVariable(
        "s_psi", -np.pi, np.pi, 50, 300, model.psi, 1000, sigma=sigmas[1]
    )
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300)
    centers = np.array([[0.0, 0.5], [1.0, -1.0], [-2.5, 3.0]])
    heights = np.array([1.0, 2.0, 0.5])
    dataframe = pd.DataFrame(
        {"s_phi": centers[:, 0], "s_psi": centers[:, 1], "Height (kJ/mole)": heights}
    )
    analyzer = ufedmm.FreeEnergyAnalyzer(ufed, dataframe)

    def expected(x):
        exponents = np.sum((np.cos(x - centers) - 1) / np.square(sigmas), axis=1)
        return -np.sum(heights * np.exp(exponents))

    points = [(0.0, 0.0), (1.0, -0.8), (3.0, -3.0)]
    for _ in range(2):
        free_energy = analyzer.metadynamics_bias_free_energy()
        for point in points:
            assert free_energy(*point) == pytest.approx(expected(np.array(point)))
