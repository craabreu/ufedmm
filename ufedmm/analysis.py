"""
.. module:: analysis
   :platform: Unix, Windows
   :synopsis: Unified Free Energy Dynamics with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

"""

import itertools
import numpy as np

from scipy import stats
from simtk import openmm
from ufedmm.ufedmm import _standardized, _get_energy_function, _get_parameters


class Analyzer(object):
    """
    UFED Analyzer.

    Parameters
    ----------
        ufed : :class:`~ufedmm.ufedmm.UnifiedFreeEnergyDynamics`
            The UFED object.
        dataframe : pandas.DataFrame
            A data frame containing sampled sets of collective variables and driver parameters.
        bins : int or list(int)
            The number of bins in each direction.

    Keyword Args
    ------------
        min_count : int, default=1
            The miminum number of hits for a given bin to be considered in the analysis.
        adjust_centers : bool, default=False
            Whether to consider the center of a bin as the mean value of the its sampled
            internal points istead of its geometric center.

    """
    def __init__(self, ufed, dataframe, bins, min_count=1, adjust_centers=False):
        self._ufed = ufed
        try:
            self._bins = [bin for bin in bins]
        except TypeError:
            self._bins = [bins]*len(ufed.variables)

        sample = [dataframe[v.id] for v in ufed.variables]
        forces = self._compute_forces(ufed, dataframe)
        ranges = [(v.min_value, v.max_value) for v in ufed.variables]

        counts = stats.binned_statistic_dd(sample, [], statistic='count', bins=self._bins, range=ranges)
        index = np.where(counts.statistic.flatten() >= min_count)

        n = len(ufed.variables)
        if adjust_centers:
            means = stats.binned_statistic_dd(sample, sample + forces, bins=self._bins, range=ranges)
            self.centers = [means.statistic[i].flatten()[index] for i in range(n)]
            self.mean_forces = [means.statistic[n+i].flatten()[index] for i in range(n)]
        else:
            means = stats.binned_statistic_dd(sample, forces, bins=self._bins, range=ranges)
            bin_centers = [0.5*(edges[1:] + edges[:-1]) for edges in counts.bin_edges]
            center_points = np.stack([np.array(point) for point in itertools.product(*bin_centers)])
            self.centers = [center_points[:, i][index] for i in range(n)]
            self.mean_forces = [statistic.flatten()[index] for statistic in means.statistic]

    def _compute_forces(self, ufed, dataframe):
        collective_variables = [colvar.id for v in ufed.variables for colvar in v.colvars]
        extended_variables = [v.id for v in ufed.variables]
        all_variables = collective_variables + extended_variables

        force = openmm.CustomCVForce(_get_energy_function(ufed.variables))
        for key, value in _get_parameters(ufed.variables).items():
            force.addGlobalParameter(key, value)
        for variable in all_variables:
            force.addGlobalParameter(variable, 0)
        for xv in extended_variables:
            force.addEnergyParameterDerivative(xv)

        system = openmm.System()
        system.addForce(force)
        system.addParticle(0)
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(system, openmm.CustomIntegrator(0), platform)
        context.setPositions([openmm.Vec3(0, 0, 0)])

        n = len(dataframe.index)
        forces = [np.empty(n) for xv in extended_variables]
        for j, row in dataframe.iterrows():
            for variable in all_variables:
                context.setParameter(variable, row[variable])
            state = context.getState(getParameterDerivatives=True)
            derivatives = state.getEnergyParameterDerivatives()
            for i, xv in enumerate(extended_variables):
                forces[i][j] = -derivatives[xv]
        return forces

    def free_energy_functions(self, sigma=None, factor=8):
        """
        Returns Python functions for evaluating the potential of mean force and their originating
        mean forces as a function of the collective variables.

        Keyword Args
        ------------
            sigma : float or unit.Quantity, default=None
                The standard deviation of kernels. If this is `None`, then values will be
                determined from the distances between nodes.
            factor : float, default=8
                If ``sigma`` is not explicitly provided, then it will be computed as
                ``sigma = factor*range/bins`` for each direction.

        Returns
        -------
            potential : function
                A Python function whose arguments are collective variable values and whose result
                is the potential of mean force at that values.
            mean_force : function
                A Python function whose arguments are collective variable values and whose result
                is the mean force at that values regarding a given direction. Such direction must
                be defined through a keyword argument `dir`, whose default value is `0` (meaning
                the direction of the first collective variable).

        """
        if sigma is None:
            variances = [(factor*v._range/self._bins[i])**2 for i, v in enumerate(self._ufed.variables)]
        else:
            try:
                variances = [_standardized(value)**2 for value in sigma]
            except TypeError:
                variances = [_standardized(sigma)**2]*len(self._ufed.variables)

        exponent = []
        derivative = []
        for v, variance in zip(self._ufed.variables, variances):
            if v.periodic:  # von Mises
                factor = 2*np.pi/v._range
                exponent.append(lambda x: (np.cos(factor*x)-1.0)/(factor*factor*variance))
                derivative.append(lambda x: -np.sin(factor*x)/(factor*variance))
            else:  # Gauss
                exponent.append(lambda x: -0.5*x**2/variance)
                derivative.append(lambda x: -x/variance)

        n = len(self._ufed.variables)

        def kernel(x):
            return np.exp(np.sum(exponent[i](x[i]) for i in range(n)))

        def gradient(x, i):
            return kernel(x)*derivative[i](x[i])

        centers = [np.array(xc) for xc in zip(*self.centers)]
        coefficients = []
        for i in range(n):
            for x in centers:
                coefficients.append(np.array([gradient(x-xc, i) for xc in centers]))
        M = np.vstack(coefficients)
        F = -np.hstack(self.mean_forces)
        A, _, _, _ = np.linalg.lstsq(M, F, rcond=None)

        kernels = np.empty((len(centers), len(centers)))
        for i, x in enumerate(centers):
            kernels[i, :] = np.array([kernel(x-xc) for xc in centers])
        potentials = kernels.dot(A)
        minimum = potentials.min()

        def potential(*x):
            xa = np.array(x)
            kernels = np.array([kernel(xa-xc) for xc in centers])
            return np.sum(A*kernels) - minimum

        def mean_force(*x, dir=0):
            xa = np.array(x)
            gradients = np.array([gradient(xa-xc, dir) for xc in centers])
            return -np.sum(A*gradients)

        return np.vectorize(potential), np.vectorize(mean_force)
