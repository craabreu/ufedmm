"""
.. module:: analysis
   :platform: Unix, Windows
   :synopsis: Unified Free Energy Dynamics with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

"""

import itertools
import numpy as np

from collections import namedtuple
from scipy import stats
from simtk import openmm
from ufedmm.ufedmm import _standardized, _get_energy_function, _get_parameters


class FreeEnergyAnalyzer(object):
    """
    Calculate free energy landscapes from UFED simulation results.

    Parameters
    ----------
        ufed : :class:`~ufedmm.ufedmm.UnifiedFreeEnergyDynamics`
            The UFED object.
        dataframe : pandas.DataFrame
            A data frame containing sampled sets of collective variables and driver parameters.

    """
    def __init__(self, ufed, dataframe):
        self._ufed = ufed
        self._dataframe = dataframe
        self._bias_variables = filter(lambda v: v.sigma is not None, self._ufed.variables)

    def metadynamics_bias_free_energy(self):
        """
        Returns a Python function which, in turn, receives the values of extended-space variables
        and returns the energy estimated from a Metadynamics bias potential reconstructed from the
        simulation data.

        Returns
        -------
            function
                The free energy function.

        """
        Variable = namedtuple('Variable', 'sigma factor periodic centers')
        variables = [
            Variable(v.sigma, 2*np.pi/v._range, v.periodic, self._dataframe[v.id].values)
            for v in self._bias_variables
        ]
        try:
            heights = self._dataframe['Height (kJ/mole)'].values
        except KeyError:
            heights = self._ufed.height

        def free_energy(*position):
            exponents = 0.0
            for v, x in zip(variables, position):
                if v.periodic:
                    exponents += (np.cos(v.factor*(v.centers - x)) - 1.0)/(v.factor*v.sigma)**2
                else:
                    exponents += -0.5*((v.centers - x)/v.sigma)**2
            return -np.sum(heights*np.exp(exponents))

        return np.vectorize(free_energy)

    def centers_and_mean_forces(self, bins, min_count=1, adjust_centers=False):
        """
        Performs binned statistics of the UFED simulation data.

        Parameters
        ----------
            bins : int or list(int)
                The number of bins in each direction.

        Keyword Args
        ------------
            min_count : int, default=1
                The miminum number of hits for a given bin to be considered in the analysis.
            adjust_centers : bool, default=False
                Whether to consider the center of a bin as the mean value of the its sampled
                internal points istead of its geometric center.

        Returns
        -------
            centers : list(numpy.array)
                The bin centers.
            mean_forces : list(numpy.array)
                The mean forces.

        """
        variables = self._ufed.variables
        sample = [self._dataframe[v.id] for v in variables]
        forces = self._compute_forces()
        ranges = [(v.min_value, v.max_value) for v in variables]

        counts = stats.binned_statistic_dd(sample, [], statistic='count', bins=bins, range=ranges)
        index = np.where(counts.statistic.flatten() >= min_count)

        n = len(variables)
        if adjust_centers:
            means = stats.binned_statistic_dd(sample, sample + forces, bins=bins, range=ranges)
            centers = [means.statistic[i].flatten()[index] for i in range(n)]
            mean_forces = [means.statistic[n+i].flatten()[index] for i in range(n)]
        else:
            means = stats.binned_statistic_dd(sample, forces, bins=bins, range=ranges)
            bin_centers = [0.5*(edges[1:] + edges[:-1]) for edges in counts.bin_edges]
            center_points = np.stack([np.array(point) for point in itertools.product(*bin_centers)])
            centers = [center_points[:, i][index] for i in range(n)]
            mean_forces = [statistic.flatten()[index] for statistic in means.statistic]
        return centers, mean_forces

    def mean_force_free_energy(self, centers, mean_forces, sigma):
        """
        Returns Python functions for evaluating the potential of mean force and their originating
        mean forces as a function of the collective variables.

        Parameters
        ----------
            centers : list(numpy.array)
                The bin centers.
            mean_forces : list(numpy.array)
                The mean forces.
            sigmas : float or unit.Quantity or list
                The standard deviation of kernels.

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
        variables = self._ufed.variables
        n = len(variables)
        try:
            variances = [_standardized(value)**2 for value in sigma]
        except TypeError:
            variances = [_standardized(sigma)**2]*n

        exponent = []
        derivative = []
        for v, variance in zip(variables, variances):
            if v.periodic:  # von Mises
                factor = 2*np.pi/v._range
                exponent.append(lambda x: (np.cos(factor*x)-1.0)/(factor*factor*variance))
                derivative.append(lambda x: -np.sin(factor*x)/(factor*variance))
            else:  # Gauss
                exponent.append(lambda x: -0.5*x**2/variance)
                derivative.append(lambda x: -x/variance)

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

    def _compute_forces(self):
        variables = self._ufed.variables
        collective_variables = [colvar.id for v in variables for colvar in v.colvars]
        extended_variables = [v.id for v in variables]
        all_variables = collective_variables + extended_variables

        force = openmm.CustomCVForce(_get_energy_function(variables))
        for key, value in _get_parameters(variables).items():
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

        n = len(self._dataframe.index)
        forces = [np.empty(n) for xv in extended_variables]
        for j, row in self._dataframe.iterrows():
            for variable in all_variables:
                context.setParameter(variable, row[variable])
            state = context.getState(getParameterDerivatives=True)
            derivatives = state.getEnergyParameterDerivatives()
            for i, xv in enumerate(extended_variables):
                forces[i][j] = -derivatives[xv]
        return forces


class Analyzer(FreeEnergyAnalyzer):
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
        super().__init__(ufed, dataframe)
        try:
            self._bins = [bin for bin in bins]
        except TypeError:
            self._bins = [bins]*len(ufed.variables)
        self._min_count = min_count
        self._adjust_centers = adjust_centers

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
        self.centers, self.mean_forces = self.centers_and_mean_forces(
            self._bins,
            self._min_count,
            self._adjust_centers,
        )
        variables = self._ufed.variables
        if sigma is None:
            sigmas = [factor*v._range/bin for v, bin in zip(variables, self._bins)]
        else:
            try:
                sigmas = [_standardized(value) for value in sigma]
            except TypeError:
                sigmas = [_standardized(sigma)]*len(variables)

        return self.mean_force_free_energy(self.centers, self.mean_forces, sigmas)
