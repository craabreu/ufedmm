"""
.. module:: io
   :platform: Unix, Windows
   :synopsis: Unified Free Energy Dynamics Outputs

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _StateDataReporter: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.StateDataReporter.html

"""

import sys
import yaml
import ufedmm

from simtk import unit
from simtk.openmm import app
from ufedmm.ufedmm import _Metadynamics


class Tee:
    """
    Allows the use of multiple outputs in an OpenMM Reporter.

    Parameters
    ----------
        A list of valid outputs (file names and/or output streams).

    Example
    -------
        >>> import ufedmm
        >>> import tempfile
        >>> from sys import stdout
        >>> file = tempfile.TemporaryFile(mode='w+t')
        >>> print('test', file=ufedmm.Tee(stdout, file))
        test

    """

    def __init__(self, *files):
        self._files = list()
        for output in files:
            self._files.append(open(output, 'w') if isinstance(output, str) else output)

    def __del__(self):
        for output in self._files:
            if output != sys.stdout and output != sys.stderr:
                output.close()

    def write(self, message):
        for output in self._files:
            output.write(message)

    def flush(self):
        for output in self._files:
            output.flush()


class StateDataReporter(app.StateDataReporter):
    """
    An extension of OpenMM's StateDataReporter_ class, which outputs information about a simulation,
    such as energy, temperature, etc.

    All original functionalities of StateDataReporter_ are preserved.

    Besides, if it is added to an :class:`~ufedmm.ufedmm.ExtendedSpaceSimulation` object, e.g. one
    created through the :func:`~ufedmm.ufedmm.UnifiedFreeEnergyDynamics.simulation` method, then a
    new set of keywords are available.

    Parameters
    ----------
        file : str or stream or afed.temperature
            The file to write to, specified as a file name, file object, or
            :class:`~ufedmm.io.Tee` object.
        report_interval : int
            The interval (in time steps) at which to report state data.

    Keyword Args
    ------------
        variables : bool, default=False
            If this is `True`, then the current values of all collective variables and
            dynamical variables related to the extended-space simulation will be reported.
        multipleTemperatures : bool, default=False
            If this is `True`, then the running temperature estimates will be reported separately
            for the atoms and for the extended-space dynamical variables.
        hillHeights : bool, default=False
            If this is `True`, then the height of the latest deposited metadynamics hill will be
            reported.

    Example
    -------
        >>> import ufedmm
        >>> from simtk import openmm, unit
        >>> from sys import stdout
        >>> model = ufedmm.AlanineDipeptideModel(water='tip3p')
        >>> mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
        >>> Ks = 1000*unit.kilojoules_per_mole/unit.radians**2
        >>> T = 300*unit.kelvin
        >>> Ts = 1500*unit.kelvin
        >>> dt = 2*unit.femtoseconds
        >>> gamma = 10/unit.picoseconds
        >>> limit = 180*unit.degrees
        >>> s_phi = ufedmm.DynamicalVariable('s_phi', -limit, limit, mass, Ts, model.phi, Ks)
        >>> s_psi = ufedmm.DynamicalVariable('s_psi', -limit, limit, mass, Ts, model.psi, Ks)
        >>> ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], T)
        >>> integrator = ufedmm.GeodesicLangevinIntegrator(dt, T, gamma)
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> simulation = ufed.simulation(model.topology, model.system, integrator, platform)
        >>> simulation.context.setPositions(model.positions)
        >>> simulation.context.setVelocitiesToTemperature(300*unit.kelvin, 1234)
        >>> reporter = ufedmm.StateDataReporter(stdout, 1, variables=True)
        >>> reporter.report(simulation, simulation.context.getState())
        #"s_phi","phi","s_psi","psi"
        -3.141592653589793,3.141592653589793,-3.141592653589793,3.141592653589793

    """
    def __init__(self, file, report_interval, **kwargs):
        self._variables = kwargs.pop('variables', False)
        self._multitemps = kwargs.pop('multipleTemperatures', False)
        self._hill_heights = kwargs.pop('hillHeights', False)
        super().__init__(file, report_interval, **kwargs)
        self._backSteps = -sum([self._volume, self._density, self._speed, self._elapsedTime, self._remainingTime])
        if self._multitemps:
            self._needsVelocities = self._needEnergy = True

    def _add_item(self, lst, item):
        if self._backSteps == 0:
            lst.append(item)
        else:
            lst.insert(self._backSteps, item)

    def _initializeConstants(self, simulation):
        if self._multitemps and not self._temperature:
            self._temperature = True
            super()._initializeConstants(simulation)
            self._temperature = False
        else:
            super()._initializeConstants(simulation)

        self._extended_space = isinstance(simulation, ufedmm.ExtendedSpaceSimulation)
        if self._extended_space:
            force = simulation.context.driving_force
            self._cv_names = [force.getCollectiveVariableName(i) for i in range(force.getNumCollectiveVariables())]
            self._var_names = [v.id for v in simulation.context.variables]
            if self._hill_heights:
                metadynamics_tasks = filter(lambda x: isinstance(x, _Metadynamics), simulation._periodic_tasks)
                try:
                    self._metadynamics = next(metadynamics_tasks)
                except StopIteration:
                    raise Exception("hillHeights keyword involked for simulation w/o metadynamics bias")
                bias_factor = self._metadynamics.bias_factor
                self._height_scaling = 1 if bias_factor is None else bias_factor/(bias_factor - 1)

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        if self._extended_space:
            if self._multitemps:
                self._add_item(headers, 'T[atoms] (K)')
                for var in self._var_names:
                    self._add_item(headers, f'T[{var}] (K)')
            if self._variables:
                for cv in self._cv_names:
                    self._add_item(headers, cv)
            if self._hill_heights:
                self._add_item(headers, 'Height (kJ/mole)')
        return headers

    def _constructReportValues(self, simulation, state):
        values = super()._constructReportValues(simulation, state)
        if self._extended_space:
            if self._multitemps:
                system = simulation.context.getSystem()
                Nt = system.getNumParticles()
                Nv = len(simulation.context.variables)
                masses = [system.getParticleMass(i)/unit.dalton for i in range(Nt-Nv, Nt)]
                velocities = [v.x for v in state.getVelocities(extended=True)[Nt-Nv: Nt]]
                double_kinetic_energies = [m*v*v for m, v in zip(masses, velocities)]
                TotalKE = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
                kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoules_per_mole/unit.kelvin)
                self._add_item(values, (2*TotalKE-sum(double_kinetic_energies))/((self._dof-Nv)*kB))
                for twoKE in double_kinetic_energies:
                    self._add_item(values, twoKE/kB)
            if self._variables:
                for cv in simulation.context.driving_force.getCollectiveVariableValues(simulation.context):
                    self._add_item(values, cv)
            if self._hill_heights:
                self._add_item(values, self._metadynamics.height*self._height_scaling)
        return values


def serialize(object, file):
    """
    Serializes a ufedmm object.

    """
    dump = yaml.dump(object)
    if isinstance(file, str):
        with open(file, 'w') as f:
            f.write(dump)
    else:
        file.write(dump)


def deserialize(file):
    """
    Deserializes a ufedmm object.

    """
    if isinstance(file, str):
        with open(file, 'r') as f:
            object = yaml.load(f.read(), Loader=yaml.FullLoader)
    else:
        object = yaml.load(file.read(), Loader=yaml.FullLoader)
    return object
