"""
.. module:: output
   :platform: Unix, Windows
   :synopsis: Adiabatic Free Energy Dynamics Outputs

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _StateDataReporter: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.StateDataReporter.html

"""

import sys

from simtk import unit
from simtk.openmm import app

from ufedmm import CustomIntegrator


class MultipleFiles:
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
        >>> print('test', file=ufedmm.MultipleFiles(stdout, file))
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

    Besides, it is possible to report the current values of all collective variables, driver
    parameters, and other properties associated to a passed AFED integrator.

    Parameters
    ----------
        file : str or stream or ufedmm.MultipleFiles
            The file to write to, specified as a file name, file object, or
            :class:`~ufedmm.output.MultipleFiles` object.
        reportInterval : int
            The interval (in time steps) at which to report state data.

    Keyword Args
    ------------
        collectiveVariables : bool, defaul=False
            Whether to output the values of the collective variables included in the integrator's
            :class:`~ufedmm.ufedmm.DrivingForce`.
        driverParameters : bool, defaul=False
            Whether to output the values of the driver parameters included in the integrator's
            :class:`~ufedmm.ufedmm.DrivingForce`.
        parameterTemperatures : bool, defaul=False
            Whether to output the "instantaneous temperatures" of the driver parameters included
            in the integrator's :class:`~ufedmm.ufedmm.DrivingForce`.
        conservedEnergy : bool, default=False
            Whether to output a conserved energy (if any).

    Example
    -------
        >>> import ufedmm
        >>> from sys import stdout
        >>> from simtk import openmm, unit
        >>> model = ufedmm.AlanineDipeptideModel()
        >>> simulation = model.createDefaultSimulation()
        >>> reporter = ufedmm.StateDataReporter(
        ...     stdout,
        ...     1,
        ...     step=True,
        ...     collectiveVariables=True,
        ...     driverParameters=True,
        ...     parameterTemperatures=True,
        ... )
        >>> reporter.report(simulation, simulation.context.getState())
        #"Step","psi (radian)","phi (radian)","psi_s (radian)","phi_s (radian)","T_psi_s (K)","T_phi_s (K)"
        0,3.141592653589793,3.141592653589793,3.141592653589793,3.141592653589793,0.0,0.0

    """

    def __init__(self, file, reportInterval, **kwargs):
        self._collective_variables = kwargs.pop('collectiveVariables', False)
        self._driver_parameters = kwargs.pop('driverParameters', False)
        self._parameter_temperatures = kwargs.pop('parameterTemperatures', False)
        self._conserved_energy = kwargs.pop('conservedEnergy', False)
        super().__init__(file, reportInterval, **kwargs)
        self._backSteps = -sum([self._speed, self._elapsedTime, self._remainingTime])
        self._kB = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        self._needEnergy = self._needEnergy or self._conserved_energy

    def _add_item(self, lst, item):
        if self._backSteps == 0:
            lst.append(item)
        else:
            lst.insert(self._backSteps, item)

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        if self._ufedmm_integrator is not None:
            driving_force = self._ufedmm_integrator._driving_force
            if self._collective_variables:
                for cv in driving_force._driven_variables:
                    self._add_item(headers, f'{cv._name} ({cv._dimension})')
            if self._driver_parameters:
                for parameter in driving_force._driver_parameters:
                    self._add_item(headers, f'{parameter._name} ({parameter._dimension})')
            if self._parameter_temperatures:
                for parameter in driving_force._driver_parameters:
                    self._add_item(headers, f'T_{parameter._name} (K)')
            if self._conserved_energy:
                self._add_item(headers, f'Conserved Energy (kJ/mole)')
        return headers

    def _constructReportValues(self, simulation, state):
        values = super()._constructReportValues(simulation, state)
        if self._ufedmm_integrator is not None:
            driving_force = self._ufedmm_integrator._driving_force
            if self._collective_variables:
                for cv in driving_force.getCollectiveVariableValues(simulation.context):
                    self._add_item(values, cv)
            if self._driver_parameters:
                for parameter in driving_force._driver_parameters:
                    self._add_item(values, simulation.context.getParameter(parameter._name))
            if self._parameter_temperatures:
                for KE in self._ufedmm_integrator.getPerParameterKineticEnergy():
                    self._add_item(values, (2*KE/self._kB).value_in_unit(unit.kelvin))
            if self._conserved_energy:
                nhc = self._ufedmm_integrator
                energy = state.getPotentialEnergy() + state.getKineticEnergy()
                energy += sum(nhc.getPerParameterKineticEnergy(), nhc.getThermostatEnergy())
                self._add_item(values, energy.value_in_unit(unit.kilojoules_per_mole))
        return values

    def _initializeConstants(self, simulation):
        super()._initializeConstants(simulation)
        self._ufedmm_integrator = simulation.integrator
        if not isinstance(self._ufedmm_integrator, CustomIntegrator):
            raise Exception('Simulation integrator is not AFED-compatible')
        if self._conserved_energy and not self._ufedmm_integrator._has_conserved_energy:
            raise Exception('Simulation integrator does not have a conserved energy')
