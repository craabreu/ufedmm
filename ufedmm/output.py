"""
.. module:: output
   :platform: Unix, Windows
   :synopsis: Unified Free Energy Dynamics Outputs

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _StateDataReporter: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.StateDataReporter.html

"""

import sys


class MultipleFiles:
    """
    Allows the use of multiple outputs in an OpenMM Reporter.

    Parameters
    ----------
        A list of valid outputs (file names and/or output streams).

    Example
    -------
        # >>> import ufedmm
        # >>> import tempfile
        # >>> from sys import stdout
        # >>> file = tempfile.TemporaryFile(mode='w+t')
        # >>> print('test', file=ufedmm.MultipleFiles(stdout, file))
        # test

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
