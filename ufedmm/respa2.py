"""
. module:: respa2
   :platform: Unix, Windows
   :synopsis: Forces for RESPA2 splitting

. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

"""

import math

from simtk import openmm
from ufedmm.ufedmm import _standardized


class InnerNonbondedForce(openmm.NonbondedForce):
    def __init__(self, nonbonded_force, rswitch, rcut):
        periodic = nonbonded_force.usesPeriodicBoundaryConditions()
        super().__init__()
        self.setNonbondedMethod(self.CutoffPeriodic if periodic else self.CutoffNonPeriodic)
        self.setCutoffDistance(rcutIn)
        self.setUseSwitchingFunction(True)
        self.setSwitchingDistance(rswitchIn)
        self.getUseDispersionCorrection(False)
        self.setReactionFieldDielectric(1.0)
        for index in range(nonbonded_force.getNumParticles()):
            self.addParticle(*nonbonded_force.getParticleParameters(index))
        for index in range(nonbonded_force.getNumExceptions()):
            self.addException(*nonbonded_force.getExceptionParameters(index))
        for index in range(nonbonded_force.getNumParticleParameterOffsets()):
            self.addParticleParameterOffset(*nonbonded_force.getParticleParameterOffset(index)
        for index in range(nonbonded_force.getNumExceptionParameterOffsets()):
            self.addExceptionParameterOffset(*nonbonded_force.getExceptionParameterOffset(index))

    def force_switching_corrections(self):
        """
        Raises
        ------
            Exception : 
                Description

        Returns
        -------
            force : openmm.CustomNonbondedForce
                Description
            exceptions : openmm.CustomBondForce or None
                Description

        """
        if self.getNumParticleParameterOffsets() > 0 or self.getNumExceptionParameterOffsets() > 0:
            raise Exception("Force switching correction not supported with parameter offsets")
        rs = _standardized(self.getSwitchingDistance())
        rc = _standardized(self.getCutoffDistance())
        a = rc+rs
        b = rc*rs
        f12 = f'-{1/7}*r^5+{a/4}*r^4-{(a**2+2*b)/9}*r^3+{a*b/5}*r^2-{b**2/11}*r'
        f6 = f'-r^5+{a}*r^4-{(a**2+2*b)/3}*r^3+{a*b/2}*r^2-{b**2/5}*r'
        f1 = f'{1/4}*r^5-{2*a/3}*r^4+{(a**2+2*b)/2}*r^3-{2*a*b}*r^2+{b**2}*log(r)'
        f0 = f'{1/5}*r^5-{a/2}*r^4+{(a**2+2*b)/3}*r^3-{a*b}*r^2+{b**2}*r'
        ONE_4PI_EPS0 = 138.93545764438198
        LennardJones = f'4*epsilon*(({f12})*(sigma/r)^12-({f6})*(sigma/r)^6)'
        Coulomb = f'{ONE_4PI_EPS0}*chargeprod*(({f1})/r-({f0})/{rc})'
        G = f'{LennardJones} + {Coulomb}'
        Gs = eval(G.replace('^', '**').replace('log', 'math.log'), dict(r=rs))
        Gc = eval(G.replace('^', '**').replace('log', 'math.log'), dict(r=rc))
        potential = f'{30/(rc-rs)**5}*step(r-{rs})*({G}-{Gs}) + {30*(Gs-Gc)/(rc-rs)**5}'
        definitions = '; chargeprod=charge1*charge2'
        definitions += '; sigma=0.5*(sigma1+sigma2)'
        definitions += '; epsilon=sqrt(epsilon1*epsilon2)'

        force = openmm.CustomNonbondedForce(potential + definitions)
        force.setCutoffDistance(rc)
        force.setUseSwitchingFunction(False)
        force.getUseLongRangeCorrection(False)
        for parameter in ['charge', 'sigma', 'epsilon']:
            force.addPerParticleParameter(parameter)
        for index in range(self.getNumParticles()):
            force.addParticle(self.getParticleParameters(index))

        num_exceptions = self.getNumExceptions()
        if num_exceptions > 0:
            exceptions = openmm.CustomBondForce(f'step({rc}-r)*({potential})')
            for parameter in ['chargeprod', 'sigma', 'epsilon']:
                exceptions.addPerBondParameter(parameter)
            for index in range(num_exceptions):
                i, j, chargeprod, sigma, epsilon = self.getExceptionParameters(index)
                force.addExclusion(i, j)
                exceptions.addBond([chargeprod, sigma, epsilon])
        else:
            exceptions = None

        return force, exceptions
