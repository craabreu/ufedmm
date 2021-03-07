import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import ufedmm

from scipy import stats
from simtk import unit

nbins = 20
factor = 4

ufed = ufedmm.deserialize('ufed_object.yml')
df = pd.read_csv('output.csv')

print(df[['T[atoms] (K)'] + [f'T[{v.id}] (K)' for v in ufed.variables]].mean())

analyzer = ufedmm.FreeEnergyAnalyzer(ufed, df)

centers, mean_forces = analyzer.centers_and_mean_forces(nbins)

delta = 2*np.pi/nbins
potential, mean_force = analyzer.mean_force_free_energy(centers, mean_forces, sigma=factor*delta)

ranges = [(cv.min_value, cv.max_value) for cv in ufed.variables]
x = [np.linspace(*range, num=101) for range in ranges]
X = np.meshgrid(*x)
Z = potential(*X)
fe = Z-Z.min()

def in_degrees(angles):
    return [180*angle/np.pi for angle in angles]

fig, ax = plt.subplots()
cmap = plt.get_cmap('jet')
extent = in_degrees([item for sublist in ranges for item in sublist])
ax.imshow(fe, extent=extent, cmap=cmap, interpolation='spline36', origin='lower', zorder=0)
ax.contour(*in_degrees(x), fe, 20, cmap=cmap, linewidths=0.5, zorder=10)
ax.quiver(*in_degrees(centers), *mean_forces, zorder=20)
plt.savefig('figure.png')
plt.show()
