import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import lineStyles

import dvr_complex_scale

# Physical constants
h_bar = 1
m = 1
L = 20 # box length
N_basis = 20 # number of basis set functions for approximation

# Calculation options
position_method = "analytical"      # "analytical" or "numerical"
theta = 0.1                         # Scaling angle in radians

# Grids
states = np.arange(N_basis)
x = np.linspace(-L/2,L/2,10000)
V_grid = (0.5*x**2 - 0.8)*np.exp(-0.1*x**2)

X, energies, coeff, V_dvr, lambdas, H = dvr_complex_scale.run_dvr(h_bar, m, L, N_basis, x, position_method, theta)

# Sort eigenvalues by real part
order = np.argsort(energies.real)
energies = energies[order]
coeff = coeff[:, order]

for E in energies:
    if E.imag < 0:
        E_r = E.real
        Gamma = -2*E.imag
        print(f"E = {E_r:.6f} - {Gamma / 2:.6f}i")
        print(f"Resonance width Gamma = {Gamma:.6f}")

fig, ax = plt.subplots()
ax.scatter(energies.real, energies.imag)
ax.set_xlabel(r'$\mathrm{Re}(E)$')
ax.set_ylabel(r'$\mathrm{Im}(E)$')
plt.show()
