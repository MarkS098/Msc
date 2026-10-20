import numpy as np
from matplotlib import pyplot as plt

import dvr_complex_scale

# Physical constants
h_bar = 1
m = 1
L = 20 # box length
N_basis = 150 # number of basis set functions for approximation

# Calculation options
position_method = "analytical"      # "analytical" or "numerical"
thetas = np.linspace(0.1,0.5,10)                         # Scaling angle in radians

# Grids
states = np.arange(N_basis)
x = np.linspace(-L/2,L/2,10000)
all_energies = []


for theta in thetas:

    X, energies, coeff, V_dvr, lambdas, H = dvr_complex_scale.run_dvr(h_bar, m, L, N_basis, x, position_method, theta)
    all_energies.append(energies)

all_energies = np.array(all_energies)

# plot bound-continuum
fig, ax = plt.subplots()
colors = plt.cm.viridis(np.linspace(0, 1, len(thetas)))

for i, theta in enumerate(thetas):

    E = all_energies[i]
    ax.scatter(E.real, E.imag, color=colors[i],label=fr"$\theta = {theta:.2f}$")

ax.axhline(0, color = "black", linewidth = 0.8)
ax.axvline(0, color = "black", linewidth = 0.8)
ax.set_xlabel(r"$\mathrm{Re}(E)$")
ax.set_ylabel(r"$\mathrm{Im}(E)$")
ax.set_title(f"Complex scaled spectrum for $L = {L}$, N = {N_basis}")
ax.legend()
ax.grid()
plt.tight_layout()
plt.show()


