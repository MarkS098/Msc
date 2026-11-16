
import numpy as np
from matplotlib import pyplot as plt
import dvr_complex_scale

# Physical constants
h_bar = 1
m = 1

L = 20
N_basis = 300
theta = 0.0
position_method = "analytical"

L_grid = np.linspace(10, 70, 180)
N_grid = np.linspace(10, 500, 99).astype(int)
n_levels = 10

# Calculate the spectrum for different box lengths
# Store the lowest 10 energy levels for each box length
E_L = np.zeros((len(L_grid), n_levels))
E_N = np.zeros((len(N_grid), n_levels))

for i, L_test in enumerate(L_grid):

    x_test = np.linspace(-L_test/2, L_test/2, 10000)
    X, energies, coeff, V_dvr, lambdas, H = dvr_complex_scale.run_dvr(h_bar, m, L_test, N_basis, x_test,position_method, theta)

    # At theta = 0, energies should be real
    energies = np.sort(energies.real)
    E_L[i, :] = energies[:n_levels]

for j, N_test in enumerate(N_grid):

    x_test = np.linspace(-L/2, L/2, 10000)
    X, energies, coeff, V_dvr, lambdas, H = dvr_complex_scale.run_dvr(h_bar, m, L, N_test, x_test,position_method, theta)

    # At theta = 0, energies should be real
    energies = np.sort(energies.real)
    E_N[j, :] = energies[:n_levels]

# Plot stabilization diagram
fig, ax = plt.subplots(figsize=(12, 7))

for n in range(n_levels):
    ax.plot(L_grid, E_L[:, n], linewidth=1.5,linestyle="-", marker="o", markersize=3,label=fr"$E_{{{n}}}$")

# Continuum threshold
ax.axhline(0, color="black", linestyle="--", linewidth=1)

ax.set_xlabel("Box length L")
ax.set_ylabel("Energy")
ax.set_title(fr"Stabilization plot ($\theta=0$, $N={N_basis}$)")

ax.legend()
ax.grid(True)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(12, 7))

for n in range(n_levels):
    ax.plot(N_grid,E_N[:, n],linewidth=1.5,label=fr"$E_{{{n}}}$")

# Continuum threshold
ax.axhline(0, color="black", linestyle="--", linewidth=1)

ax.set_xlabel("Basis Sets N")
ax.set_ylabel("Energy")
ax.set_title(fr"Stabilization plot ($\theta=0$, $L={L}$)")

ax.legend()
ax.grid(True)
plt.tight_layout()

plt.show()
