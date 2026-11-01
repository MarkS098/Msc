import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

import dvr_complex_scale

# Physical constants
h_bar = 1
m = 1
L = 20 # box length
N_basis = 200 # number of basis set functions for approximation

# Calculation options
position_method = "analytical"      # "analytical" or "numerical"
thetas = np.linspace(0.1,0.5,10)                         # Scaling angle in radians

# Grids
states = np.arange(N_basis)
x = np.linspace(-L/2,L/2,10000)
N_grid = np.linspace(1,200,201)
L_grid = np.linspace(1,100,100)
all_energies = []


for theta in thetas:

    X, energies, coeff, V_dvr, lambdas, H = dvr_complex_scale.run_dvr(h_bar, m, L, N_basis, x, position_method, theta)
    all_energies.append(energies)

all_energies = np.array(all_energies)

# Most likely resonance candidate calculations
n_theta, n_states = all_energies.shape

tracks = np.empty((n_theta, n_states), dtype=complex)
tracks[0] = all_energies[0]

for i in range(1, n_theta):
    previous = tracks[i - 1]
    current = all_energies[i]

    # Match eigenvalues by minimizing total distance
    cost = np.abs(previous[:, None] - current[None, :])
    rows, cols = linear_sum_assignment(cost)

    tracks[i, rows] = current[cols]

# Measure how much each tracked eigenvalue moves
mean_E = np.mean(tracks, axis=0)
rms_motion = np.sqrt(np.mean(np.abs(tracks - mean_E[None, :])**2, axis=0))

# Candidate resonances have positive real energy, negative imaginary energy
mask = (mean_E.real > 0) & (mean_E.imag < -1e-8)
candidates = np.where(mask)[0]

# Rank candidates by absolute movement across the angle scan
candidates = candidates[np.argsort(rms_motion[candidates])]

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


# Plot the trajectories of the most stable candidates
fig, ax = plt.subplots()

for j in candidates[:4]:
    ax.plot(tracks[:, j].real, tracks[:, j].imag,".-", markersize=5, linewidth=1,label=fr"$E\approx{mean_E[j].real:.3f}$")

ax.axhline(0, color="black", linewidth=1.5)
ax.axvline(0, color="black", linewidth=1.5)
ax.set_xlabel(r"$\mathrm{Re}(E)$")
ax.set_ylabel(r"$\mathrm{Im}(E)$")
ax.set_title(f"Stable candidate trajectory for $L = {L}$, $N = {N_basis}$")
ax.grid()
ax.legend(fontsize=8)
plt.tight_layout()

theta_ref = thetas[len(thetas) // 2]
res_candidate = candidates[2]

stability_results = {}
E_reference = tracks[len(thetas) // 2, res_candidate]
E_L = []
E_N = []

for L_test in L_grid:
    x_test = np.linspace(-L_test/2,L_test/2,10000)
    _, energies, _, _, _, _ = dvr_complex_scale.run_dvr(h_bar, m, L_test, N_basis, x_test,position_method, theta_ref)

    res_mask = ((energies.real > 0)&(energies.imag < -1e-8))
    resonance_candidate = energies[res_mask]

    idx = np.argmin(np.abs(resonance_candidate - E_reference))
    E_L.append(resonance_candidate[idx])

E_L = np.array(E_L)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0, 0].plot(L_grid, E_L.real, "o-")
axes[0, 0].set_xlabel("L")
axes[0, 0].set_ylabel(r"$\mathrm{Re}(E)$")
axes[0, 0].set_title("Resonance position vs L")
axes[0, 0].grid(True)

axes[0, 1].plot(L_grid, -2 * E_L.imag, "o-")
axes[0, 1].set_xlabel("L")
axes[0, 1].set_ylabel(r"$\Gamma=-2\,\mathrm{Im}(E)$")
axes[0, 1].set_title("Resonance width vs L")
axes[0, 1].grid(True)
plt.show()



