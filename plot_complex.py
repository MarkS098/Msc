import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

import dvr_complex_scale

# Physical constants
h_bar = 1
m = 1
L = 20 # box length
N_basis = 201 # number of basis set functions for approximation

# Calculation options
position_method = "analytical"      # "analytical" or "numerical"
thetas = np.linspace(0.1,0.5,10)                         # Scaling angle in radians

# Grids
states = np.arange(N_basis)
x = np.linspace(-L/2,L/2,10000)
N_grid = np.linspace(1,200,200)
L_grid = np.linspace(1,100,100)
all_energies = []
all_coeffs = []


for theta in thetas:

    X, energies, coeff, V_dvr, lambdas, H = dvr_complex_scale.run_dvr(h_bar, m, L, N_basis, x, position_method, theta)
    all_energies.append(energies)
    all_coeffs.append(coeff)

all_energies = np.array(all_energies)
all_coeffs = np.array(all_coeffs)

# Most likely resonance candidate calculations
n_theta, n_states = all_energies.shape

tracks = np.empty((n_theta, n_states), dtype=complex)
state_indices = np.empty((n_theta, n_states), dtype=int)

tracks[0] = all_energies[0]
state_indices[0] = np.arange(n_states)

for i in range(1, n_theta):
    previous = tracks[i - 1]
    current = all_energies[i]

    # Match eigenvalues by minimizing total distance
    cost = np.abs(previous[:, None] - current[None, :])
    rows, cols = linear_sum_assignment(cost)

    tracks[i, rows] = current[cols]
    state_indices[i, rows] = cols

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

# Plot wave functions over potential
theta_index = len(thetas) // 2
V_grid = (0.5*x**2 - 0.8)*np.exp(-0.1*x**2)
scale = 0.2 * (np.max(V_grid) - np.min(V_grid))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, V_grid, color="black", linewidth=2, label="Potential")

for resonance in candidates[:4]:

    state = state_indices[theta_index, resonance]
    coeff = all_coeffs[theta_index]

    psi = np.zeros_like(x, dtype=complex)

    for n in range(N_basis):
        psi += coeff[n, state] * dvr_complex_scale.chi_basis(x, L, n + 1)

    psi /= np.max(np.abs(psi))
    E_res = tracks[theta_index, resonance]
    ax.plot(x, E_res.real + scale*np.abs(psi),label=fr"$E={E_res.real:.4f}{E_res.imag:+.4f}i$")
    ax.axhline(E_res.real, color="grey", linestyle=":", alpha=0.5)

ax.set_xlabel("x")
ax.set_ylabel("Energy/scaled wavefunction")
ax.set_title(f"Stable resonance candidates ($L={L}$, $N={N_basis}$)")
ax.legend()
ax.grid()
plt.tight_layout()


## Convergence tests
stability_results = {}
E_reference = tracks[len(thetas) // 2, res_candidate]
E_L = []
E_N = []

# Well length stability
for L_test in L_grid:
    x_test = np.linspace(-L_test/2,L_test/2,10000)
    _, energies, _, _, _, _ = dvr_complex_scale.run_dvr(h_bar, m, L_test, N_basis, x_test,position_method, theta_ref)

    res_mask = ((energies.real > 0)&(energies.imag < -1e-8))
    resonance_candidate = energies[res_mask]

    idx = np.argmin(np.abs(resonance_candidate - E_reference))
    E_L.append(resonance_candidate[idx])

E_L = np.array(E_L)

# Calculate the optimal L
L_position_change = np.abs(np.diff(E_L.real)) / np.abs(E_L.real[:-1]) * 100
L_width_change = np.abs(np.diff(-2 * E_L.imag)) / np.abs((-2 * E_L.imag)[:-1]) * 100

L_stable = (L_position_change < 1) & (L_width_change < 1)

L_optimal = None

for i in range(len(L_stable) - 2):
    if np.all(L_stable[i:i+3]):
        L_optimal = L_grid[i]
        break

if L_optimal is None:
    L_optimal = L_grid[-1]
    print("Warning: L convergence not reached")

print("Optimal L:", L_optimal)

# Recalculate reference energy
x_optimal = np.linspace(-L_optimal/2, L_optimal/2, 10000)

_, energies_optimal, _, _, _, _ = dvr_complex_scale.run_dvr(h_bar, m, L_optimal, N_basis, x_optimal, position_method, theta_ref)

res_mask = (energies_optimal.real > 0) & (energies_optimal.imag < -1e-8)
resonance_candidate = energies_optimal[res_mask]

idx = np.argmin(np.abs(resonance_candidate - E_reference))
E_reference = resonance_candidate[idx]

# Basis set stability
for N_test in N_grid:
    N_test = int(N_test)
    x_test = np.linspace(-L_optimal/2, L_optimal/2, 10000)

    _, energies, _, _, _, _ = dvr_complex_scale.run_dvr(h_bar, m, L_optimal, N_test, x_test,position_method, theta_ref)

    res_mask = ((energies.real > 0) & (energies.imag < -1e-8))
    resonance_candidate = energies[res_mask]

    if len(resonance_candidate) == 0:
        E_N.append(np.nan + 1j * np.nan)
    else:
        idx = np.argmin(np.abs(resonance_candidate - E_reference))
        E_N.append(resonance_candidate[idx])

E_N = np.array(E_N)

# Calculate optimal N
N_position_change = np.abs(np.diff(E_N.real)) / np.abs(E_N.real[:-1]) * 100
N_width_change = np.abs(np.diff(-2 * E_N.imag)) / np.abs((-2 * E_N.imag)[:-1]) * 100

N_stable = (N_position_change < 1.0) & (N_width_change < 1.0)

N_optimal = None

for i in range(len(N_stable) - 2):
    if np.all(N_stable[i:i+3]):
        N_optimal = N_grid[i]
        break

if N_optimal is None:
    N_optimal = N_grid[-1]
    print("Warning: N convergence not reached.")

print("Optimal N:", N_optimal)


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

axes[1, 0].plot(N_grid, E_N.real, "o-")
axes[1, 0].set_xlabel("N")
axes[1, 0].set_ylabel(r"$\mathrm{Re}(E)$")
axes[1, 0].set_title("Resonance position vs N")
axes[1, 0].grid(True)

axes[1, 1].plot(N_grid, -2 * E_N.imag, "o-")
axes[1, 1].set_xlabel("N")
axes[1, 1].set_ylabel(r"$\Gamma=-2\,\mathrm{Im}(E)$")
axes[1, 1].set_title("Resonance width vs N")
axes[1, 1].grid(True)

plt.show()



