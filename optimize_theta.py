import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
import os
import dvr_complex_scale

# Same physical model and conventions as plot_complex.py
h_bar = 1
m = 1
L = 20
N_basis = 300
position_method = "analytical"

thetas_opt = np.linspace(0.1, 1, 131)
n_candidates_to_plot = 4
x = np.linspace(-L / 2, L / 2, 10000)
results = []

# Calculate the complex spectrum at every angle
all_energies_opt = []
for theta in thetas_opt:
    _, energies, _, _, _, _ = dvr_complex_scale.run_dvr(h_bar, m, L, N_basis, x, position_method, theta)
    all_energies_opt.append(energies)

all_energies_opt = np.asarray(all_energies_opt)

# Track eigenvalue branches by minimizing distance in the complex plane
n_theta, n_states = all_energies_opt.shape
tracks_opt = np.empty((n_theta, n_states), dtype=complex)
tracks_opt[0] = all_energies_opt[0]

for i in range(1, n_theta):
    cost = np.abs(tracks_opt[i - 1, :, None] - all_energies_opt[i, None, :])
    rows, cols = linear_sum_assignment(cost)
    tracks_opt[i, rows] = all_energies_opt[i, cols]

# Select candidate resonances using the same criteria as plot_complex.py
mean_E_opt = np.mean(tracks_opt, axis=0)
rms_motion_opt = np.sqrt(np.mean(np.abs(tracks_opt - mean_E_opt[None, :]) ** 2, axis=0))

mask_opt = (mean_E_opt.real > 0) & (mean_E_opt.imag < -1e-8)
candidates_opt = np.where(mask_opt)[0]
candidates_opt = candidates_opt[np.argsort(rms_motion_opt[candidates_opt])]

# Differentiate each tracked complex energy and locate its interior minimum
fig, ax = plt.subplots()
fig_trajectory, ax_trajectory = plt.subplots()

for j in candidates_opt[:n_candidates_to_plot]:
    E_track = tracks_opt[:, j]
    dE_dtheta = np.gradient(E_track, thetas_opt, edge_order=2)
    derivative_magnitude = np.abs(dE_dtheta)

    # Ignore the two boundary points; a minimum at the boundary is inconclusive
    idx_opt = np.argmin(derivative_magnitude[1:-1]) + 1
    theta_opt = thetas_opt[idx_opt]
    E_opt = E_track[idx_opt]
    results.append([theta_opt, E_opt.real, E_opt.imag])

    print(f"Candidate {j}")
    print(f"  theta_opt = {theta_opt:.4f} rad")
    print(f"  E(theta_opt) = {E_opt.real:.4f} {E_opt.imag:+.4f}j")
    print(f"  Gamma = {-2 * E_opt.imag:.4f}")
    print(f"  |dE/dtheta| = {derivative_magnitude[idx_opt]:.4e}")
    print(f"  RMS motion = {rms_motion_opt[j]:.4e}\n")

    line, = ax.plot(thetas_opt, derivative_magnitude, ".-",label=fr"$E_r\approx{E_opt.real:.3f}$")
    ax.scatter(theta_opt, derivative_magnitude[idx_opt],color= "black", marker="x", s=90, zorder=5)
    ax.set_yscale("log")
    ax_trajectory.plot(E_track.real, E_track.imag, ".-", label=fr"$E_r\approx{E_opt.real:.3f}$")
    ax_trajectory.scatter(E_opt.real, E_opt.imag, marker="x",color = "black", s=90, zorder=5)

output_path = os.path.join(os.path.dirname(__file__),"theta_optimization.xlsx")

df = pd.DataFrame(results, columns=[
    "Optimal theta (rad)",
    "Re(E)",
    "Im(E)"
])

df.to_excel(output_path, index=False)
print(f"Excel saved to: {output_path}")

ax.set_xlabel(r"Scaling angle $\theta$ (rad)")
ax.set_ylabel(r"$|dE/d\theta|$ (log scale)")
ax.set_title(fr"$\theta$ optimization ($L={L}$, $N={N_basis}$)")
ax.grid()
ax.legend()
fig.tight_layout()

ax_trajectory.axhline(0, color="black", linewidth=0.8)
ax_trajectory.axvline(0, color="black", linewidth=0.8)
ax_trajectory.set_xlabel(r"$\mathrm{Re}(E)$")
ax_trajectory.set_ylabel(r"$\mathrm{Im}(E)$")
ax_trajectory.set_title("Candidate trajectories (crosses: derivative minima)")
ax_trajectory.grid()
ax_trajectory.legend()
fig_trajectory.tight_layout()
plt.show()

