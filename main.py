import numpy as np
import var, dvr
from matplotlib import pyplot as plt
from matplotlib import gridspec

# Physical constants
h_bar = 1
m = 1

omega = 1
L = 20 # box length
N_basis = 50 # number of basis set functions for approximation

states = np.arange(N_basis)
x = np.linspace(-L/2,L/2,1000)
states_to_plot = [1, 2, 5, 9]

X, abs_error_energy_dvr, rel_error_energy_dvr,L2_errors_dvr, E_exact, E_dvr, C_dvr = dvr.run_dvr(h_bar, omega, m, L, N_basis, x)
abs_error_energy_var, rel_error_energy_var, L2_errors_var, E_exact, E_var, C_var = var.run_var(h_bar, omega, m, L, N_basis, x)

gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.6)
fig = plt.figure()
fig.suptitle(f"DVR Parameters: $L = {L}$, $N = {N_basis}$, $\\omega = {omega}$")
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2:])
ax3 = fig.add_subplot(gs[1, 1:3])

ax1.semilogy(states, L2_errors_dvr, 'o-')
ax1.set_xlabel(r'State $n$')
ax1.set_ylabel(r'$\|\psi_n^{\mathrm{DVR}}-\psi_n^{\mathrm{exact}}\|_2$')
ax1.set_title("Wavefunction $L^2$ Error")
ax1.grid(True)

ax2.set_title("DVR Energy Error")
ax2.semilogy(states, abs_error_energy_dvr, 'o-')
ax2.set_xlabel(r'State $n$')
ax2.set_ylabel(r'$|E_n^{\mathrm{DVR}}-E_n^{\mathrm{exact}}|$ [$\hbar\omega$]')
ax2.grid(True)

ax3.set_title("DVR Energy vs Exact Energy")
ax3.plot(states, E_exact, 'ko-', label='Exact')
ax3.plot(states, E_dvr, 'rs--', label='DVR')
ax3.set_xlabel(r'State $n$')
ax3.set_ylabel(r'$E_n$ [$\hbar\omega$]')
ax3.legend()
ax3.grid(True)

# Plots for energy comparison and errors for varying N
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.6)
fig = plt.figure()
fig.suptitle(f"Variational Parameters: $L = {L}$, $N = {N_basis}$, $\\omega = {omega}$")
ax4 = fig.add_subplot(gs[0, :2])
ax5 = fig.add_subplot(gs[0, 2:])
ax6 = fig.add_subplot(gs[1, 1:3])

ax4.semilogy(states, L2_errors_var, 'o-')
ax4.set_xlabel(r'State $n$')
ax4.set_ylabel(r'$\|\psi_n^{\mathrm{var}}-\psi_n^{\mathrm{exact}}\|_2$')
ax4.set_title("Wavefunction $L^2$ Error")
ax4.grid(True)

ax5.set_title("Variational Energy Error")
ax5.semilogy(states, abs_error_energy_var, 'o-')
ax5.set_xlabel(r'State $n$')
ax5.set_ylabel(r'$|E_n^{\mathrm{var}}-E_n^{\mathrm{exact}}|$ [$\hbar\omega$]')
ax5.grid(True)

ax6.set_title("Variational Energy vs Exact Energy")
ax6.plot(states, E_exact, 'ko-', label='Exact')
ax6.plot(states, E_var, 'rs--', label='Variational')
ax6.set_xlabel(r'State $n$')
ax6.set_ylabel(r'$E_n$ [$\hbar\omega$]')
ax6.legend()
ax6.grid(True)

fig = plt.figure()
fig.suptitle("Variational vs DVR vs Exact")
plt.title(f"Parameters: $L = {L}$, $N = {N_basis}$, $\\omega = {omega}$")
plt.plot(states,E_dvr, 'o-',label='DVR')
plt.plot(states,E_var, 'rs--', label='Variational')
plt.plot(states,E_exact, 'ko-',label='Exact')
plt.xlabel(r'State $n$')
plt.ylabel(r'$E_n$ [$\hbar\omega$]')
plt.legend()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
fig.suptitle(f"Wave function comparison for: $L = {L}$, $N = {N_basis}$, $\\omega = {omega}$")

for i, state in enumerate(states_to_plot):

    phi_e = var.phi_exact(state, x, omega, h_bar, m)
    psi_v = var.psi_variational(N_basis, C_var, L, state, x)
    psi_d = dvr.psi_dvr(N_basis, C_dvr, L, state, x)

    # Align signs
    if np.trapezoid(phi_e * psi_v, x) < 0:
        psi_v = -psi_v

    if np.trapezoid(phi_e * psi_d, x) < 0:
        psi_d = -psi_d

    axes[i].plot(x, psi_d, 'b-.', label='DVR')
    axes[i].plot(x, phi_e, 'k-', label='Exact')
    axes[i].plot(x, psi_v, 'r--', label='Variational')

    axes[i].set_title(f'$n={state}$')
    axes[i].set_xlabel('$x$')
    axes[i].set_ylabel(r'$\psi_n(x)$')
    axes[i].grid(True)

axes[0].legend()
plt.tight_layout()

plt.show()

