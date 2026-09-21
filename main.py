import numpy as np
import var, dvr
from matplotlib import pyplot as plt
from matplotlib import gridspec

h_bar = 1
m = 1
omega = 1

L = 20
N = 44

L2_errors_dvr = np.zeros(N)
L2_errors_var = np.zeros(N)
states = np.arange(N)
x = np.linspace(-L/2,L/2,1000)

X, abs_error_energy_dvr, rel_error_energy_dvr,E_exact, E_dvr = dvr.run_dvr(h_bar, omega, m, L, N, x)
abs_error_energy_var, rel_error_energy_var, ratio, E_exact, E_var, C = var.run_var(h_bar, omega, m, L, N, x)

for state in range(N):
    phi_e = dvr.phi_exact(x, state, h_bar, omega, m)
    chi_b = dvr.chi_basis(x, L, state)

    overlap = np.trapezoid(phi_e * chi_b, x)
    if overlap < 0:
        chi_b = -chi_b

    L2_errors_dvr[state] = np.linalg.norm(chi_b - phi_e)

for state in range(N):
    phi_e = var.phi_exact(state, x, omega, h_bar, m)
    chi_v = var.chi_variational(N, C, L, state, x)

    overlap = np.trapezoid(phi_e * chi_v, x)
    if overlap < 0:
        chi_v = -chi_v

    L2_errors_var[state] = np.linalg.norm(chi_v - phi_e)

gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.6)
fig = plt.figure()
fig.suptitle(f"DVR Parameters: $L = {L}$, $N = {N}$, $\\omega = {omega}$")
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
fig.suptitle(f"Variational Parameters: $L = {L}$, $N = {N}$, $\\omega = {omega}$")
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
plt.plot(states,E_dvr, 'o-',label='DVR')
plt.plot(states,E_var, 'rs--', label='Variational')
plt.plot(states,E_exact, 'ko-',label='Exact')
plt.xlabel(r'State $n$')
plt.ylabel(r'$E_n$ [$\hbar\omega$]')
plt.legend()

plt.show()

