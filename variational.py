import numpy as np
from matplotlib.lines import lineStyles
from scipy.integrate import quad
from scipy.special import eval_hermite, factorial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import model

# Natural units
h_bar = 1.0
m = 1.0
omega = 1

# Parameters
L = 20           # Box length
N = 44         # Number of basis states

# Particle-in-a-box basis
def chi(l,n, x):
    return model.well(x, l, n)

# Exact harmonic oscillator basis
def phi_exact(n, x, omega, h_bar, m):
    return model.harmonic(x, n, h_bar, omega, m)

def chi_variational(l, state, x):
    psi = np.zeros_like(x)

    for n in range(N):
        psi += C[n, state]*chi(l, n+1, x)

    return psi

## N dependence ##
# Kinetic matrix construction
T = np.zeros((N, N))

for n in range(N):
    energy = h_bar**2 * np.pi**2 * (n+1)**2 / (2*m*L**2)
    T[n, n] = energy

# Potential matrix construction
V = np.zeros((N, N))

for n in range(N):
    for k in range(N):

        integrand = lambda x: (
            chi(L, n+1, x)
            * 0.5*m*omega**2*x**2
            * chi(L, k+1, x)
        )

        V[n, k], _ = quad(integrand, -L/2, L/2)

# Hamiltonian and eigenvalue problem solution
H = T + V
E, C = np.linalg.eigh(H)

x = np.linspace(-L/2, L/2, 10000)
L2_errors = np.zeros(N)
states = np.arange(N)

for state in range(N):
    phi_e = phi_exact(state, x, omega, h_bar, m)
    chi_v = chi_variational(L, state, x)

    overlap = np.trapezoid(phi_e * chi_v, x)
    if overlap < 0:
        chi_v = -chi_v

    L2_errors[state] = np.linalg.norm(chi_v - phi_e)

# Exact harmonic oscillator energies
n = np.arange(N)
E_exact = h_bar * omega * (n + 0.5)

# Errors
abs_error_energy = np.abs(E - E_exact)
rel_error_energy = abs_error_energy / E_exact
ratio = abs_error_energy[::2]/abs_error_energy[1::2]

# Print comparison
print(" Level    Approx        Exact      Abs Error   Rel Error")

for i in range(10):
    print(f"{i:3d}   {E[i]:10.6f}  {E_exact[i]:10.6f}  {abs_error_energy[i]:10.6e}  {rel_error_energy[i]:10.6e}")

for i in range(20, 40, 2):
    print(
        f"Even n={i}: error = {abs_error_energy[i]:.3e}"
    )
    print(
        f"Odd  n={i+1}: error = {abs_error_energy[i+1]:.3e}"
    )
    print(
        f"Ratio = {abs_error_energy[i]/abs_error_energy[i+1]:.3e}"
    )


# Plots for energy comparison and errors for varying N
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.6)
fig = plt.figure()
fig.suptitle(f"Parameters: $L = {L}$, $N = {N}$, $\omega = {omega}$")
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2:])
ax3 = fig.add_subplot(gs[1, 1:3])

ax1.semilogy(states, L2_errors, 'o-')
ax1.set_xlabel(r'State $n$')
ax1.set_ylabel(r'$\|\psi_n^{\mathrm{var}}-\psi_n^{\mathrm{exact}}\|_2$')
ax1.set_title("Wavefunction $L^2$ Error")
ax1.grid(True)

ax2.set_title("Variational Energy Error")
ax2.semilogy(n, abs_error_energy, 'o-')
ax2.set_xlabel(r'State $n$')
ax2.set_ylabel(r'$|E_n^{\mathrm{var}}-E_n^{\mathrm{exact}}|$ [$\hbar\omega$]')
ax2.grid(True)

ax3.set_title("Variational Energy vs Exact Energy")
ax3.plot(n, E_exact, 'ko-', label='Exact')
ax3.plot(n, E, 'rs--', label='Variational')
ax3.set_xlabel(r'State $n$')
ax3.set_ylabel(r'$E_n$ [$\hbar\omega$]')
ax3.legend()
ax3.grid(True)

plt.figure()
plt.plot(ratio,linestyle = 'none',marker = 'o')
plt.title("Pair-wise ratio of steps")
plt.show()
