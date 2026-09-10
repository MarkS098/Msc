import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import model

h_bar = 1
omega = 1
m = 1

L = 20
N = 44

def chi_basis(x ,L, n):
    return model.well(x, L, n)

def phi_exact(x, n, h_bar, omega, m):
    return model.harmonic(x, n, h_bar, omega, m)

# pre-allocate arrays
V_dvr = np.zeros((N,N))
V = np.zeros((N,N))
X = np.zeros((N,N))
T = np.zeros((N,N))

x = np.linspace(-L/2, L/2, 10000)

# Calculate X matrix
for i in range(N):
    for j in range(N):
        integrand = lambda x: (
            chi_basis(x, L, i)
            * x *
            chi_basis(x, L, j)
        )

        X[i,j], _ = quad(integrand, -L / 2, L / 2)

# Get the lambda weights from the X matrix
lambdas, U = np.linalg.eigh(X)
Lambda = np.diag(lambdas)

# Calculate the potential matrix using the weights
V_dvr = 0.5 * m * omega**2 * Lambda**2

# Diagonalize the DVR potential matrix to get an actual representation
V = U @ V_dvr @ U.conj().T

# Calculate the kinetic energy matrix
for i in range(N):
    T[i,i] = (h_bar**2 * np.pi**2 * i**2)/(2 * m * L**2)

# Hamiltonian and eigenvalue problem solution
H = T + V
E, C = np.linalg.eigh(H)

# Calculate
n = np.arange(N)
E_exact = h_bar * omega * (n + 0.5)

L2_errors = np.zeros(N)
states = np.arange(N)

# Errors
abs_error_energy = np.abs(E - E_exact)
rel_error_energy = abs_error_energy / E_exact

for state in range(N):
    phi_e = phi_exact(x, state, h_bar, omega, m)
    chi_v = chi_basis(x, L, state)

    overlap = np.trapezoid(phi_e * chi_v, x)
    if overlap < 0:
        chi_v = -chi_v

    L2_errors[state] = np.linalg.norm(chi_v - phi_e)


gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.6)
fig = plt.figure()
fig.suptitle(f"Parameters: $L = {L}$, $N = {N}$, $\omega = {omega}$")
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2:])
ax3 = fig.add_subplot(gs[1, 1:3])

ax1.semilogy(states, L2_errors, 'o-')
ax1.set_xlabel(r'State $n$')
ax1.set_ylabel(r'$\|\psi_n^{\mathrm{DVR}}-\psi_n^{\mathrm{exact}}\|_2$')
ax1.set_title("Wavefunction $L^2$ Error")
ax1.grid(True)

ax2.set_title("DVR Energy Error")
ax2.semilogy(n, abs_error_energy, 'o-')
ax2.set_xlabel(r'State $n$')
ax2.set_ylabel(r'$|E_n^{\mathrm{DVR}}-E_n^{\mathrm{exact}}|$ [$\hbar\omega$]')
ax2.grid(True)

ax3.set_title("DVR Energy vs Exact Energy")
ax3.plot(n, E_exact, 'ko-', label='Exact')
ax3.plot(n, E, 'rs--', label='DVR')
ax3.set_xlabel(r'State $n$')
ax3.set_ylabel(r'$E_n$ [$\hbar\omega$]')
ax3.legend()
ax3.grid(True)

plt.show()

def run_dvr(N, L, omega,h_bar, m):
    """
    Run the DVR calculation
    :param N: int
              Number of DVR grid points
    :param L: float
              box length
    :param omega: float
           Harmonic oscillator frequency
    :param h_bar: float
           reduced planck constant
    :param m: float
              particle mass
    :return:
    results : dict
              Numerical results and errors
    """



