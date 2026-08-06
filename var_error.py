import numpy as np
from scipy.integrate import quad
from scipy.special import eval_hermite, factorial
import matplotlib.pyplot as plt

# Natural units
h_bar = 1.0
m = 1.0
omega = 1.0

# Parameters
L = 24           # Box length
N = 60           # Number of basis states

# Particle-in-a-box basis
def chi(n, x):
    return np.sqrt(2/L)*np.sin(n*np.pi*(x + L/2)/L)

# Exact harmonic oscillator basis
def psi_exact(n, x):
    alpha = m * omega/h_bar
    norm = (alpha/np.pi)**0.25
    norm /= np.sqrt(2**n * factorial(n))

    return norm * eval_hermite(n, np.sqrt(alpha)*x) * np.exp(-alpha * x**2 / 2)


def psi_variational(state, x):
    psi = np.zeros_like(x)

    for n in range(N):
        psi += C[n, state]*chi(n+1, x)

    return psi

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
            chi(n+1, x)
            * 0.5*m*omega**2*x**2
            * chi(k+1, x)
        )

        V[n, k], _ = quad(integrand, -L/2, L/2)

# Hamiltonian and eigenvalue problem solution
H = T + V
E, C = np.linalg.eigh(H)

# Exact harmonic oscillator energies
n = np.arange(N)
E_exact = h_bar * omega * (n + 0.5)

# Errors
abs_error_energy = np.abs(E - E_exact)
rel_error_energy = abs_error_energy/E_exact

x = np.linspace(-L/2, L/2, 1000)
L2_errors = np.zeros(N)
states = np.arange(N)

for state in range(N):
    psi_e = psi_exact(state, x)
    psi_v = psi_variational(state, x)

    overlap = np.trapezoid(psi_e * psi_v, x)
    if overlap < 0:
        psi_v = -psi_v

    L2_errors[state] = np.sqrt(
                   np.trapezoid((psi_e - psi_v)**2, x)
                   )


# Print comparison
print(" Level    Approx        Exact      Abs Error   Rel Error")

for i in range(10):
    print(f"{i:3d}   {E[i]:10.6f}  {E_exact[i]:10.6f}  {abs_error_energy[i]:10.6e}  {rel_error_energy[i]:10.6e}")

# Plot energy comparison
plt.figure()
plt.title("Variational Energy vs Exact Energy")
plt.plot(n, E_exact, 'ko-', label='Exact')
plt.plot(n, E, 'rs--', label='Variational')
plt.xlabel("State n")
plt.ylabel("Energy")
plt.legend()
plt.grid()

plt.figure()
plt.title("Variational Energy Error")
plt.semilogy(n, abs_error_energy, 'o-')
plt.xlabel("State n")
plt.ylabel("Absolute Error")
plt.grid()

plt.figure()
plt.semilogy(states, L2_errors, 'o-')
plt.xlabel("state $n$")
plt.ylabel(r"$L^2$ error")
plt.title("Wavefunction $L^2$ Error")
plt.grid(True)

plt.show()