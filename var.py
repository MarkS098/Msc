import numpy as np
from scipy.integrate import quad
import model


# Particle-in-a-box basis
def chi_basis(l, n, x):
    return model.well(x, l, n)


# Exact harmonic oscillator basis
def phi_exact(n, x, omega, h_bar, m):
    return model.harmonic(x, n, h_bar, omega, m)


def psi_variational(N, C, L, state, x):
    psi = np.zeros_like(x)

    for n in range(N):
        psi += C[n, state] * chi_basis(L, n + 1, x)

    return psi

def run_var(h_bar, omega, m, L, N, x):
    """
    Run the variational approximation calculation
    :param N: int
              Number of basis set functions
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
                chi_basis(L, n+1, x)
                * 0.5*m*omega**2*x**2
                * chi_basis(L, k+1, x)
            )

            V[n, k], _ = quad(integrand, -L/2, L/2)

    # Hamiltonian and eigenvalue problem solution
    H = T + V
    E, C = np.linalg.eigh(H)

    # Exact harmonic oscillator energies
    n = np.arange(N)
    E_exact = h_bar * omega * (n + 0.5)

    L2_errors_var = np.zeros(N)

    # Errors
    abs_error_energy = np.abs(E - E_exact)
    rel_error_energy = abs_error_energy / E_exact
    ratio = abs_error_energy[::2]/abs_error_energy[1::2]

    for state in range(N):
        phi_e = phi_exact(state, x, omega, h_bar, m)
        psi_v = psi_variational(N, C, L, state, x)

        overlap = np.trapezoid(phi_e * psi_v, x)
        if overlap < 0:
            psi_v = -psi_v

        L2_errors_var[state] = np.linalg.norm(psi_v - phi_e)

    return abs_error_energy, rel_error_energy, L2_errors_var, E_exact, E
