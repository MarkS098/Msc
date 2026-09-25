import numpy as np
from scipy.integrate import quad
import model

def phi_exact(x, n, h_bar, omega, m):
    return model.harmonic(x, n, h_bar, omega, m)

def chi_basis(x, L, n):
    return model.well(x, L, n)

def psi_dvr(N, C, L, state, x):
    psi = np.zeros_like(x)

    for n in range(N):
        psi += C[n, state] * chi_basis(L, n + 1, x)

    return psi

def run_dvr(h_bar, omega, m, L, N, x):
    """
    Run the DVR calculation
    :param N: int
              Number of DVR grid points
    :param N_basis: int
              Number of basis functions
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

    # pre-allocate arrays
    V_dvr = np.zeros((N,N))
    V = np.zeros((N,N))
    X = np.zeros((N,N))
    T = np.zeros((N,N))

    # Calculate X matrix
    for i in range(N):
        for j in range(N):
            integrand = lambda x: (
                chi_basis(x, L, i + 1)
                * x *
                chi_basis(x, L, j + 1)
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
    for n in range(N):
        T[n,n] = (h_bar**2 * np.pi**2 * (n+1)**2)/(2 * m * L**2)

    # Hamiltonian and eigenvalue problem solution
    H = T + V
    E, C = np.linalg.eigh(H)

    # Calculate
    n = np.arange(N)
    E_exact = h_bar * omega * (n + 0.5)

    L2_errors_dvr = np.zeros(N)
    states = np.arange(N)

    # Errors
    abs_error_energy = np.abs(E - E_exact)
    rel_error_energy = abs_error_energy / E_exact

    for state in range(N):
        phi_e = phi_exact(x, state, h_bar, omega, m)
        psi_b = psi_dvr(N, C, L, state, x)

        overlap = np.trapezoid(phi_e * psi_b, x)
        if overlap < 0:
            psi_b = -psi_b

        L2_errors_dvr[state] = np.linalg.norm(psi_b - phi_e)

    return X, abs_error_energy, rel_error_energy, L2_errors_dvr,E_exact, E, C

