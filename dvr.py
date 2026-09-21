import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import model

def chi_basis(x, L, n):
    return model.well(x, L, n)


def phi_exact(x, n, h_bar, omega, m):
    return model.harmonic(x, n, h_bar, omega, m)

def run_dvr(h_bar, omega, m, L, N, x):
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

    # pre-allocate arrays
    V_dvr = np.zeros((N,N))
    V = np.zeros((N,N))
    X = np.zeros((N,N))
    T = np.zeros((N,N))

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

    return X, abs_error_energy, rel_error_energy,E_exact, E

