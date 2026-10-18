import numpy as np
from scipy.integrate import quad, simpson
import model


# Particle-in-a-box basis
def chi_basis(x, L, n):
    return model.well(x, L, n)

def psi_dvr(N, C, L, state, x):
    psi = np.zeros_like(x)

    for n in range(N):
        psi += C[n, state] * chi_basis(x, L, n + 1)

    return psi

def run_dvr(h_bar, m, L, N, x):
    """
    Run the DVR calculation
    :param N: int
              Number of DVR grid points
    :param L: float
              box length
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
    # for i in range(N):
    #     for j in range(N):
    #         integrand = (
    #             chi_basis(x, L, i + 1)
    #             * x *
    #             chi_basis(x, L, j + 1)
    #         )
    #
    #         X[i,j] = simpson(integrand, x=x)

    for i in range(N):
        for j in range(N):
            n = i + 1
            k = j + 1

            if n != k and (n - k) % 2 != 0:
                X[n - 1, k - 1] = (-8*L*n*k)/(np.pi**2 * (n**2 - k**2)**2)

    # Get the lambda weights from the X matrix
    lambdas, U = np.linalg.eigh(X)

    # Calculate the potential matrix using the weights
    V_dvr = (0.5*lambdas**2 - 0.8)*np.exp(-0.1*lambdas**2)
    V_dvr = np.diag(V_dvr)

    # Diagonalize the DVR potential matrix to get an actual representation
    V = U @ V_dvr @ U.conj().T

    # Calculate the kinetic energy matrix
    for n in range(N):
        T[n,n] = (h_bar**2 * np.pi**2 * (n+1)**2)/(2 * m * L**2)

    # Hamiltonian and eigenvalue problem solution
    H = T + V
    E, C = np.linalg.eigh(H)

    return X, E, C, V_dvr, lambdas, H

