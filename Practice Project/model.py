import numpy as np
from scipy.special import eval_hermite, factorial


def well (x, L, n):
    """
    Calculate the wave function for a particle in a well
    :param x: float array
              spatial grid coordinates
    :param L: float
              width of the well
    :param n: int
              Energy state
    :return: ndarray
             wave function
    """
    psi_well = np.sqrt(2/L) * np.sin(n*np.pi * (x+ L/2)/L)
    return psi_well

def harmonic(x, n, h_bar, omega, m):
    """
    Calculate the quantum harmonic oscillator function
    :param x: float array
              spatial grid coordinates
    :param n: int
              energy state
    :param h_bar: float
                  reduced plancks constant
    :param omega: float
                  frequency of harmonic oscillator
    :param m: float
              mass
    :return: ndarray
             wave function
    """
    alpha = m * omega/h_bar
    norm = (alpha/np.pi)**0.25
    norm /= np.sqrt(2**n * factorial(n))

    return norm * eval_hermite(n, np.sqrt(alpha)*x) * np.exp(-alpha * x**2 / 2)