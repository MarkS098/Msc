import numpy as np
import dvr
from matplotlib import pyplot as plt
from matplotlib import gridspec

# Physical constants
h_bar = 1
m = 1

L = 20 # box length
N_basis = 50 # number of basis set functions for approximation
states_to_plot = [1, 2, 5, 10, 15, 20]

# Grids
states = np.arange(N_basis)
x = np.linspace(-L/2,L/2,1000)

X, E, C, V, lambdas, H = dvr.run_dvr(h_bar, m, L, N_basis)

plt.figure(figsize=(15,8))
plt.title(r"Potential: $V(x) = (0.5x^2 - 0.8)e^{-0.1x^2}$")
plt.plot(lambdas, np.diag(V))
plt.xlabel('$\\lambda$')
plt.ylabel('V($\\lambda$)')
plt.grid(True)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
fig.suptitle(f"Wave function comparison for: $L = {L}$, $N = {N_basis}$")

for i, state in enumerate(states_to_plot):

    psi_d = dvr.psi_dvr(N_basis, C, L, state, x)

    axes[i].plot(x, psi_d, 'b-.', label='DVR')
    axes[i].set_title(f'$n={state}$')
    axes[i].set_xlabel('$x$')
    axes[i].set_ylabel(r'$\psi_n(x)$')
    axes[i].grid(True)

axes[0].legend()
plt.tight_layout()

plt.show()


