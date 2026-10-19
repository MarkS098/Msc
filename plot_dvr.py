import numpy as np
import dvr
from matplotlib import pyplot as plt

# Physical constants
h_bar = 1
m = 1
L = 20 # box length
N_basis = 201 # number of basis set functions for approximation

states_to_plot = [0, 1, 2, 5, 8, 10]

# Calculation options
position_method = "analytical"      # "analytical" or "numerical"

# Grids
states = np.arange(N_basis)
x = np.linspace(-L/2,L/2,10000)
V_grid = (0.5*x**2 - 0.8)*np.exp(-0.1*x**2)

X, E, C, V_dvr, lambdas, H = dvr.run_dvr(h_bar, m, L, N_basis, x, position_method)

# Mask to draw the ground state energy over the potential plot
E_ground = E[0]
mask = V_grid <= E_ground

fig, ax = plt.subplots(figsize=(15,8))
ax.set_title(r"Potential: $V(x) = (0.5x^2 - 0.8)e^{-0.1x^2}$")
ax.plot(x, V_grid)
ax.plot(x[mask], np.full(np.sum(mask), E_ground), color='r', linestyle='--')
ax.legend([r'$V(x)$',r'$E_0$'])
plt.xlabel('x')
plt.ylabel('V(x)')
plt.grid(True)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
fig.suptitle(f"Wave functions for $L = {L}$, $N = {N_basis}$")

for i, state in enumerate(states_to_plot):

    psi_d = np.abs(dvr.psi_dvr(N_basis, C, L, state, x))

    axes[i].plot(x, psi_d, 'b-.', label='DVR')
    axes[i].set_title(f'$n={state}$')
    axes[i].set_xlabel('$x$')
    axes[i].set_ylabel(r'$|\psi_n(x)|$')
    axes[i].grid(True)

axes[0].legend()
plt.tight_layout()

L_values = np.linspace(1,100,100)
E_ground = []

for L in L_values:

    X, E, C, V_dvr, lambdas, H = dvr.run_dvr(h_bar, m, L, N_basis, x, position_method)
    E_ground.append(np.min(E))

fig, ax = plt.subplots(figsize=(15,8))
ax.plot(L_values, E_ground, color='red')
plt.xlabel('$L$')
plt.ylabel('$E_0$')
plt.title('Ground-state energy vs box length')
plt.grid(True)
plt.tight_layout()

plt.show()

plt.show()


