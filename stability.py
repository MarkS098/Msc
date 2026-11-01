
import numpy as np
import matplotlib.pyplot as plt
import dvr_complex_scale

# Physical constants
h_bar = 1.0
m = 1.0

# Baseline parameters
L_0 = 20
N_0 = 201
theta = 0.1
position_method = "analytical"

# Set this to the approximate resonance energy from your theta scan.
# Example format only: E_target = 0.5 - 0.02j
E_target = -3.0


def calculate_energies(L, N):
    x = np.linspace(-L / 2, L / 2, 10000)

    X, energies, coeff, V_dvr, lambdas, H = (
        dvr_complex_scale.run_dvr(
            h_bar, m, L, N, x, position_method, theta
        )
    )

    return energies


def select_resonance(energies, target):
    # Candidate resonances: positive real part, negative imaginary part
    candidates = energies[
        (energies.real > 0) & (energies.imag < -1e-8)
    ]

    if len(candidates) == 0:
        return np.nan + 1j * np.nan

    # Select the candidate closest to the target complex energy
    return candidates[np.argmin(np.abs(candidates - target))]


if E_target is None:
    energies = calculate_energies(L_0, N_0)
    candidates = energies[
        (energies.real > 0) & (energies.imag < -1e-8)
    ]

    print("Candidate resonance energies at baseline:")
    for i, E in enumerate(candidates):
        print(
            f"{i}: E = {E.real:.8f} {E.imag:+.8f}j, "
            f"Gamma = {-2 * E.imag:.8f}"
        )

    raise ValueError(
        "Choose a candidate from the printed list and set E_target."
    )


# Test 1: Fix N and vary L
L_values = np.arange(12.0, 32.1, 2.0)

E_vs_L = []
previous_E = E_target

for L in L_values:
    energies = calculate_energies(L, N_0)
    E = select_resonance(energies, previous_E)

    E_vs_L.append(E)

    if np.isfinite(E.real):
        previous_E = E

E_vs_L = np.array(E_vs_L)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7))

axes[0].plot(L_values, E_vs_L.real, "o-")
axes[0].set_ylabel(r"$\mathrm{Re}(E)$")
axes[0].set_title(f"Resonance stabilization versus L (N={N_0})")
axes[0].grid(True)

axes[1].plot(L_values, -2 * E_vs_L.imag, "o-")
axes[1].set_xlabel("Box length L")
axes[1].set_ylabel(r"$\Gamma=-2\,\mathrm{Im}(E)$")
axes[1].grid(True)

plt.tight_layout()
plt.show()


# Test 2: Fix L and vary N
N_values = np.array([51, 81, 101, 151, 201, 251, 301])

E_vs_N = []

for N in N_values:
    energies = calculate_energies(L_0, N)
    E = select_resonance(energies, E_target)
    E_vs_N.append(E)

E_vs_N = np.array(E_vs_N)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7))

axes[0].plot(N_values, E_vs_N.real, "o-")
axes[0].set_ylabel(r"$\mathrm{Re}(E)$")
axes[0].set_title(f"Resonance stabilization versus N (L={L_0})")
axes[0].grid(True)

axes[1].plot(N_values, -2 * E_vs_N.imag, "o-")
axes[1].set_xlabel("Number of basis functions N")
axes[1].set_ylabel(r"$\Gamma=-2\,\mathrm{Im}(E)$")
axes[1].grid(True)

plt.tight_layout()
plt.show()
