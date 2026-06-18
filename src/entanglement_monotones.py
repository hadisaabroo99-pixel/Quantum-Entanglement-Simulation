"""
entanglement_monotones.py
==========================

Entanglement quantification for two-qubit mixed states.

Implements concurrence, entanglement of formation, and related
monotones for Werner states and general two-qubit density matrices.

References:
-----------
[1] S. Hill & W.K. Wootters, Phys. Rev. Lett. 78, 5022 (1997)
[2] W.K. Wootters, Phys. Rev. Lett. 80, 2245 (1998)
[3] R.F. Werner, Phys. Rev. A 40, 4277 (1989)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def spin_flip(rho: np.ndarray) -> np.ndarray:
    """
    Compute the spin-flipped density matrix.

    For a two-qubit state: rho_tilde = (sigma_y ⊗ sigma_y) rho* (sigma_y ⊗ sigma_y)

    Parameters
    ----------
    rho : np.ndarray
        Two-qubit density matrix of shape (4, 4).

    Returns
    -------
    np.ndarray
        Spin-flipped density matrix.
    """
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sy_sy = np.kron(sigma_y, sigma_y)
    return sy_sy @ rho.conj() @ sy_sy


def concurrence(rho: np.ndarray) -> float:
    """
    Compute the concurrence of a two-qubit state.

    C(rho) = max{0, lambda_1 - lambda_2 - lambda_3 - lambda_4}

    where lambda_i are the square roots of eigenvalues of rho * rho_tilde
    in decreasing order.

    Parameters
    ----------
    rho : np.ndarray
        Two-qubit density matrix.

    Returns
    -------
    float
        Concurrence in [0, 1].
    """
    rho_tilde = spin_flip(rho)

    # Compute rho * rho_tilde
    R = rho @ rho_tilde

    # Eigenvalues of R
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.maximum(eigvals, 0)  # Numerical safety

    # Square roots in decreasing order
    lambdas = np.sqrt(eigvals)
    lambdas = np.sort(lambdas)[::-1]

    return float(max(0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]))


def binary_entropy(x: float) -> float:
    """
    Compute the binary entropy H(x) = -x*log2(x) - (1-x)*log2(1-x).

    Parameters
    ----------
    x : float
        Input in [0, 1].

    Returns
    -------
    float
        Binary entropy in bits.
    """
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))


def entanglement_of_formation(rho: np.ndarray) -> float:
    """
    Compute the entanglement of formation for a two-qubit state.

    E_f(rho) = H((1 + sqrt(1 - C^2)) / 2)

    where H is the binary entropy and C is the concurrence.

    Parameters
    ----------
    rho : np.ndarray
        Two-qubit density matrix.

    Returns
    -------
    float
        Entanglement of formation in [0, 1].
    """
    C = concurrence(rho)

    if C == 0:
        return 0.0

    x = (1 + np.sqrt(1 - C**2)) / 2
    return binary_entropy(x)


def werner_concurrence_analytical(p: float) -> float:
    """
    Analytical concurrence for the Werner state.

    C(rho_W) = max{0, (3p - 1) / 2}

    Parameters
    ----------
    p : float
        Werner parameter.

    Returns
    -------
    float
        Concurrence.
    """
    return max(0, (3*p - 1) / 2)


def werner_eof_analytical(p: float) -> float:
    """
    Analytical entanglement of formation for the Werner state.

    Parameters
    ----------
    p : float
        Werner parameter.

    Returns
    -------
    float
        Entanglement of formation.
    """
    C = werner_concurrence_analytical(p)

    if C == 0:
        return 0.0

    x = (1 + np.sqrt(1 - C**2)) / 2
    return binary_entropy(x)


def negativity(rho: np.ndarray, dims: Tuple[int, int] = (2, 2)) -> float:
    """
    Compute the negativity entanglement measure.

    N(rho) = (||rho^T_A||_1 - 1) / 2 = sum of absolute values of negative eigenvalues

    Parameters
    ----------
    rho : np.ndarray
        Bipartite density matrix.
    dims : Tuple[int, int]
        Dimensions of subsystems.

    Returns
    -------
    float
        Negativity.
    """
    d_A, d_B = dims

    # Partial transpose over subsystem A
    rho_tensor = rho.reshape(d_A, d_B, d_A, d_B)
    rho_pt = np.swapaxes(rho_tensor, 0, 2).reshape(d_A * d_B, d_A * d_B)

    # Sum of absolute values of negative eigenvalues
    eigvals = np.linalg.eigvalsh(rho_pt)
    return float(np.sum(np.abs(eigvals[eigvals < 0])))


def plot_entanglement_monotones(save_path: str = None):
    """
    Generate the entanglement monotones figure for Werner states.

    Plots concurrence and entanglement of formation vs. Werner parameter p.

    Parameters
    ----------
    save_path : str, optional
        Path to save the figure.
    """
    p_values = np.linspace(0, 1, 500)

    # Analytical results
    concurrence_vals = np.array([werner_concurrence_analytical(p) for p in p_values])
    eof_vals = np.array([werner_eof_analytical(p) for p in p_values])

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(p_values, concurrence_vals, color='#E67E22', linewidth=2.5,
            label='Concurrence C(ρ)')
    ax.plot(p_values, eof_vals, color='#16A085', linewidth=2.5,
            label='Entanglement of Formation E_f(ρ)')

    # Separability threshold
    ax.axvline(x=1/3, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Separability Threshold (p=1/3)')
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Fill entangled region
    ax.fill_between(p_values, 0, concurrence_vals, where=(concurrence_vals > 0),
                    alpha=0.1, color='#E67E22', label='Entangled Region')

    ax.set_xlabel('Werner State Parameter p', fontweight='bold', fontsize=12)
    ax.set_ylabel('Entanglement Measure', fontweight='bold', fontsize=12)
    ax.set_title('Entanglement Monotones for Werner States',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")

    return fig, ax


def verify_werner_monotones():
    """
    Verify numerical concurrence against analytical formula for Werner states.
    """
    from bell_states import werner_state

    print("Verifying concurrence for Werner states:")
    print("-" * 50)

    test_points = np.linspace(0, 1, 11)
    for p in test_points:
        rho = werner_state(p)
        C_num = concurrence(rho)
        C_ana = werner_concurrence_analytical(p)
        diff = abs(C_num - C_ana)
        status = "✓" if diff < 1e-10 else "✗"
        print(f"p={p:.2f}: numerical={C_num:.6f}, analytical={C_ana:.6f}, "
              f"diff={diff:.2e} {status}")


if __name__ == "__main__":
    print("=" * 60)
    print("Entanglement Monotones - Validation")
    print("=" * 60)

    verify_werner_monotones()

    print("\n" + "=" * 60)
    print("Generating entanglement monotones figure...")
    print("=" * 60)

    plot_entanglement_monotones("entanglement_monotones.png")
    plt.show()
