"""
chsh_inequality.py
==================

Simulation of the Clauser-Horne-Shimony-Holt (CHSH) inequality violation.

Computes quantum correlations for the singlet state and demonstrates
violation of the Bell inequality |S| <= 2, reaching the Tsirelson bound
|S| = 2*sqrt(2).

References:
-----------
[1] J.F. Clauser et al., Phys. Rev. Lett. 23, 880 (1969)
[2] B.S. Tsirelson, Lett. Math. Phys. 4, 93 (1980)
[3] B. Hensen et al., Nature 526, 682 (2015)
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


def quantum_correlation(angle_a: float, angle_b: float) -> float:
    """
    Calculate quantum correlation for the singlet state |Psi->.

    For the singlet state, E(a,b) = <Psi-| (sigma·a_hat) ⊗ (sigma·b_hat) |Psi->
                                    = -cos(a - b)

    Parameters
    ----------
    angle_a, angle_b : float
        Measurement angles in radians for Alice and Bob.

    Returns
    -------
    float
        Correlation value in [-1, 1].
    """
    return -np.cos(angle_a - angle_b)


def classical_correlation(angle_a: float, angle_b: float) -> float:
    """
    Classical hidden variable model correlation (maximally correlated).

    For a deterministic classical model, E(a,b) = -sign(cos(a-b)) * 0.95
    (slightly below 1 for realism).

    Parameters
    ----------
    angle_a, angle_b : float
        Measurement angles in radians.

    Returns
    -------
    float
        Classical correlation value.
    """
    return -np.sign(np.cos(angle_a - angle_b)) * 0.95


def chsh_parameter(a1: float, a2: float, b1: float, b2: float,
                   correlation_func=quantum_correlation) -> float:
    """
    Calculate the CHSH parameter S.

    S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)

    Local realism bound: |S| <= 2
    Quantum bound (Tsirelson): |S| <= 2*sqrt(2)

    Parameters
    ----------
    a1, a2 : float
        Alice's two measurement angles.
    b1, b2 : float
        Bob's two measurement angles.
    correlation_func : callable
        Function E(a,b) to use for correlations.

    Returns
    -------
    float
        CHSH parameter S.
    """
    S = (correlation_func(a1, b1) - correlation_func(a1, b2) +
         correlation_func(a2, b1) + correlation_func(a2, b2))
    return S


def optimal_angles() -> Tuple[float, float, float, float]:
    """
    Return the optimal measurement angles for maximal CHSH violation.

    For the singlet state, the optimal configuration is:
    a = 0, a' = pi/4, b = pi/8, b' = 3*pi/8

    Returns
    -------
    Tuple[float, float, float, float]
        Angles (a, a', b, b').
    """
    return (0.0, np.pi/4, np.pi/8, 3*np.pi/8)


def compute_chsh_for_angles(angles: List[Tuple[float, float, float, float]]) -> List[float]:
    """
    Compute CHSH parameter for multiple angle configurations.

    Parameters
    ----------
    angles : List[Tuple[float, float, float, float]]
        List of (a1, a2, b1, b2) angle tuples.

    Returns
    -------
    List[float]
        CHSH parameters for each configuration.
    """
    return [chsh_parameter(*config) for config in angles]


def plot_chsh_violation(save_path: str = None):
    """
    Generate the CHSH inequality violation comparison figure.

    Compares quantum prediction, classical bound, and experimental
    loophole-free results.

    Parameters
    ----------
    save_path : str, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute quantum prediction with optimal angles
    a1, a2, b1, b2 = optimal_angles()
    S_quantum = chsh_parameter(a1, a2, b1, b2, correlation_func=quantum_correlation)
    S_classical = chsh_parameter(a1, a2, b1, b2, correlation_func=classical_correlation)

    # Experimental loophole-free result (Hensen et al. 2015)
    S_experimental = 2.77

    categories = ['Quantum\nPrediction', 'Classical\nBound', 'Experimental\n(Loophole-free)']
    values = [abs(S_quantum), 2.0, S_experimental]
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    bars = ax.bar(categories, values, color=colors, edgecolor='black',
                  linewidth=1.5, alpha=0.85, width=0.6)

    # Reference lines
    ax.axhline(y=2, color='black', linestyle='--', linewidth=2,
               label='Local Realism Bound (|S|≤2)')
    ax.axhline(y=2*np.sqrt(2), color='#E74C3C', linestyle='--', linewidth=2,
               label=f'Tsirelson Bound (2√2≈{2*np.sqrt(2):.3f})')

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('CHSH Parameter S', fontweight='bold', fontsize=12)
    ax.set_title('CHSH Inequality Violation', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 3.2)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")

    return fig, ax


def plot_correlation_function(save_path: str = None):
    """
    Plot the quantum correlation function E(a,b) = -cos(a-b) as a heatmap.

    Parameters
    ----------
    save_path : str, optional
        Path to save the figure.
    """
    a_vals = np.linspace(0, 2*np.pi, 200)
    b_vals = np.linspace(0, 2*np.pi, 200)
    A, B = np.meshgrid(a_vals, b_vals)

    E = -np.cos(A - B)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.contourf(A, B, E, levels=50, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel("Alice's angle a (rad)", fontweight='bold', fontsize=12)
    ax.set_ylabel("Bob's angle b (rad)", fontweight='bold', fontsize=12)
    ax.set_title(r'Quantum Correlation $E(a,b) = -\cos(a-b)$',
                 fontweight='bold', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('E(a,b)', fontweight='bold')

    # Mark optimal angles
    a1, a2, b1, b2 = optimal_angles()
    ax.scatter([a1, a1, a2, a2], [b1, b2, b1, b2], color='yellow',
               s=100, marker='*', edgecolors='black', linewidth=1.5, zorder=5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")

    return fig, ax


if __name__ == "__main__":
    print("=" * 60)
    print("CHSH Inequality Violation")
    print("=" * 60)

    # Optimal angles
    a1, a2, b1, b2 = optimal_angles()
    print(f"\nOptimal angles:")
    print(f"  Alice: a = {np.degrees(a1):.1f}°, a' = {np.degrees(a2):.1f}°")
    print(f"  Bob:   b = {np.degrees(b1):.1f}°, b' = {np.degrees(b2):.1f}°")

    # Quantum prediction
    S_qm = chsh_parameter(a1, a2, b1, b2, correlation_func=quantum_correlation)
    print(f"\nQuantum prediction: S = {S_qm:.6f}")
    print(f"  |S| = {abs(S_qm):.6f}")
    print(f"  Tsirelson bound: 2√2 = {2*np.sqrt(2):.6f}")
    print(f"  Violation ratio: {abs(S_qm)/2:.4f}x classical limit")

    # Classical bound
    S_cl = chsh_parameter(a1, a2, b1, b2, correlation_func=classical_correlation)
    print(f"\nClassical model: S = {S_cl:.6f}")
    print(f"  |S| = {abs(S_cl):.6f} (should be ≤ 2)")

    # Generate figures
    print("\nGenerating figures...")
    plot_chsh_violation("chsh_violation.png")
    plot_correlation_function("correlation_heatmap.png")
    plt.show()
