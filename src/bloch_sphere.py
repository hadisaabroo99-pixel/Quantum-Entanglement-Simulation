"""
bloch_sphere.py
===============

Bloch sphere visualization for quantum states.

Uses QuTiP for high-quality Bloch sphere rendering and matplotlib
for custom 3D visualizations of state trajectories.

References:
-----------
[1] J.R. Johansson, P.D. Nation, F. Nori, Comp. Phys. Comm. 183, 1760 (2012)
[2] F. Bloch, Phys. Rev. 70, 460 (1946)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple

# Try to import QuTiP for Bloch sphere
try:
    import qutip
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    print("Warning: QuTiP not available. Using matplotlib fallback.")


def state_to_bloch_vector(state: np.ndarray) -> np.ndarray:
    """
    Convert a single-qubit state to Bloch vector coordinates.

    For pure state |psi> = cos(theta/2)|0> + e^(i*phi)*sin(theta/2)|1>,
    the Bloch vector is (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)).

    Parameters
    ----------
    state : np.ndarray
        Single-qubit state vector of shape (2, 1) or (2,).

    Returns
    -------
    np.ndarray
        Bloch vector (x, y, z) of length 1 for pure states.
    """
    psi = state.flatten()

    # Density matrix
    rho = np.outer(psi, psi.conj())

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    x = np.real(np.trace(rho @ sigma_x))
    y = np.real(np.trace(rho @ sigma_y))
    z = np.real(np.trace(rho @ sigma_z))

    return np.array([x, y, z])


def bell_state_bloch_trajectory() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the effective Bloch sphere trajectory for a Bell state.

    For |Phi+>, the correlated spins trace a trajectory on the equator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays of x, y, z coordinates.
    """
    theta = np.linspace(0, 2*np.pi, 100)

    # Bell state |Phi+> = (|00> + |11>)/sqrt(2) has correlated spins
    # Effective trajectory on equator
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.zeros_like(theta)

    return x, y, z


def plot_bloch_sphere_matplotlib(save_path: str = None):
    """
    Generate a 3D Bloch sphere visualization using matplotlib.

    Parameters
    ----------
    save_path : str, optional
        Path to save the figure.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Sphere surface (wireframe)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1,
                    color='lightblue', edgecolor='none')

    # Equator and meridians
    ax.plot(np.cos(u), np.sin(u), np.zeros_like(u), 'k--',
            alpha=0.3, lw=0.5)
    ax.plot(np.zeros_like(u), np.cos(u), np.sin(u), 'k--',
            alpha=0.3, lw=0.5)
    ax.plot(np.cos(u), np.zeros_like(u), np.sin(u), 'k--',
            alpha=0.3, lw=0.5)

    # Bell state trajectory (equator)
    x_traj, y_traj, z_traj = bell_state_bloch_trajectory()
    ax.plot(x_traj, y_traj, z_traj, color='#E74C3C', linewidth=3,
            label=r'$|\Phi^+\rangle$ trajectory')

    # Mark key states
    states = {
        '|0⟩': (0, 0, 1),
        '|1⟩': (0, 0, -1),
        '|+⟩': (1, 0, 0),
        '|-⟩': (-1, 0, 0),
    }

    for label, (x, y, z) in states.items():
        ax.scatter([x], [y], [z], s=100, color='#2C3E50',
                   edgecolors='black', linewidth=1.5, zorder=5)
        offset = 0.15
        ax.text(x + offset, y + offset, z + offset, label,
               fontsize=11, fontweight='bold')

    ax.set_xlabel('X', fontweight='bold', fontsize=11)
    ax.set_ylabel('Y', fontweight='bold', fontsize=11)
    ax.set_zlabel('Z', fontweight='bold', fontsize=11)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_title('Bell State on Bloch Sphere', fontweight='bold',
                 fontsize=14, pad=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")

    return fig, ax


def plot_bloch_sphere_qutip(state: Optional[np.ndarray] = None,
                             save_path: str = None):
    """
    Generate a Bloch sphere visualization using QuTiP.

    Parameters
    ----------
    state : np.ndarray, optional
        Single-qubit state vector. If None, uses |0>.
    save_path : str, optional
        Path to save the figure.
    """
    if not QUTIP_AVAILABLE:
        print("QuTiP not available. Using matplotlib fallback.")
        return plot_bloch_sphere_matplotlib(save_path)

    b = qutip.Bloch()

    if state is not None:
        psi = qutip.Qobj(state)
        b.add_states(psi)
    else:
        b.add_states(qutip.basis(2, 0))

    # Add some vector states for reference
    b.add_vectors([1, 0, 0])   # |+>
    b.add_vectors([0, 0, -1])  # |1>

    b.figsize = (8, 8)
    b.render()

    if save_path:
        b.save(save_path)
        print(f"Figure saved to {save_path}")

    return b


def plot_gate_trajectory(gate_sequence: list,
                          save_path: str = None):
    """
    Plot the trajectory of a state under a sequence of gates.

    Parameters
    ----------
    gate_sequence : list
        List of (gate_name, params) tuples.
    save_path : str, optional
        Path to save the figure.
    """
    from scipy.linalg import expm

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Start from |0>
    psi = np.array([1, 0], dtype=complex)
    trajectory = [state_to_bloch_vector(psi)]

    for gate_name, params in gate_sequence:
        if gate_name == 'Rx':
            theta = params['theta']
            U = expm(-1j * theta * sigma_x / 2)
        elif gate_name == 'Ry':
            theta = params['theta']
            U = expm(-1j * theta * sigma_y / 2)
        elif gate_name == 'Rz':
            phi = params['phi']
            U = expm(-1j * phi * sigma_z / 2)
        else:
            continue

        psi = U @ psi
        trajectory.append(state_to_bloch_vector(psi))

    trajectory = np.array(trajectory)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_s = np.outer(np.cos(u), np.sin(v))
    y_s = np.outer(np.sin(u), np.sin(v))
    z_s = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_s, y_s, z_s, alpha=0.1, color='lightblue')

    # Trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            'o-', color='#E74C3C', markersize=4, linewidth=2,
            label='State trajectory')

    # Start and end points
    ax.scatter(*trajectory[0], s=150, color='green', marker='o',
               edgecolors='black', linewidth=2, label='|0⟩', zorder=5)
    ax.scatter(*trajectory[-1], s=150, color='blue', marker='s',
               edgecolors='black', linewidth=2, label='Final', zorder=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_title('Single-Qubit Gate Evolution', fontweight='bold')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")

    return fig, ax


if __name__ == "__main__":
    print("=" * 60)
    print("Bloch Sphere Visualization")
    print("=" * 60)

    # Test state to Bloch vector
    psi_0 = np.array([1, 0])
    psi_plus = np.array([1, 1]) / np.sqrt(2)
    psi_1 = np.array([0, 1])

    print("\nBloch vectors:")
    for name, psi in [("|0⟩", psi_0), ("|+⟩", psi_plus), ("|1⟩", psi_1)]:
        bv = state_to_bloch_vector(psi)
        print(f"  {name}: ({bv[0]:.3f}, {bv[1]:.3f}, {bv[2]:.3f})")

    print("\nGenerating Bloch sphere figure...")
    plot_bloch_sphere_matplotlib(save_path="bloch_sphere.png")

    print("\nGenerating gate trajectory...")
    gates = [
        ('Rx', {'theta': np.pi/2}),
        ('Rz', {'phi': np.pi/4}),
        ('Rx', {'theta': np.pi/4}),
    ]
    plot_gate_trajectory(gates, save_path="gate_trajectory.png")

    plt.show()
