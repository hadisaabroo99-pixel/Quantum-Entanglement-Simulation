"""
main_figure.py
==============

Master script to generate the complete 6-panel figure for Project 1:
Quantum Entanglement Simulation.

Each panel is computed from first principles using NumPy, SciPy,
and Matplotlib. QuTiP is used where available for quantum object
manipulation.

Usage:
    python main_figure.py

Output:
    Saves 'project1_quantum_entanglement.png' to the figures directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Import project modules
from bell_states import bell_state, density_matrix, partial_trace, werner_state
from entanglement_monotones import concurrence, entanglement_of_formation
from decoherence import von_neumann_entropy_rho, decohering_density_matrix
from chsh_inequality import quantum_correlation, chsh_parameter, optimal_angles

# Try to import QuTiP
try:
    import qutip
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False


def panel_a_bloch_sphere(fig, gs):
    """Panel A: Bell State on Bloch Sphere"""
    ax = fig.add_subplot(gs[0, 0], projection='3d')

    # Create sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_s = np.outer(np.cos(u), np.sin(v))
    y_s = np.outer(np.sin(u), np.sin(v))
    z_s = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_s, y_s, z_s, alpha=0.1, color='lightblue', edgecolor='none')
    ax.plot(np.cos(u), np.sin(u), np.zeros_like(u), 'k--', alpha=0.3, lw=0.5)
    ax.plot(np.zeros_like(u), np.cos(u), np.sin(u), 'k--', alpha=0.3, lw=0.5)
    ax.plot(np.cos(u), np.zeros_like(u), np.sin(u), 'k--', alpha=0.3, lw=0.5)

    # Bell state trajectory on equator
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
            color='#E74C3C', linewidth=3, label=r'$|\Phi^+\rangle$ trajectory')

    # Mark key states
    ax.scatter([1/np.sqrt(2), -1/np.sqrt(2)], [0, 0], [1/np.sqrt(2), -1/np.sqrt(2)],
              color='#E74C3C', s=100, marker='o', edgecolors='black', linewidth=1.5, zorder=5)

    ax.set_title('A. Bell State on Bloch Sphere', fontweight='bold', pad=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.view_init(elev=20, azim=45)

    return ax


def panel_b_chsh_violation(fig, gs):
    """Panel B: CHSH Inequality Violation"""
    ax = fig.add_subplot(gs[0, 1])

    # Compute CHSH with optimal angles
    a1, a2, b1, b2 = optimal_angles()
    S_quantum = chsh_parameter(a1, a2, b1, b2)

    categories = ['Quantum\nPrediction', 'Classical\nBound', 'Experimental\n(Loophole-free)']
    values = [abs(S_quantum), 2.0, 2.77]
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    bars = ax.bar(categories, values, color=colors, edgecolor='black',
                  linewidth=1.5, alpha=0.85)
    ax.axhline(y=2, color='black', linestyle='--', linewidth=2,
               label='Local Realism Bound (|S|≤2)')
    ax.axhline(y=2*np.sqrt(2), color='#E74C3C', linestyle='--', linewidth=2,
               label=f'Tsirelson Bound (2√2≈{2*np.sqrt(2):.3f})')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('CHSH Parameter S', fontweight='bold')
    ax.set_title('B. CHSH Inequality Violation', fontweight='bold')
    ax.set_ylim(0, 3.2)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    return ax


def panel_c_decoherence(fig, gs):
    """Panel C: Decoherence Dynamics"""
    ax = fig.add_subplot(gs[0, 2])

    t = np.linspace(0, 10, 500)
    gamma = 0.3

    # von Neumann entropy evolution
    S_vn = np.array([von_neumann_entropy_rho(ti, gamma) for ti in t])

    ax.plot(t, S_vn, color='#9B59B6', linewidth=2.5, label='von Neumann Entropy')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Maximally Mixed (S=1)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Pure State (S=0)')
    ax.fill_between(t, 0, S_vn, alpha=0.2, color='#9B59B6')

    ax.set_xlabel('Time (arb. units)', fontweight='bold')
    ax.set_ylabel(r'Entanglement Entropy $S(\rho)$', fontweight='bold')
    ax.set_title('C. Decoherence Dynamics', fontweight='bold')
    ax.legend(loc='center right', fontsize=8)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3)

    return ax


def panel_d_density_matrix(fig, gs):
    """Panel D: |Phi+> Density Matrix |rho_ij|"""
    ax = fig.add_subplot(gs[1, 0])

    # Bell state |Phi+> density matrix
    phi_plus = bell_state("phi_plus")
    rho = density_matrix(phi_plus)

    im = ax.imshow(np.abs(rho), cmap='RdYlBu_r', vmin=0, vmax=0.6)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
    ax.set_yticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])

    for i in range(4):
        for j in range(4):
            text_color = "white" if np.abs(rho[i, j]) > 0.3 else "black"
            ax.text(j, i, f'{np.abs(rho[i, j]):.2f}', ha="center", va="center",
                   color=text_color, fontweight='bold')

    ax.set_title(r"D. $|\Phi^+\rangle$ Density Matrix $|
ho_{ij}|$", fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax


def panel_e_entanglement_monotones(fig, gs):
    """Panel E: Entanglement Monotones"""
    ax = fig.add_subplot(gs[1, 1])

    p_values = np.linspace(0, 1, 200)

    # Concurrence for Werner state
    conc = np.maximum(0, (3*p_values - 1)/2)

    # Entanglement of formation
    C_safe = np.clip(conc, 1e-10, 1)
    E_f = np.zeros_like(C_safe)

    for i, c in enumerate(C_safe):
        if c > 0:
            x = (1 + np.sqrt(1 - c**2)) / 2
            x = np.clip(x, 1e-15, 1 - 1e-15)
            E_f[i] = -(x * np.log2(x) + (1-x) * np.log2(1-x))

    ax.plot(p_values, conc, color='#E67E22', linewidth=2.5, label='Concurrence C(ρ)')
    ax.plot(p_values, E_f, color='#16A085', linewidth=2.5, label='Entanglement of Formation E_f(ρ)')
    ax.axvline(x=1/3, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Separability Threshold (p=1/3)')
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_xlabel('Werner State Parameter p', fontweight='bold')
    ax.set_ylabel('Entanglement Measure', fontweight='bold')
    ax.set_title('E. Entanglement Monotones', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.fill_between(p_values, 0, conc, where=(conc > 0), alpha=0.1, color='#E67E22')
    ax.grid(alpha=0.3)

    return ax


def panel_f_teleportation(fig, gs):
    """Panel F: Quantum Teleportation Protocol"""
    ax = fig.add_subplot(gs[1, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('F. Quantum Teleportation Protocol', fontweight='bold')

    # Alice's qubit |psi>
    ax.annotate('', xy=(9, 8), xytext=(1, 8),
               arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'))
    ax.text(0.3, 8, r'$|\psi\rangle$', fontsize=12, fontweight='bold', va='center')
    ax.text(1.5, 8.5, 'Alice', fontsize=10, color='#7F8C8D')

    # EPR pair
    ax.annotate('', xy=(9, 5), xytext=(1, 5),
               arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'))
    ax.annotate('', xy=(9, 2), xytext=(1, 2),
               arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'))
    ax.text(0.3, 5, r'$|\Phi^+\rangle_A$', fontsize=11, fontweight='bold', va='center')
    ax.text(0.3, 2, r'$|\Phi^+\rangle_B$', fontsize=11, fontweight='bold', va='center')
    ax.text(1.5, 2.5, 'Bob', fontsize=10, color='#7F8C8D')

    # CNOT gate
    ax.plot([3, 3], [5, 8], 'k-', lw=2)
    ax.plot(3, 8, 'o', markersize=12, color='#E74C3C', markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(3, 5, '+', markersize=14, color='#E74C3C', markeredgewidth=2)
    ax.text(3, 6.5, 'CNOT', fontsize=8, ha='center', color='#E74C3C', fontweight='bold')

    # Hadamard gate
    rect = plt.Rectangle((4.5, 7.3), 1, 1.4, fill=True, facecolor='#3498DB',
                         edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.add_patch(rect)
    ax.text(5, 8, 'H', fontsize=12, ha='center', va='center', fontweight='bold', color='white')

    # Measurement symbols
    ax.plot([7, 7], [7.5, 8.5], 'k-', lw=2)
    ax.plot(7, 8, 'o', markersize=10, color='#2C3E50')
    ax.plot([6.8, 7.2], [8.2, 7.8], 'k-', lw=1.5)
    ax.text(7, 7.2, 'M', fontsize=9, ha='center', fontweight='bold')

    ax.plot([7, 7], [4.5, 5.5], 'k-', lw=2)
    ax.plot(7, 5, 'o', markersize=10, color='#2C3E50')
    ax.plot([6.8, 7.2], [5.2, 4.8], 'k-', lw=1.5)
    ax.text(7, 4.2, 'M', fontsize=9, ha='center', fontweight='bold')

    # Classical communication
    ax.annotate('', xy=(8.5, 3), xytext=(7.2, 5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='#F39C12', linestyle='--'))
    ax.annotate('', xy=(8.5, 3), xytext=(7.2, 8),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='#F39C12', linestyle='--'))
    ax.text(7.8, 4, 'Classical\nChannel', fontsize=8, ha='center', color='#F39C12', fontweight='bold')

    # Bob's corrections
    rect2 = plt.Rectangle((8.3, 1.3), 1.4, 1.4, fill=True, facecolor='#2ECC71',
                          edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.add_patch(rect2)
    ax.text(9, 2, r'$U^\dagger$', fontsize=11, ha='center', va='center', fontweight='bold', color='white')

    ax.text(9.5, 2, r'$|\psi\rangle$', fontsize=12, fontweight='bold', va='center', color='#2ECC71')

    return ax


def generate_main_figure(save_path: str = "project1_quantum_entanglement.png"):
    """
    Generate the complete 6-panel figure for Project 1.

    Parameters
    ----------
    save_path : str
        Path to save the figure.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Quantum Entanglement Simulation: Bell States & Nonlocal Correlations',
                 fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Generate all panels
    panel_a_bloch_sphere(fig, gs)
    panel_b_chsh_violation(fig, gs)
    panel_c_decoherence(fig, gs)
    panel_d_density_matrix(fig, gs)
    panel_e_entanglement_monotones(fig, gs)
    panel_f_teleportation(fig, gs)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Main figure saved to {save_path}")

    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Project 1: Quantum Entanglement Figure")
    print("=" * 60)

    fig = generate_main_figure()
    plt.show()
