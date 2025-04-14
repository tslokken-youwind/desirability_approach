import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from desirability_functions import (
    one_sided_desirability,
    negative_one_sided_desirability,
    two_sided_desirability,
    segmented_one_sided_desirability,
    segmented_negative_one_sided_desirability
)

# Create figures directory if it doesn't exist
pl.Path('../figures').mkdir(parents=True, exist_ok=True)

def plot_one_sided_profiles():
    """Plot examples of one-sided desirability functions with different shape parameters."""
    y = np.linspace(0, 10, 1000)
    y_min, y_max = 2, 8
    
    plt.figure(figsize=(10, 6))
    
    for r in [0.5, 1, 2, 4]:
        d = [one_sided_desirability(val, y_min, y_max, r) for val in y]
        plt.plot(y, d, label=f'r = {r}')
    
    plt.axvline(x=y_min, color='gray', linestyle='--')
    plt.axvline(x=y_max, color='gray', linestyle='--')
    plt.xlabel('Response Value')
    plt.ylabel('Desirability')
    plt.title('One-Sided Desirability Functions (Larger is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/one_sided_profiles.pdf')
    plt.close()

def plot_negative_one_sided_profiles():
    """Plot examples of negative one-sided desirability functions with different shape parameters."""
    y = np.linspace(0, 10, 1000)
    y_min, y_max = 2, 8
    
    plt.figure(figsize=(10, 6))
    
    for r in [0.5, 1, 2, 4]:
        d = [negative_one_sided_desirability(val, y_min, y_max, r) for val in y]
        plt.plot(y, d, label=f'r = {r}')
    
    plt.axvline(x=y_min, color='gray', linestyle='--')
    plt.axvline(x=y_max, color='gray', linestyle='--')
    plt.xlabel('Response Value')
    plt.ylabel('Desirability')
    plt.title('Negative One-Sided Desirability Functions (Smaller is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/negative_one_sided_profiles.pdf')
    plt.close()

def plot_two_sided_profiles():
    """Plot examples of two-sided desirability functions with different shape parameters."""
    y = np.linspace(0, 10, 1000)
    y_min, y_max = 2, 8
    y_target = 5
    
    plt.figure(figsize=(10, 6))
    
    # Different combinations of r1 and r2
    r_pairs = [(1, 1), (0.5, 0.5), (2, 2), (0.5, 3), (3, 0.5)]
    
    for r1, r2 in r_pairs:
        d = [two_sided_desirability(val, y_min, y_max, y_target, r1, r2) for val in y]
        plt.plot(y, d, label=f'r1={r1}, r2={r2}')
    
    plt.axvline(x=y_min, color='gray', linestyle='--')
    plt.axvline(x=y_max, color='gray', linestyle='--')
    plt.axvline(x=y_target, color='red', linestyle='--')
    plt.xlabel('Response Value')
    plt.ylabel('Desirability')
    plt.title('Two-Sided Desirability Functions (Target is Optimal)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/two_sided_profiles.pdf')
    plt.close()

def plot_segmented_one_sided_profiles():
    """Plot examples of segmented one-sided desirability functions with different shape parameters."""
    y = np.linspace(0, 10, 1000)
    y_min, y_max = 2, 8
    y_mid = 5
    
    plt.figure(figsize=(10, 6))
    
    # Different combinations of r1 and r2
    r_pairs = [(1, 1), (0.5, 2), (2, 0.5), (0.5, 0.5), (3, 3)]
    
    for r1, r2 in r_pairs:
        d = [segmented_one_sided_desirability(val, y_min, y_max, y_mid, r1, r2) for val in y]
        plt.plot(y, d, label=f'r1={r1}, r2={r2}')
    
    plt.axvline(x=y_min, color='gray', linestyle='--')
    plt.axvline(x=y_max, color='gray', linestyle='--')
    plt.axvline(x=y_mid, color='red', linestyle='--')
    plt.xlabel('Response Value')
    plt.ylabel('Desirability')
    plt.title('Segmented One-Sided Desirability Functions (Larger is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/segmented_one_sided_profiles.pdf')
    plt.close()

def plot_segmented_negative_one_sided_profiles():
    """Plot examples of segmented negative one-sided desirability functions with different shape parameters."""
    d_min, d_max = 2, 8
    d_mid = 5
    y = np.linspace(0, 10, 1000)  # Expanded range for better visualization
    
    plt.figure(figsize=(10, 6))
    
    # Different combinations of r1 and r2 to match the image
    r_pairs = [(0.5, 0.5), (1, 1), (2, 2)]  # s_j > 1, s_j = 1, s_j < 1
    labels = [("s_j < 1", "t_j < 1"), ("s_j = 1", "t_j = 1"), ("s_j > 1", "t_j > 1")]
    
    for i, ((r1, r2), (label_s, label_t)) in enumerate(zip(r_pairs, labels)):
        plt.figure(figsize=(10, 6))

        # Calculate desirability
        d = [segmented_negative_one_sided_desirability(val, d_min, d_max, d_mid, r1, r2) for val in y]
        plt.plot(y, d, linewidth=2, color='black')

        # Add vertical lines
        plt.axvline(x=d_min, color='gray', linestyle='-', linewidth=0.5)
        plt.axvline(x=d_max, color='gray', linestyle='-', linewidth=0.5)
        plt.axvline(x=d_mid, color='gray', linestyle='-', linewidth=0.5)

        # Add text labels at the boundaries
        plt.text(d_min, -0.15, r'$D_j^{min}$', ha='center', va='top')
        plt.text(d_mid, -0.15, r'$D_j^{baseline}$', ha='center', va='top')
        plt.text(d_max, -0.15, r'$D_j^{max}$', ha='center', va='top')

        # Add parameter labels on the curves
        plt.text(3.5, 0.85, label_s, ha='center', rotation=-25)
        plt.text(7, 0.10, label_t, ha='center', rotation=-25)

        # Set axis labels and limits
        plt.xlabel(r'$\hat{D}_i(x)$', labelpad=20, loc='right')
        plt.ylabel(r'$d_i(\hat{D}_i(x))$', labelpad=20, loc='top')
        plt.xlim(0, 10)
        plt.ylim(-0.1, 1.1)

        # Configure ticks
        plt.xticks([])  # Hide x-ticks as in the image
        plt.yticks([0, 1])
        
        # Remove legend as the image uses direct labels
        plt.grid(False)
        
        plt.tight_layout()
        plt.savefig(f'figures/segmented_negative_one_sided_profiles_{i}.pdf')
        plt.close()

if __name__ == "__main__":
    print("Generating desirability function profile plots...")
    plot_one_sided_profiles()
    plot_negative_one_sided_profiles()
    plot_two_sided_profiles()
    plot_segmented_one_sided_profiles()
    plot_segmented_negative_one_sided_profiles()
    print("All plots have been saved to the 'figures' directory.")
