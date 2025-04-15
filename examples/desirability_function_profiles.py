import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import sys
import os

plt.rcParams.update({'font.size': 22})
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

def plot_profile_tuning_profiles():
    """Plot examples of segmented negative one-sided desirability functions with different shape parameters."""
    d_min, d_max = 2, 8
    d_mid = 5
    y = np.linspace(0, 10, 1000)  # Expanded range for better visualization
    
    # Different combinations of r1 and r2 to match the image
    r_pairs = [(0.5, 0.5), (1, 1), (2, 2)]  # s_j > 1, s_j = 1, s_j < 1
    labels = [("$s_i < 1$", "$t_i < 1$"), ("$s_i = 1$", "$t_i = 1$"), ("$s_i > 1$", "$t_i > 1$")]
    rotation_s = [-28, -40, -43]
    rotation_t = [-28, -40, -43]
    y_position_s = [0.85, 0.75, 0.63]
    y_position_t = [0.03, 0.11, 0.23]
    
    for i, ((r1, r2), (label_s, label_t)) in enumerate(zip(r_pairs, labels)):
        plt.figure(figsize=(10, 6))

        # Calculate desirability
        d = [segmented_negative_one_sided_desirability(val, d_min, d_max, d_mid, gamma=0.5, r1=r1, r2=r2) for val in y]
        plt.plot(y, d, linewidth=2.5, color='black')

        # Add vertical lines
        plt.axvline(x=d_min, color='black', linestyle='--', linewidth=2.5)
        plt.axvline(x=d_max, color='black', linestyle='--', linewidth=2.5)
        plt.axvline(x=d_mid, color='black', linestyle='--', linewidth=2.5)

        # Add text labels at the boundaries
        plt.text(d_min, -0.15, r'$D_i^{min}$', ha='center', va='top')
        plt.text(d_mid, -0.15, r'$D_i^{baseline}$', ha='center', va='top')
        plt.text(d_max, -0.15, r'$D_i^{max}$', ha='center', va='top')

        # Add parameter labels on the curves
        plt.text(3.5, y_position_s[i], label_s, ha='center', rotation=rotation_s[i])
        plt.text(6.5, y_position_t[i], label_t, ha='center', rotation=rotation_t[i])

        # Set axis labels and limits
        plt.xlabel(r'$D_i(x)$', labelpad=20, loc='right')
        plt.ylabel(r'$d_i(D_i(x))$', labelpad=20, loc='top')
        plt.xlim(0, 10)
        plt.ylim(-0.1, 1.1)

        # Configure ticks
        plt.xticks([])  # Hide x-ticks as in the image
        plt.yticks([0, 1])
        
        # Remove legend as the image uses direct labels
        plt.grid(False)
        
        plt.tight_layout()
        plt.savefig(f'figures/segmented_negative_one_sided_profiles_{i}.pdf')
        plt.savefig(f'figures/segmented_negative_one_sided_profiles_{i}.png')
        plt.close()

def plot_range_tuning_profiles():
    """Plot examples of desirability functions with different range sizes."""
    y = np.linspace(0, 10, 1000)
    center = 5  # Central point of the range
    
    # Different range sizes
    range_sizes = [(center - 2, center + 2), (center - 2.5, center + 2.5), (center - 3, center + 3)]
    labels = ["Reduced Range", "Regular Range", "Increased Range"]
    
    for i, (d_min, d_max) in enumerate(range_sizes):
        plt.figure(figsize=(10, 6))
        
        # Calculate desirability
        d = [segmented_negative_one_sided_desirability(val, d_min, d_max, center) for val in y]
        plt.plot(y, d, color='black', linewidth=2.5)
        
        # Add vertical lines
        plt.axvline(x=d_min, color='black', linestyle='--', linewidth=2.5)
        plt.axvline(x=d_max, color='black', linestyle='--', linewidth=2.5)
        
        # Add text labels at the boundaries
        plt.text(d_min, -0.15, r'$D_i^{min}$', ha='center', va='top')
        plt.text(center, -0.15, r'$D_i^{baseline}$', ha='center', va='top')
        plt.text(d_max, -0.15, r'$D_i^{max}$', ha='center', va='top')

        # Set axis labels and limits
        plt.xlabel(r'$D_i(x)$', labelpad=20, loc='right')
        plt.ylabel(r'$d_i(D_i(x))$', labelpad=20, loc='top')
        plt.xlim(0, 10)
        plt.ylim(-0.1, 1.1)
        
        # Configure ticks
        plt.xticks([])  # Hide x-ticks as in the image
        plt.yticks([0, 1])
        
        # Configure plot appearance
        plt.grid(False)
        
        plt.tight_layout()
        plt.savefig(f'figures/range_tuning_profile_{i}.pdf')
        plt.savefig(f'figures/range_tuning_profile_{i}.png')
        plt.close()

def plot_gamma_tuning_profiles():
    """Plot examples of desirability functions with different gamma values."""
    y = np.linspace(0, 10, 1000)
    center = 5  # Central point of the range
    
    # Different gamma values
    gamma_values = [0.3, 0.5, 0.7]
    labels = ["Lower Gamma (γ=0.3)", "Standard Gamma (γ=0.5)", "Higher Gamma (γ=0.7)"]
    
    d_min, d_max = 2, 8
    d_mid = center
    
    for i, gamma in enumerate(gamma_values):
        plt.figure(figsize=(10, 6))
        
        # Calculate desirability using segmented_one_sided_desirability with different gamma values
        d = [segmented_negative_one_sided_desirability(val, d_min, d_max, d_mid, gamma=gamma, r1=1, r2=1) for val in y]
        plt.plot(y, d, color='black', linewidth=2.5)
        
        # Add vertical lines
        plt.axvline(x=d_min, color='black', linestyle='--', linewidth=2.5)
        plt.axvline(x=d_max, color='black', linestyle='--', linewidth=2.5)
        plt.axvline(x=d_mid, color='black', linestyle='--', linewidth=2.5)
        
        # Add text labels at the boundaries
        plt.text(d_min, -0.15, r'$D_i^{min}$', ha='center', va='top')
        plt.text(d_mid, -0.15, r'$D_i^{baseline}$', ha='center', va='top')
        plt.text(d_max, -0.15, r'$D_i^{max}$', ha='center', va='top')
        
        # Add gamma value label
        plt.text(7, 0.5, f"γ = {gamma}", ha='center')
        
        # Set axis labels and limits
        plt.xlabel(r'$D_i(x)$', labelpad=20, loc='right')
        plt.ylabel(r'$d_i(D_i(x))$', labelpad=20, loc='top')
        plt.xlim(0, 10)
        plt.ylim(-0.1, 1.1)
        
        # Configure ticks
        plt.xticks([])  # Hide x-ticks as in the image
        plt.yticks([0, 1])
        
        # Configure plot appearance
        plt.grid(False)
        
        plt.tight_layout()
        plt.savefig(f'figures/gamma_tuning_profile_{i}.pdf')
        plt.savefig(f'figures/gamma_tuning_profile_{i}.png')
        plt.close()

def plot_center_tuning_profiles():
    """Plot examples of desirability functions with different center points."""
    y = np.linspace(0, 10, 1000)
    
    # Different center points
    center_values = [4, 5, 6]
    labels = ["Lower Center", "Standard Center", "Higher Center"]
    
    d_min, d_max = 2, 8
    
    for i, center in enumerate(center_values):
        plt.figure(figsize=(10, 6))
        
        # Calculate desirability with different center points
        d = [segmented_negative_one_sided_desirability(val, d_min, d_max, center, gamma=0.5, r1=1, r2=1) for val in y]
        plt.plot(y, d, color='black', linewidth=2.5)
        
        # Add vertical lines
        plt.axvline(x=d_min, color='black', linestyle='--', linewidth=2.5)
        plt.axvline(x=d_max, color='black', linestyle='--', linewidth=2.5)
        plt.axvline(x=center, color='black', linestyle='--', linewidth=2.5)
        
        # Add text labels at the boundaries
        plt.text(d_min, -0.15, r'$D_i^{min}$', ha='center', va='top')
        plt.text(center, -0.15, r'$D_i^{baseline}$', ha='center', va='top')
        plt.text(d_max, -0.15, r'$D_i^{max}$', ha='center', va='top')
        
        # Set axis labels and limits
        plt.xlabel(r'$D_i(x)$', labelpad=20, loc='right')
        plt.ylabel(r'$d_i(D_i(x))$', labelpad=20, loc='top')
        plt.xlim(0, 10)
        plt.ylim(-0.1, 1.1)
        
        # Configure ticks
        plt.xticks([])  # Hide x-ticks as in the image
        plt.yticks([0, 1])
        
        # Configure plot appearance
        plt.grid(False)
        
        plt.tight_layout()
        plt.savefig(f'figures/center_tuning_profile_{i}.pdf')
        plt.savefig(f'figures/center_tuning_profile_{i}.png')
        plt.close()
    

if __name__ == "__main__":
    print("Generating desirability function profile plots...")
    plot_profile_tuning_profiles()
    plot_range_tuning_profiles()
    plot_gamma_tuning_profiles()
    plot_center_tuning_profiles()
    print("All plots have been saved to the 'figures' directory.")
