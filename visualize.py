'''
Visualize various aspects of cells of arrays
'''
import numpy as np
import matplotlib.pyplot as plt

def plot_orthogonal_views(data_3d, center=None, title="Orthogonal Views"):
    """
    Plots the three orthogonal views (Axial, Coronal, Sagittal) of a 3D Numpy array.

    Parameters:
    - data_3d: 3D numpy array (shape: Z, Y, X).
    - center: Tuple of (z, y, x) coordinates to slice through.
    """
    if data_3d.ndim != 3:
        raise ValueError("Input data must be a 3D array.")

    nz, ny, nx = data_3d.shape

    # Use center of volume if specific coordinates aren't provided
    if center is None:
        cz, cy, cx = nz // 2, ny // 2, nx // 2
    else:
        cz, cy, cx = center

    # Create figure and 3 axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title} @ Center: ({cz}, {cy}, {cx})", fontsize=16)

    # --- View 1: Axial (XY plane, slice at fixed Z) ---
    ax1 = axes[0]
    # Slicing: [z, y, x] -> we lock z
    ax1.imshow(data_3d[cz, :, :], cmap='viridis', origin='lower', aspect='auto')
    ax1.axvline(x=cx, color='r', linestyle='--', alpha=0.7)
    ax1.axhline(y=cy, color='r', linestyle='--', alpha=0.7)
    ax1.set_title(f"Axial (Z={cz})")
    ax1.set_xlabel("X axis")
    ax1.set_ylabel("Y axis")

    # --- View 2: Coronal (XZ plane, slice at fixed Y) ---
    ax2 = axes[1]
    # Slicing: [z, y, x] -> we lock y
    ax2.imshow(data_3d[:, cy, :], cmap='viridis', origin='lower', aspect='auto')
    ax2.axvline(x=cx, color='r', linestyle='--', alpha=0.7)
    ax2.axhline(y=cz, color='r', linestyle='--', alpha=0.7)
    ax2.set_title(f"Coronal (Y={cy})")
    ax2.set_xlabel("X axis")
    ax2.set_ylabel("Z axis")

    # --- View 3: Sagittal (YZ plane, slice at fixed X) ---
    ax3 = axes[2]
    # Slicing: [z, y, x] -> we lock x
    ax3.imshow(data_3d[:, :, cx], cmap='viridis', origin='lower', aspect='auto')
    ax3.axvline(x=cy, color='r', linestyle='--', alpha=0.7)
    ax3.axhline(y=cz, color='r', linestyle='--', alpha=0.7)
    ax3.set_title(f"Sagittal (X={cx})")
    ax3.set_xlabel("Y axis")
    ax3.set_ylabel("Z axis")

    plt.tight_layout()
    plt.show()
