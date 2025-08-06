import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def multimodal_func(x1, x2):
    return (2 * np.sin(0.8 * x1) * np.cos(0.6 * x2) +
            1.5 * np.exp(-0.1 * ((x1 - 3)**2 + (x2 - 7)**2)) +
            2.0 * np.exp(-0.1 * ((x1 - 7)**2 + (x2 - 3)**2)) +
            0.5 * np.sin(1.2 * x1 + 0.8 * x2) +
            0.3 * (x1 + x2))

def sinusoidal_func(x1, x2):
    return (3 * np.sin(x1) * np.sin(x2) +
            2 * np.cos(0.5 * x1 + 0.3 * x2) +
            0.5 * x1 * x2 / 10)

def polynomial_func(x1, x2):
    return (0.1 * x1**2 + 0.05 * x2**2 +
            0.02 * x1 * x2 +
            0.3 * x1 + 0.2 * x2 +
            0.001 * x1**3 - 0.001 * x2**3)

def rbf_mixture_func(x1, x2):
    centers = np.array([[2, 2], [5, 8], [8, 3], [6, 6]])
    scales = np.array([1.5, 2.0, 1.8, 1.2])
    weights = np.array([2.0, -1.5, 3.0, -1.0])
    y = np.zeros_like(x1)
    for center, scale, weight in zip(centers, scales, weights):
        distances = np.sqrt((x1 - center[0])**2 + (x2 - center[1])**2)
        y += weight * np.exp(-distances**2 / (2 * scale**2))
    y += 0.1 * x1 + 0.05 * x2
    return y

def plot_height_map(func, name, save_dir):
    grid_size = 100
    x1 = np.linspace(0, 10, grid_size)
    x2 = np.linspace(0, 10, grid_size)
    x1_mesh, x2_mesh = np.meshgrid(x1, x2)
    y_mesh = func(x1_mesh, x2_mesh)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x1_mesh, x2_mesh, y_mesh, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title(f'Height Map: {name}')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'height_map.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, 'height_map.png')}")

if __name__ == "__main__":
    base_dir = "E:\TUM\RCI-S5-SS25\GP\Practice\data\synthetic"
    plot_height_map(multimodal_func, "Multimodal", os.path.join(base_dir, "multimodal"))
    plot_height_map(sinusoidal_func, "Sinusoidal", os.path.join(base_dir, "sinusoidal"))
    plot_height_map(polynomial_func, "Polynomial", os.path.join(base_dir, "polynomial"))
    plot_height_map(rbf_mixture_func, "RBF Mixture", os.path.join(base_dir, "rbf_mixture"))
