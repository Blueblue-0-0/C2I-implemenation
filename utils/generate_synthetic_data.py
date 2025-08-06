import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import pdist, squareform
import torch

def generate_synthetic_2d_dataset(
    n_samples=10000,
    input_bounds=(0, 10),
    noise_level=0.1,
    function_type='multimodal',
    random_seed=42
):
    """
    Generate synthetic 2D input, 1D output dataset
    
    Args:
        n_samples: Number of samples to generate
        input_bounds: Tuple of (min, max) for input space
        noise_level: Standard deviation of noise
        function_type: Type of function ('multimodal', 'sinusoidal', 'polynomial', 'rbf_mixture')
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    
    # Generate 2D input points
    x_min, x_max = input_bounds
    x1 = np.random.uniform(x_min, x_max, n_samples)
    x2 = np.random.uniform(x_min, x_max, n_samples)
    X = np.column_stack([x1, x2])
    
    # Generate output based on function type
    if function_type == 'multimodal':
        # Multi-modal function with several peaks and valleys
        y = (2 * np.sin(0.8 * x1) * np.cos(0.6 * x2) + 
             1.5 * np.exp(-0.1 * ((x1 - 3)**2 + (x2 - 7)**2)) +
             2.0 * np.exp(-0.1 * ((x1 - 7)**2 + (x2 - 3)**2)) +
             0.5 * np.sin(1.2 * x1 + 0.8 * x2) +
             0.3 * (x1 + x2))
    
    elif function_type == 'sinusoidal':
        # Sinusoidal function
        y = (3 * np.sin(x1) * np.sin(x2) + 
             2 * np.cos(0.5 * x1 + 0.3 * x2) +
             0.5 * x1 * x2 / 10)
    
    elif function_type == 'polynomial':
        # Polynomial function
        y = (0.1 * x1**2 + 0.05 * x2**2 + 
             0.02 * x1 * x2 + 
             0.3 * x1 + 0.2 * x2 +
             0.001 * x1**3 - 0.001 * x2**3)
    
    elif function_type == 'rbf_mixture':
        # RBF mixture (mixture of Gaussians)
        centers = np.array([[2, 2], [5, 8], [8, 3], [6, 6]])
        scales = np.array([1.5, 2.0, 1.8, 1.2])
        weights = np.array([2.0, -1.5, 3.0, -1.0])
        
        y = np.zeros(n_samples)
        for center, scale, weight in zip(centers, scales, weights):
            distances = np.sqrt(np.sum((X - center)**2, axis=1))
            y += weight * np.exp(-distances**2 / (2 * scale**2))
        
        # Add linear trend
        y += 0.1 * x1 + 0.05 * x2
    
    else:
        raise ValueError(f"Unknown function type: {function_type}")
    
    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    y_noisy = y + noise
    
    # Create dataset dictionary - FIX: Use 'y' instead of 'y_true'
    dataset = {
        'X': X,
        'y_true': y,        # FIXED: Changed from y_true to y
        'y_noisy': y_noisy,
        'function_type': function_type,
        'noise_level': noise_level,
        'input_bounds': input_bounds,
        'n_samples': n_samples
    }
    
    return dataset

def save_synthetic_dataset(dataset, agent_data, inducing_points, inducing_y, 
                          save_dir="project/data/synthetic", test_ratio=0.2):
    """Save all generated data to files with SHORTER file names"""
    os.makedirs(save_dir, exist_ok=True)
    
    X = dataset['X']
    y_true = dataset['y_true']
    y_noisy = dataset['y_noisy']
    n_samples = len(X)
    
    # STEP 1: Create proper train/test split FIRST
    test_size = int(n_samples * test_ratio)
    train_size = n_samples - test_size
    
    # Use random indices for train/test split
    np.random.seed(42)  # For reproducible splits
    all_indices = np.arange(n_samples)
    np.random.shuffle(all_indices)
    
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:]
    
    print(f"Data split: {train_size} training, {test_size} testing samples")
    
    # STEP 2: Create training data (what agents will see)
    X_train = X[train_indices]
    y_true_train = y_true[train_indices]
    y_noisy_train = y_noisy[train_indices]
    
    # STEP 3: Create test data (completely separate, agents never see this)
    X_test = X[test_indices]
    y_true_test = y_true[test_indices] 
    y_noisy_test = y_noisy[test_indices]
    
    # STEP 4: Save FULL dataset (for reference only) - SHORTER NAME
    full_df = pd.DataFrame({
        'x0': X[:, 0],
        'x1': X[:, 1],
        'y_true': y_true,
        'y': y_noisy,
        'split': ['train' if i in train_indices else 'test' for i in range(n_samples)]
    })
    full_path = os.path.join(save_dir, 'full.csv')
    full_df.to_csv(full_path, index=False)
    print(f"Full dataset saved to: {full_path}")
    
    # STEP 5: Re-separate TRAINING data for agents (not the full dataset!)
    train_dataset = {
        'X': X_train,
        'y_true': y_true_train,
        'y_noisy': y_noisy_train,
        'function_type': dataset['function_type'],
        'noise_level': dataset['noise_level'],
        'input_bounds': dataset['input_bounds'],
        'n_samples': len(X_train)
    }
    
    # FIXED: Remove num_agents and method parameters
    agent_data_train = separate_data_for_agents(
        train_dataset, 
        overlap_ratio=0.1  # Only pass overlap_ratio
    )
    
    # STEP 6: Save agent training data - SHORTER NAMES
    agent_paths = []
    for i, data in enumerate(agent_data_train):
        agent_df = pd.DataFrame({
            'x0': data['x'][:, 0],
            'x1': data['x'][:, 1], 
            'y': data['y']
        })
        agent_path = os.path.join(save_dir, f'agent{i+1}.csv')
        agent_df.to_csv(agent_path, index=False)
        agent_paths.append(agent_path)
        print(f"Agent {i+1} training data saved to: {agent_path} ({len(agent_df)} samples)")
    
    # STEP 7: Re-generate inducing points from TRAINING data only
    inducing_points_train, inducing_y_train = generate_inducing_points(
        train_dataset, 
        num_points=len(inducing_points),
        method='kmeans'
    )
    
    # Save inducing points - SHORTER NAME
    inducing_df = pd.DataFrame({
        'agent_idx': np.repeat(range(len(agent_data_train)), len(inducing_points_train) // len(agent_data_train)),
        'inducing_idx': np.tile(range(len(inducing_points_train) // len(agent_data_train)), len(agent_data_train)),
        'x0': inducing_points_train[:, 0],
        'x1': inducing_points_train[:, 1],
        'y': inducing_y_train
    })
    inducing_path = os.path.join(save_dir, 'inducing.csv')
    inducing_df.to_csv(inducing_path, index=False)
    print(f"Inducing points saved to: {inducing_path}")
    
    # STEP 8: Save TEST data (completely separate!) - SHORTER NAME
    test_df = pd.DataFrame({
        'x0': X_test[:, 0],
        'x1': X_test[:, 1],
        'y_true': y_true_test,
        'y': y_noisy_test
    })
    test_path = os.path.join(save_dir, 'test.csv')
    test_df.to_csv(test_path, index=False)
    print(f"Test data saved to: {test_path} ({len(test_df)} samples)")
    
    # STEP 9: Create visualization showing train/test split - SHORTER NAME
    visualize_train_test_split(X_train, X_test, y_true_train, y_true_test, save_dir, dataset['function_type'])
    
    return {
        'full_path': full_path,
        'agent_paths': agent_paths,
        'inducing_path': inducing_path,
        'test_path': test_path,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

def add_agent_domain_boundaries(ax, X, alpha=0.8, overlap_ratio=0.1):
    """Add agent domain boundaries to a plot - Fixed to 4 agents with CORRECT mapping"""
    
    # Calculate domain boundaries (2x2 grid)
    x1_bounds = np.linspace(X[:, 0].min(), X[:, 0].max(), 3)  # 3 points for 2x2 grid
    x2_bounds = np.linspace(X[:, 1].min(), X[:, 1].max(), 3)  # 3 points for 2x2 grid
    
    # Define colors for each agent
    agent_colors = ['red', 'blue', 'green', 'orange']
    
    for i in range(4):  # Fixed to 4 agents
        # FIXED: Correct 2x2 grid mapping
        if i == 0:      # Agent 1: Bottom-left
            row, col = 0, 0
        elif i == 1:    # Agent 2: Bottom-right
            row, col = 0, 1
        elif i == 2:    # Agent 3: Top-left
            row, col = 1, 0
        elif i == 3:    # Agent 4: Top-right
            row, col = 1, 1
        
        # Define region bounds (core domain without overlap)
        x1_min = x1_bounds[col]      # FIXED: Use col for x1 (horizontal)
        x1_max = x1_bounds[col + 1]  # FIXED: Use col for x1 (horizontal)
        x2_min = x2_bounds[row]      # FIXED: Use row for x2 (vertical)
        x2_max = x2_bounds[row + 1]  # FIXED: Use row for x2 (vertical)
        
        # Draw main domain boundary (solid line)
        ax.plot([x1_min, x1_max, x1_max, x1_min, x1_min], 
                [x2_min, x2_min, x2_max, x2_max, x2_min], 
                color=agent_colors[i], linewidth=2, alpha=alpha, 
                label=f'Agent {i+1}')
        
        # Add agent label in center of domain
        center_x1 = (x1_min + x1_max) / 2
        center_x2 = (x2_min + x2_max) / 2
        ax.text(center_x1, center_x2, f'A{i+1}', 
                fontsize=12, fontweight='bold', 
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor=agent_colors[i], 
                         alpha=0.7, edgecolor='white'))
        
        # Draw overlap boundaries (dashed lines)
        if overlap_ratio > 0:
            overlap_size = overlap_ratio * (x1_bounds[1] - x1_bounds[0])
            x1_min_overlap = x1_min - overlap_size
            x1_max_overlap = x1_max + overlap_size
            x2_min_overlap = x2_min - overlap_size
            x2_max_overlap = x2_max + overlap_size
            
            # Only draw if within data bounds
            data_x1_min, data_x1_max = X[:, 0].min(), X[:, 0].max()
            data_x2_min, data_x2_max = X[:, 1].min(), X[:, 1].max()
            
            x1_min_overlap = max(x1_min_overlap, data_x1_min)
            x1_max_overlap = min(x1_max_overlap, data_x1_max)
            x2_min_overlap = max(x2_min_overlap, data_x2_min)
            x2_max_overlap = min(x2_max_overlap, data_x2_max)
            
            ax.plot([x1_min_overlap, x1_max_overlap, x1_max_overlap, x1_min_overlap, x1_min_overlap], 
                    [x2_min_overlap, x2_min_overlap, x2_max_overlap, x2_max_overlap, x2_min_overlap], 
                    color=agent_colors[i], linewidth=1, alpha=alpha*0.5, linestyle='--')
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)

def visualize_dataset(dataset, save_dir="project/data/synthetic", show_plots=True, show_agent_domains=True):
    """Visualize the generated dataset with SHORTER file names and agent domains"""
    os.makedirs(save_dir, exist_ok=True)
    
    X = dataset['X']
    y_true = dataset['y_true']
    y_noisy = dataset['y_noisy']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. True function surface WITH AGENT DOMAINS
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=1)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_title('True Function (without noise)')
    plt.colorbar(scatter1, ax=ax1)
    
    # Add agent domain boundaries
    if show_agent_domains:
        add_agent_domain_boundaries(ax1, X, alpha=0.8)  # Removed num_agents parameter
    
    # 2. Noisy observations WITH AGENT DOMAINS
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=y_noisy, cmap='viridis', alpha=0.6, s=1)
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_title('Noisy Observations')
    plt.colorbar(scatter2, ax=ax2)
    
    # Add agent domain boundaries
    if show_agent_domains:
        add_agent_domain_boundaries(ax2, X, alpha=0.8)  # Removed num_agents parameter
    
    # 3. Noise distribution
    ax3 = axes[1, 0]
    noise = y_noisy - y_true
    ax3.hist(noise, bins=50, alpha=0.7, density=True)
    ax3.set_xlabel('Noise')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Noise Distribution (σ={dataset["noise_level"]:.3f})')
    
    # 4. Y distribution
    ax4 = axes[1, 1]
    ax4.hist(y_noisy, bins=50, alpha=0.7, label='Noisy', density=True)
    ax4.hist(y_true, bins=50, alpha=0.7, label='True', density=True)
    ax4.set_xlabel('Y values')
    ax4.set_ylabel('Density')
    ax4.set_title('Output Distribution')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot - SHORTER NAME
    plot_path = os.path.join(save_dir, f'overview.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    print(f"Dataset visualization saved to: {plot_path}")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  - Function type: {dataset['function_type']}")
    print(f"  - Samples: {dataset['n_samples']}")
    print(f"  - Input bounds: {dataset['input_bounds']}")
    print(f"  - Noise level: {dataset['noise_level']}")
    print(f"  - X1 range: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]")
    print(f"  - X2 range: [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")
    print(f"  - Y_true range: [{y_true.min():.3f}, {y_true.max():.3f}]")
    print(f"  - Y_noisy range: [{y_noisy.min():.3f}, {y_noisy.max():.3f}]")
    print(f"  - SNR: {np.var(y_true) / np.var(y_noisy - y_true):.2f}")

def visualize_agent_domains_detailed(dataset, save_dir="project/data/synthetic"):
    """Create a detailed visualization showing agent domains and data distribution"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    X = dataset['X']
    y_noisy = dataset['y_noisy']
    
    # Generate agent data to show actual assignments
    agent_data = separate_data_for_agents(
        dataset, 
        overlap_ratio=0.1  # Removed num_agents and method parameters
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Agent Domain Analysis - {dataset["function_type"].title()} Function', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Overall domain boundaries
    ax = axes[0, 0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_noisy, cmap='viridis', alpha=0.4, s=1)
    add_agent_domain_boundaries(ax, X, alpha=1.0)  # Removed num_agents parameter
    ax.set_title('Agent Domain Boundaries')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    plt.colorbar(scatter, ax=ax)
    
    # Plot 2-5: Individual agent data
    agent_colors = ['red', 'blue', 'green', 'orange']
    
    for i, (agent, color) in enumerate(zip(agent_data, agent_colors)):
        if i < 4:  # Only plot first 4 agents (redundant check since we always have 4)
            row = (i + 1) // 3
            col = (i + 1) % 3
            ax = axes[row, col]
            
            # Plot all data in gray
            ax.scatter(X[:, 0], X[:, 1], c='lightgray', alpha=0.2, s=1, label='Other data')
            
            # Plot this agent's data in color
            agent_x = agent['x']
            agent_y = agent['y']
            scatter = ax.scatter(agent_x[:, 0], agent_x[:, 1], c=agent_y, 
                               cmap='viridis', alpha=0.8, s=2, edgecolor=color, linewidth=0.1)
            
            ax.set_title(f'Agent {i+1} Data ({len(agent_x)} samples)')
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            
            # Add domain boundary for this agent
            add_single_agent_boundary(ax, X, agent_id=i, color=color)  # Removed num_agents parameter
            
            # Add statistics
            ax.text(0.02, 0.98, f'Samples: {len(agent_x)}\n'
                              f'X1: [{agent_x[:, 0].min():.1f}, {agent_x[:, 0].max():.1f}]\n'
                              f'X2: [{agent_x[:, 1].min():.1f}, {agent_x[:, 1].max():.1f}]\n'
                              f'Y: [{agent_y.min():.2f}, {agent_y.max():.2f}]',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 6: Agent overlap analysis
    ax = axes[1, 2]
    
    # Create overlap heatmap
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    
    overlap_count = np.zeros_like(x1_grid)
    
    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            point = np.array([x1_grid[j, i], x2_grid[j, i]])
            count = 0
            for agent in agent_data:
                agent_x = agent['x']
                # Check if point is within agent's domain bounds
                if (point[0] >= agent_x[:, 0].min() and point[0] <= agent_x[:, 0].max() and
                    point[1] >= agent_x[:, 1].min() and point[1] <= agent_x[:, 1].max()):
                    count += 1
            overlap_count[j, i] = count
    
    im = ax.imshow(overlap_count, extent=[X[:, 0].min(), X[:, 0].max(), 
                                         X[:, 1].min(), X[:, 1].max()], 
                   origin='lower', cmap='Reds', alpha=0.7)
    ax.set_title('Agent Domain Overlap\n(Darker = More Overlap)')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    plt.colorbar(im, ax=ax, label='Number of Agents')
    
    plt.tight_layout()
    
    # Save detailed analysis
    plot_path = os.path.join(save_dir, 'agent_domains_detailed.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed agent domain analysis saved to: {plot_path}")
    
    # Print agent statistics
    print(f"\nAgent Domain Statistics:")
    for i, agent in enumerate(agent_data):
        agent_x = agent['x']
        print(f"  Agent {i+1}: {len(agent_x)} samples in region "
              f"[{agent_x[:, 0].min():.1f},{agent_x[:, 0].max():.1f}] × "
              f"[{agent_x[:, 1].min():.1f},{agent_x[:, 1].max():.1f}]")

def add_single_agent_boundary(ax, X, agent_id, color, overlap_ratio=0.1):
    """Add boundary for a single agent - Fixed to 4 agents with CORRECT mapping"""
    
    x1_bounds = np.linspace(X[:, 0].min(), X[:, 0].max(), 3)  # 3 points for 2x2 grid
    x2_bounds = np.linspace(X[:, 1].min(), X[:, 1].max(), 3)  # 3 points for 2x2 grid
    
    # FIXED: Correct 2x2 grid mapping
    if agent_id == 0:      # Agent 1: Bottom-left
        row, col = 0, 0
    elif agent_id == 1:    # Agent 2: Bottom-right
        row, col = 0, 1
    elif agent_id == 2:    # Agent 3: Top-left
        row, col = 1, 0
    elif agent_id == 3:    # Agent 4: Top-right
        row, col = 1, 1
    
    # Core domain
    x1_min = x1_bounds[col]      # FIXED: Use col for x1 (horizontal)
    x1_max = x1_bounds[col + 1]  # FIXED: Use col for x1 (horizontal)
    x2_min = x2_bounds[row]      # FIXED: Use row for x2 (vertical)
    x2_max = x2_bounds[row + 1]  # FIXED: Use row for x2 (vertical)
    
    ax.plot([x1_min, x1_max, x1_max, x1_min, x1_min], 
            [x2_min, x2_min, x2_max, x2_max, x2_min], 
            color=color, linewidth=3, alpha=0.8)

def separate_data_for_agents(dataset, overlap_ratio=0.1):
    """
    Separate dataset into agent-specific training data using spatial separation
    Fixed to 4 agents in 2x2 grid
    
    Args:
        dataset: Dictionary with 'X', 'y_noisy', etc.
        overlap_ratio: Amount of overlap between agent domains (0.1 = 10%)
    """
    
    X = dataset['X']
    y = dataset['y_noisy']
    
    return separate_spatial_2x2(X, y, overlap_ratio)

def separate_spatial_2x2(X, y, overlap_ratio=0.1):
    """Separate data into 2x2 spatial grid with overlap - Fixed to 4 agents"""
    
    # Define spatial bounds
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    
    # Create 2x2 grid boundaries
    x1_mid = (x1_min + x1_max) / 2
    x2_mid = (x2_min + x2_max) / 2
    
    # Calculate overlap amounts
    x1_overlap = overlap_ratio * (x1_max - x1_min) / 2
    x2_overlap = overlap_ratio * (x2_max - x2_min) / 2
    
    # Define agent regions with overlap (Fixed 2x2 grid)
    agent_regions = [
        # Agent 1: Bottom-left
        {
            'x1_range': [x1_min, x1_mid + x1_overlap],
            'x2_range': [x2_min, x2_mid + x2_overlap]
        },
        # Agent 2: Bottom-right  
        {
            'x1_range': [x1_mid - x1_overlap, x1_max],
            'x2_range': [x2_min, x2_mid + x2_overlap]
        },
        # Agent 3: Top-left
        {
            'x1_range': [x1_min, x1_mid + x1_overlap],
            'x2_range': [x2_mid - x2_overlap, x2_max]
        },
        # Agent 4: Top-right
        {
            'x1_range': [x1_mid - x1_overlap, x1_max],
            'x2_range': [x2_mid - x2_overlap, x2_max]
        }
    ]
    
    agent_data = []
    
    for i, region in enumerate(agent_regions):
        # Find points in this agent's region
        x1_mask = ((X[:, 0] >= region['x1_range'][0]) & 
                   (X[:, 0] <= region['x1_range'][1]))
        x2_mask = ((X[:, 1] >= region['x2_range'][0]) & 
                   (X[:, 1] <= region['x2_range'][1]))
        
        agent_mask = x1_mask & x2_mask
        agent_indices = np.where(agent_mask)[0]
        
        agent_X = X[agent_indices]
        agent_y = y[agent_indices]
        
        agent_data.append({
            'x': agent_X,
            'y': agent_y,
            'indices': agent_indices,
            'region': region
        })
        
        print(f"Agent {i+1}: {len(agent_X)} samples in region "
              f"X1[{region['x1_range'][0]:.1f}, {region['x1_range'][1]:.1f}] × "
              f"X2[{region['x2_range'][0]:.1f}, {region['x2_range'][1]:.1f}]")
    
    return agent_data

def generate_inducing_points(dataset, num_points=100, method='kmeans'):
    """Generate inducing points for the dataset"""
    
    X = dataset['X']
    y = dataset['y_noisy']
    
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        
        # Use k-means to find representative points
        kmeans = KMeans(n_clusters=num_points, random_state=42, n_init=10)
        kmeans.fit(X)
        inducing_points = kmeans.cluster_centers_
        
        # Find closest actual data points to cluster centers
        from scipy.spatial.distance import cdist
        distances = cdist(inducing_points, X)
        closest_indices = np.argmin(distances, axis=1)
        inducing_y = y[closest_indices]
        
    elif method == 'random':
        # Random selection
        indices = np.random.choice(len(X), num_points, replace=False)
        inducing_points = X[indices]
        inducing_y = y[indices]
        
    elif method == 'grid':
        # Regular grid
        x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), int(np.sqrt(num_points)))
        x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), int(np.sqrt(num_points)))
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        inducing_points = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
        
        # Find closest actual data points
        from scipy.spatial.distance import cdist
        distances = cdist(inducing_points, X)
        closest_indices = np.argmin(distances, axis=1)
        inducing_y = y[closest_indices]
        
    else:
        raise ValueError(f"Unknown inducing point method: {method}")
    
    return inducing_points, inducing_y

def visualize_train_test_split(X_train, X_test, y_train, y_test, save_dir, function_type):
    """Visualize the train/test split with SHORTER file name"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Training data
    ax1 = axes[0]
    scatter1 = ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.6, s=2)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_title(f'Training Data ({len(X_train)} samples)')
    plt.colorbar(scatter1, ax=ax1)
    
    # Test data
    ax2 = axes[1]
    scatter2 = ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', alpha=0.6, s=2)
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_title(f'Test Data ({len(X_test)} samples)')
    plt.colorbar(scatter2, ax=ax2)
    
    # Combined view
    ax3 = axes[2]
    ax3.scatter(X_train[:, 0], X_train[:, 1], c='blue', alpha=0.3, s=1, label='Training')
    ax3.scatter(X_test[:, 0], X_test[:, 1], c='red', alpha=0.3, s=1, label='Test')
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    ax3.set_title('Train/Test Split Overview')
    ax3.legend()
    
    plt.tight_layout()
    
    # Save plot - SHORTER NAME
    plot_path = os.path.join(save_dir, f'split.png')  # SHORTENED: train_test_split_{function_type}.png -> split.png
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Train/test split visualization saved to: {plot_path}")

# Fix the duplicate line in main() function:
def main():
    """Generate optimized synthetic datasets"""
    print("Generating optimized synthetic datasets...")
    
    function_configs = {
        'sinusoidal': {
            'n_samples': 1500,
            'noise_level': 0.1,
            'test_ratio': 0.2
        },
        'polynomial': {
            'n_samples': 2500,
            'noise_level': 0.12,
            'test_ratio': 0.2
        },
        'multimodal': {
            'n_samples': 4000,
            'noise_level': 0.15,
            'test_ratio': 0.2
        },
        'rbf_mixture': {
            'n_samples': 3000,
            'noise_level': 0.13,
            'test_ratio': 0.2
        }
    }
    
    for func_type, config in function_configs.items():
        print(f"\n{'='*60}")
        print(f"Generating {func_type} dataset with {config['n_samples']} samples...")
        print(f"{'='*60}")
        
        # Generate dataset with optimized size
        dataset = generate_synthetic_2d_dataset(
            n_samples=config['n_samples'],
            input_bounds=(0, 10),
            noise_level=config['noise_level'],
            function_type=func_type,
            random_seed=42
        )
        
        # Visualize full dataset with agent domains
        visualize_dataset(dataset, 
                         save_dir=f"project/data/synthetic/{func_type}",
                         show_plots=False, 
                         show_agent_domains=True)
        
        # Create detailed agent domain analysis
        visualize_agent_domains_detailed(dataset, 
                                       save_dir=f"project/data/synthetic/{func_type}")
        
        # Create initial agent separation from FULL dataset
        agent_data_temp = separate_data_for_agents(
            dataset, 
            overlap_ratio=0.1  # Removed num_agents and method parameters
        )
        
        # Generate initial inducing points (will be regenerated from training data)
        inducing_points_temp, inducing_y_temp = generate_inducing_points(
            dataset, 
            num_points=100,
            method='kmeans'
        )
        
        # Save with proper train/test separation
        file_paths = save_synthetic_dataset(
            dataset, 
            agent_data_temp,  # This will be regenerated from training data
            inducing_points_temp, 
            inducing_y_temp,
            save_dir=f"project/data/synthetic/{func_type}",
            test_ratio=0.2
        )
        
        print(f"\n{func_type} dataset complete!")
        print(f"  - Training samples: {file_paths['train_samples']}")
        print(f"  - Test samples: {file_paths['test_samples']}")
        print(f"  - Train/Test ratio: {file_paths['train_samples']/file_paths['test_samples']:.1f}:1")

        # Plot height map (surface plot) of the true function over its input domain
        print("Plotting height map of true function...")
        X = dataset['X']
        y_true = dataset['y_true']
        # Create grid for surface plot
        grid_size = 100
        x1_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), grid_size)
        x2_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), grid_size)
        x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
        # Interpolate y_true onto grid
        from scipy.interpolate import griddata
        y_grid = griddata(X, y_true, (x1_mesh, x2_mesh), method='cubic')

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x1_mesh, x2_mesh, y_grid, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('True Y')
        ax.set_title(f'Height Map of {func_type.title()} Function')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.savefig(f"project/data/synthetic/{func_type}/height_map.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Height map saved to: project/data/synthetic/{func_type}/height_map.png")

if __name__ == "__main__":
    main()