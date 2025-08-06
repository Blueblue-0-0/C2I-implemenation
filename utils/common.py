import os
import numpy as np
import pandas as pd
import torch


# Set the root directory for all synthetic data
DATA_ROOT = "project/data/synthetic"

def load_synthetic_agent_data(agent_idx, function_type, data_root=DATA_ROOT):
    """Load synthetic training data for a specific agent"""
    data_path = os.path.join(data_root, function_type, f"agent{agent_idx+1}.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Synthetic data not found: {data_path}")

    df = pd.read_csv(data_path)
    x_cols = [col for col in df.columns if col.startswith('x')]
    x = df[x_cols].values

    if 'y' in df.columns:
        y = df['y'].values.reshape(-1, 1)  # Should be noisy training data
        print(f"  Agent {agent_idx+1}: {len(x)} total training samples (noisy)")
    elif 'y_noisy' in df.columns:
        y = df['y_noisy'].values.reshape(-1, 1)  # Explicitly noisy data
        print(f"  Agent {agent_idx+1}: {len(x)} total training samples (explicitly noisy)")
    else:
        y = df['y_true'].values.reshape(-1, 1)  # Fallback (unrealistic)
        print(f"  Agent {agent_idx+1}: {len(x)} total training samples (WARNING: using true values)")

    print(f"    - X range: [{x[:, 0].min():.2f}, {x[:, 0].max():.2f}] × [{x[:, 1].min():.2f}, {x[:, 1].max():.2f}]")
    print(f"    - Y range: [{y.min():.2f}, {y.max():.2f}] (training observations)")

    return {'x': x, 'y': y}

def load_synthetic_inducing_points(function_type, data_root=DATA_ROOT):
    """Load synthetic inducing points"""
    inducing_path = os.path.join(data_root, function_type, "inducing.csv")
    if not os.path.exists(inducing_path):
        raise FileNotFoundError(f"Inducing points not found: {inducing_path}")

    inducing_df = pd.read_csv(inducing_path)
    inducing_x_cols = [col for col in inducing_df.columns if col.startswith('x')]
    inducing_points = inducing_df[inducing_x_cols].values

    print(f"  Inducing points: {inducing_points.shape[0]} points")
    return inducing_points