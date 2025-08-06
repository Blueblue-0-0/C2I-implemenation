import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

# ==== Set number of inducing points per agent here ====
INDUCING_PER_AGENT = 25  # Change this to 8, 25, etc.
# ======================================================

# Load the data
data = loadmat('e:/TUM/RCI-S5-SS25/GP/Practice/dataset/KIN40K/KIN40K_train.mat')
x = data['xtest']  
y = data['ytest'] 

# Compute medians for the first two columns
med0 = np.median(x[:, 0])
med1 = np.median(x[:, 1])

# Assign to 4 agents based on (col0 < med0, col1 < med1)
agent_idx = [
    np.where((x[:, 0] < med0) & (x[:, 1] < med1))[0],  # agent 0: low-low
    np.where((x[:, 0] < med0) & (x[:, 1] >= med1))[0], # agent 1: low-high
    np.where((x[:, 0] >= med0) & (x[:, 1] < med1))[0], # agent 2: high-low
    np.where((x[:, 0] >= med0) & (x[:, 1] >= med1))[0] # agent 3: high-high
]

# Ensure the data directory exists
out_dir = 'e:/TUM/RCI-S5-SS25/GP/Practice/project/data'
os.makedirs(out_dir, exist_ok=True)

# Save agent data as CSV and print format
for i, idx in enumerate(agent_idx):
    agent_x = x[idx]
    agent_y = y[idx]
    df = pd.DataFrame(agent_x, columns=[f'x{j+1}' for j in range(agent_x.shape[1])])
    df['y'] = agent_y
    out_path = os.path.join(out_dir, f"KIN40K_train_agent{i+1}.csv")
    df.to_csv(out_path, index=False)
    print(f"Agent {i+1}: {agent_x.shape[0]} samples saved to {out_path}")
    print(f"  Format: x shape {agent_x.shape}, y shape {agent_y.shape}")

# Collect and save inducing points as CSV and print format
all_inducing_x = []
all_inducing_y = []
agent_indices = []

for i, idx in enumerate(agent_idx):
    agent_x = x[idx]
    agent_y = y[idx]
    n = agent_x.shape[0]
    inducing_indices = np.linspace(0, n-1, INDUCING_PER_AGENT, dtype=int)
    inducing_x = agent_x[inducing_indices]
    inducing_y = agent_y[inducing_indices]
    all_inducing_x.append(inducing_x)
    all_inducing_y.append(inducing_y)
    agent_indices.extend([i]*INDUCING_PER_AGENT)

all_inducing_x = np.vstack(all_inducing_x)  # shape (4*INDUCING_PER_AGENT, D)
all_inducing_y = np.vstack(all_inducing_y)  # shape (4*INDUCING_PER_AGENT, 1)
agent_indices = np.array(agent_indices)     # shape (4*INDUCING_PER_AGENT,)

inducing_df = pd.DataFrame(all_inducing_x, columns=[f'x{j+1}' for j in range(all_inducing_x.shape[1])])
inducing_df['y'] = all_inducing_y
inducing_df['agent_idx'] = agent_indices
out_path = os.path.join(out_dir, "KIN40K_inducing_all_agents.csv")
inducing_df.to_csv(out_path, index=False)
print(f"Inducing points: x shape {all_inducing_x.shape}, y shape {all_inducing_y.shape}, agent_idx shape {agent_indices.shape}")
print(f"Saved all {all_inducing_x.shape[0]} inducing points and their agent indices to {out_path}")