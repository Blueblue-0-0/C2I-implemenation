import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from agent import Agent
from dac import DACConsensus
import matplotlib.cm as cm

NUM_AGENTS = 4
INIT_TRAIN_SIZE = 500
DEVICE = 'cpu'
CONSENSUS_STEPS = 5

# Load agent data
def load_agent_data(i):
    df = pd.read_csv(f'project/data/KIN40K_train_agent{i+1}.csv')
    x_cols = [col for col in df.columns if col.startswith('x')]
    x = df[x_cols].values
    y = df['y'].values.reshape(-1, 1)
    return {'x': x, 'y': y}

agent_data = [load_agent_data(i) for i in range(NUM_AGENTS)]

# Load inducing points and their true y
inducing_df = pd.read_csv('project/data/KIN40K_inducing_all_agents.csv')
inducing_x_cols = [col for col in inducing_df.columns if col.startswith('x')]
inducing_points = inducing_df[inducing_x_cols].values  # shape (32, D)
inducing_y = inducing_df['y'].values  # shape (32,)
inducing_agent_idx = inducing_df['agent_idx'].values  # shape (32,)

# Prepare to store mean histories for all agents
mean_history = [[] for _ in range(NUM_AGENTS)]
true_y_per_agent = []

# Get the true y for each agent's inducing points
for agent_idx in range(NUM_AGENTS):
    agent_inducing_mask = (inducing_agent_idx == agent_idx)
    agent_inducing_y = inducing_y[agent_inducing_mask]
    true_y_per_agent.append(agent_inducing_y)

# Initialize agents
agents = []
for i in range(NUM_AGENTS):
    train_x = torch.tensor(agent_data[i]['x'][:INIT_TRAIN_SIZE], dtype=torch.float32)
    train_y = torch.tensor(agent_data[i]['y'][:INIT_TRAIN_SIZE], dtype=torch.float32)
    neighbors = [(i-1)%NUM_AGENTS, (i+1)%NUM_AGENTS]
    agent = Agent(
        agent_id=i,
        inducing_points=torch.tensor(inducing_points, dtype=torch.float32),
        train_x=train_x,
        train_y=train_y,
        neighbors=neighbors,
        buffer_size=INIT_TRAIN_SIZE,
        device=DEVICE
    )
    agent.train_local(num_iter=150)
    agents.append(agent)

# Set up DAC consensus (ring topology)
A = np.zeros((NUM_AGENTS, NUM_AGENTS))
for i in range(NUM_AGENTS):
    A[i, (i-1)%NUM_AGENTS] = 1
    A[i, (i+1)%NUM_AGENTS] = 1
D = np.diag(A.sum(axis=1))
L = D - A
dac = DACConsensus(L, alpha=0.2)

# Run DAC and record mean history
for step in range(CONSENSUS_STEPS):
    means = []
    vars = []
    for agent in agents:
        var_dist = agent.model.variational_strategy._variational_distribution
        means.append(var_dist.variational_mean.detach().cpu().numpy())
        chol = var_dist.chol_variational_covar.detach().cpu().numpy()
        covar = chol @ chol.T
        
        vars.append(np.diag(covar))
    means = np.stack(means)
    vars = np.stack(vars)

    # Record all 32 mean values for each agent
    for i in range(NUM_AGENTS):
        mean_history[i].append(means[i].copy())

    # DAC consensus step for means and variances
    dac.reset(means)
    for _ in range(1):
        means = dac.step(means)
    dac.reset(vars)
    for _ in range(1):
        vars = dac.step(vars)

    # Inject consensus back into agents
    for i, agent in enumerate(agents):
        agent.consensus_mean = means[i]
        agent.consensus_var = vars[i]
        agent.inject_consensus_to_variational()

# Convert mean_history to numpy arrays for easier plotting
mean_history = [np.stack(agent_means) for agent_means in mean_history]  # shape: (NUM_AGENTS, CONSENSUS_STEPS, 32)

# Plot: Each agent, each of 32 mean values as a line, true y as dashed lines
for agent_idx in range(NUM_AGENTS):
    plt.figure(figsize=(12, 7))
    n_inducing = mean_history[agent_idx].shape[1]  # should be 32
    colors = cm.get_cmap('tab20', n_inducing)  # or 'tab20', 'tab10', etc.

    for j in range(n_inducing):
        color = colors(j)
        # Plot mean values as solid line
        plt.plot(mean_history[agent_idx][:, j], color=color, label=f'Inducing {j+1}')
        # Plot true y as dashed line in the same color
        plt.axhline(inducing_y[j], color=color, linestyle='--', linewidth=1)

    plt.xlabel('DAC Step')
    plt.ylabel('Variational Mean Value')
    plt.title(f'Agent {agent_idx+1}: Evolution of 32 Variational Means (with True y)')
    plt.tight_layout()
    # Optional: show only a subset of labels to avoid clutter
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:8], labels[:8], title="Inducing Points", loc='best', fontsize='small')
    plt.savefig(f'project/train_record/variational_mean_evolution_agent_{agent_idx+1}.png')
    plt.close()