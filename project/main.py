import threading
import numpy as np
import torch
from agent import Agent
from dac import DACConsensus
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import csv
import matplotlib.cm as cm

NUM_AGENTS = 4
BUFFER_SIZE = 500
INIT_TRAIN_SIZE = 1000
INDUCING_PER_AGENT = 25
INDUCING_TOTAL = NUM_AGENTS * INDUCING_PER_AGENT
DEVICE = 'cpu'

# 1. Load separated data for each agent (from CSV)
def load_agent_data(i):
    df = pd.read_csv(f'project/data/KIN40K_train_agent{i+1}.csv')
    x_cols = [col for col in df.columns if col.startswith('x')]
    x = df[x_cols].values
    y = df['y'].values.reshape(-1, 1)
    return {'x': x, 'y': y}

agent_data = [load_agent_data(i) for i in range(NUM_AGENTS)]

# 2. Load pre-saved inducing points and true mean (from CSV)
inducing_df = pd.read_csv('project/data/KIN40K_inducing_all_agents.csv')
inducing_x_cols = [col for col in inducing_df.columns if col.startswith('x')]
inducing_points = inducing_df[inducing_x_cols].values  # shape (32, D)
inducing_true_mean = inducing_df['y'].values  # shape (32,)
inducing_agent_idx = inducing_df['agent_idx'].values  # shape (32,)

# 3. Initialize agents with first 500 samples
agents = []
train_threads = []

for i in range(NUM_AGENTS):
    train_x = torch.tensor(agent_data[i]['x'][:INIT_TRAIN_SIZE], dtype=torch.float32)
    train_y = torch.tensor(agent_data[i]['y'][:INIT_TRAIN_SIZE], dtype=torch.float32)
    neighbors = [(i-1)%NUM_AGENTS, (i+1)%NUM_AGENTS]  # ring topology
    agent = Agent(
        agent_id=i,
        inducing_points=torch.tensor(inducing_points, dtype=torch.float32),
        train_x=train_x,
        train_y=train_y,
        neighbors=neighbors,
        buffer_size=BUFFER_SIZE,
        device=DEVICE
    )
    agents.append(agent)
    # Start training in a separate thread
    def train_agent(agent, idx):
        import time
        start_time = time.time()
        agent.train_local(num_iter=100)
        agent.ready_for_dac.set()
        end_time = time.time()
        print(f"Agent {idx} initial training took {end_time - start_time:.2f} seconds")
    t = threading.Thread(target=train_agent, args=(agent, i))
    t.start()
    train_threads.append(t)

# Wait for all training threads to finish before proceeding
for t in train_threads:
    t.join()

for i, agent in enumerate(agents):
    plt.figure()
    plt.plot(agent.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Negative ELBO Loss')
    plt.title(f"Initial Training Loss Curve -  agent {i}")
    save_dir = "project/train_record"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"Training_Loss_Curve_agent_{i}.png")
    plt.savefig(save_path)
    plt.close()

print("Initial training completed.")

# 4. Set up Laplacian for ring topology and DACConsensus
A = np.zeros((NUM_AGENTS, NUM_AGENTS))
for i in range(NUM_AGENTS):
    A[i, (i-1)%NUM_AGENTS] = 1
    A[i, (i+1)%NUM_AGENTS] = 1
D = np.diag(A.sum(axis=1))
L = D - A
dac = DACConsensus(L, alpha=0.2)

# 5. DAC thread function with detailed logging
dac_plot_event = threading.Event()
mean_history = [[] for _ in range(NUM_AGENTS)]  # List of lists for each agent

def dac_thread_func(agents, dac, consensus_steps=5, plot_every=10):
    global mean_history
    step = 0
    csv_path = "project/train_record/variational_stats.csv"
    header_written = False

    print("[DAC Thread] Started.")
    while step < 5:
        print(f"[DAC Thread] Step {step} collecting means and variances...")
        means, vars = [], []
        for agent in agents:
            with agent.lock:
                var_dist = agent.model.variational_strategy._variational_distribution
                mean = var_dist.variational_mean.detach().cpu().numpy()
                chol = var_dist.chol_variational_covar.detach().cpu().numpy()
                covar = chol @ chol.T
                vars.append(np.diag(covar))
                means.append(mean)
        means = np.stack(means)
        vars = np.stack(vars)

        print(f"[DAC Thread] Step {step} running DAC consensus.")
        # DAC consensus step
        dac.reset(means)
        for _ in range(consensus_steps):
            means = dac.step(means)
        dac.reset(vars)
        for _ in range(consensus_steps):
            vars = dac.step(vars)

        # Record the CONSENSUS values for plotting
        for i in range(len(agents)):
            mean_history[i].append(means[i].copy())  # ← Record AFTER consensus

        print(f"[DAC Thread] Step {step} saving to CSV and updating mean_history.")
        # Save consensus values to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not header_written:
                header = ['step', 'agent_id'] + [f'mean_{i}' for i in range(means.shape[1])] + [f'var_{i}' for i in range(vars.shape[1])]
                writer.writerow(header)
                header_written = True
            for agent_id in range(len(agents)):
                row = [step, agent_id] + list(means[agent_id]) + list(vars[agent_id])
                writer.writerow(row)

        # Inject consensus for NEXT iteration
        for i, agent in enumerate(agents):
            with agent.lock:
                agent.consensus_mean = means[i]
                agent.consensus_var = vars[i]
                agent.inject_consensus_to_variational()
                agent.ready_for_dac.clear()

        print(f"[DAC Thread] Step {step} completed.")
        step += 1
        if step % plot_every == 0:
            dac_plot_event.set()
    
    # After 5 steps, generate final plots and CSVs
    print("[DAC Thread] Completed 5 DAC steps. Generating final plots and CSVs...")
    save_variational_mean_evolution_csv(mean_history, inducing_true_mean)
    plot_variational_mean_evolution(mean_history, inducing_true_mean)
    print("[DAC Thread] DAC thread finished.")

# 6. Start DAC thread
dac_thread = threading.Thread(target=dac_thread_func, args=(agents, dac), daemon=True)
dac_thread.start()

def plot_variational_mean_evolution(mean_history, inducing_true_mean, save_dir="project/train_record"):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import os

    os.makedirs(save_dir, exist_ok=True)
    NUM_AGENTS = len(mean_history)
    for agent_idx in range(NUM_AGENTS):
        plt.figure(figsize=(12, 7))
        n_inducing = mean_history[agent_idx][0].shape[0]
        colors = plt.colormaps['tab20'].resampled(n_inducing)
        for j in range(n_inducing):
            color = colors(j)
            plt.plot([mh[j] for mh in mean_history[agent_idx]], color=color, label=f'Inducing {j+1}')
            plt.axhline(inducing_true_mean[j], color=color, linestyle='--', linewidth=1)
        plt.xlabel('DAC Step')
        plt.ylabel('Variational Mean Value')
        plt.title(f'Agent {agent_idx+1}: Evolution of {n_inducing} Variational Means (with True y)')
        plt.tight_layout()
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:8], labels[:8], title="Inducing Points", loc='best', fontsize='small')
        plt.savefig(os.path.join(save_dir, f'variational_mean_evolution_agent_{agent_idx+1}.png'))
        plt.close()

def save_variational_mean_evolution_csv(mean_history, inducing_true_mean, save_dir="project/train_record"):
    import csv
    import os

    os.makedirs(save_dir, exist_ok=True)
    NUM_AGENTS = len(mean_history)
    for agent_idx in range(NUM_AGENTS):
        csv_path = os.path.join(save_dir, f'variational_mean_evolution_agent_{agent_idx+1}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            n_inducing = mean_history[agent_idx][0].shape[0]
            header = ['step'] + [f'mean_{j}' for j in range(n_inducing)] + [f'true_mean_{j}' for j in range(n_inducing)]
            writer.writerow(header)
            for step, mh in enumerate(mean_history[agent_idx]):
                row = [step] + list(mh) + list(inducing_true_mean[:n_inducing])
                writer.writerow(row)

# Main thread: periodically check for plot event and plot if set
while True:
    if dac_plot_event.is_set():
        print("[Main Thread] Plot event set. Saving variational mean evolution CSVs...")
        save_variational_mean_evolution_csv(mean_history, inducing_true_mean)
        print("[Main Thread] CSVs saved.")
        dac_plot_event.clear()
    time.sleep(1)

# 7. Online learning thread for each agent (simulate streaming)
def agent_online_thread(agent, agent_data, start_idx=INIT_TRAIN_SIZE):
    x_stream = agent_data['x'][start_idx:]
    y_stream = agent_data['y'][start_idx:]
    idx = 0
    print(f"[Agent Thread {agent.agent_id}] Started streaming from index {start_idx}.")
    while idx < len(x_stream):
        new_x = torch.tensor(x_stream[idx:idx+1], dtype=torch.float32)
        new_y = torch.tensor(y_stream[idx:idx+1], dtype=torch.float32)
        with agent.lock:
            agent.update_data(new_x, new_y)
            if agent.train_x.shape[0] % agent.buffer_size == 0:
                print(f"[Agent Thread {agent.agent_id}] Buffer full at {agent.train_x.shape[0]} samples. Retraining...")
                agent.train_local(num_iter=50)
                agent.ready_for_dac.set()
                print(f"[Agent Thread {agent.agent_id}] Retraining done and ready_for_dac set.")
        idx += 1
        if idx % 100 == 0:
            print(f"[Agent Thread {agent.agent_id}] Processed {idx} streaming samples.")

# 8. Start online learning threads
agent_threads = []
for i, agent in enumerate(agents):
    t = threading.Thread(target=agent_online_thread, args=(agent, agent_data[i]), daemon=True)
    t.start()
    agent_threads.append(t)

# 9. (Optional) Wait for threads to finish or run for a set time
# for t in agent_threads:
#     t.join()

for i, agent in enumerate(agents):
    plt.figure()
    plt.plot(agent.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Negative ELBO Loss')
    plt.title(f"Training Loss Curve - agent {i}")
    save_dir = "project/train_record"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"Training_Loss_Curve_agent_{i}.png")
    plt.savefig(save_path)
    plt.close()

