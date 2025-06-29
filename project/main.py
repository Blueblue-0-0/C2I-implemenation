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

NUM_AGENTS = 4
BUFFER_SIZE = 500
INIT_TRAIN_SIZE = 500
INDUCING_PER_AGENT = 8
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
        agent.train_local(num_iter=50)
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

# 5. DAC thread function with plotting
dac_plot_event = threading.Event()
mean_history = [[] for _ in range(NUM_AGENTS)]  # List of lists for each agent

def dac_thread_func(agents, dac, consensus_steps=5, plot_every=10):
    global mean_history
    step = 0
    while True:
        # # Wait for all agents to be ready
        # for agent in agents:
        #     if not agent.ready_for_dac.wait(timeout=5):
        #         print(f"Timeout: Agent {agent.id} not ready for DAC within 5 seconds. Exiting.")
        #         sys.exit(1)
        # Collect means and variances
        means, vars = [], []
        for agent in agents:
            with agent.lock:
                mean, var = agent.predict_mean_and_var(torch.tensor(agent_data[agent.id]['x'][:INIT_TRAIN_SIZE], dtype=torch.float32))
                means.append(mean)
                vars.append(var)
        means = np.stack(means)
        vars = np.stack(vars)
        dac.reset(means)
        for _ in range(consensus_steps):
            means = dac.step(means)
        dac.reset(vars)
        for _ in range(consensus_steps):
            vars = dac.step(vars)
        # Update agents and record mean
        for i, agent in enumerate(agents):
            with agent.lock:
                agent.consensus_mean = means[i]
                agent.consensus_var = vars[i]
                agent.ready_for_dac.clear()
                mean_history[i].append(np.mean(means[i]))  # Record the mean of the mean vector

        step += 1
        # Plot every 'plot_every' DAC steps
        if step % plot_every == 0:
            dac_plot_event.set()  # Signal main thread to plot

# 6. Start DAC thread
dac_thread = threading.Thread(target=dac_thread_func, args=(agents, dac), daemon=True)
dac_thread.start()

# Main thread: periodically check for plot event and plot if set
while True:
    if dac_plot_event.is_set():
        plt.figure()
        for i, history in enumerate(mean_history):
            plt.plot(history, label=f'Agent {i+1}')
        plt.axhline(np.mean(inducing_true_mean), color='k', linestyle='--', label='True Mean (Inducing)')
        plt.xlabel('DAC Step')
        plt.ylabel('Mean of Consensus Mean')
        plt.title('Consensus Mean Evolution')
        plt.legend()
        save_dir = "project/train_record"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'dac_mean_evolution_step_{len(mean_history[0])}.png'))
        plt.close()
        dac_plot_event.clear()
    time.sleep(1)  # Avoid busy waiting

# 7. Online learning thread for each agent (simulate streaming)
def agent_online_thread(agent, agent_data, start_idx=INIT_TRAIN_SIZE):
    x_stream = agent_data['x'][start_idx:]
    y_stream = agent_data['y'][start_idx:]
    idx = 0
    while idx < len(x_stream):
        # Simulate streaming by adding one sample at a time
        new_x = torch.tensor(x_stream[idx:idx+1], dtype=torch.float32)
        new_y = torch.tensor(y_stream[idx:idx+1], dtype=torch.float32)
        with agent.lock:
            agent.update_data(new_x, new_y)
            # If buffer is full, retrain and signal DAC
            if agent.train_x.shape[0] % agent.buffer_size == 0:
                agent.train_local(num_iter=50)
                agent.ready_for_dac.set()
        idx += 1

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