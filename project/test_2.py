import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from agent import Agent
from dac import DACConsensus
import matplotlib.cm as cm
import os
import sys
import time
from datetime import datetime

NUM_AGENTS = 4
INIT_TRAIN_SIZE = 500
DEVICE = 'cpu'
CONSENSUS_STEPS = 5
NUM_ITER = 500 
# Plot 4 inducing points from each agent's domain
POINTS_PER_AGENT = 4  # Keep this as 4 (points per agent region)
TOTAL_POINTS_TO_PLOT = 16  # 4 agents × 4 points each = 16 total points
ADDITIONAL_DATA_SIZE = 500  # New data to add for validation

# Create validation folder for Full Covariance experiment
validation_dir = 'project/train_record/test_2'
os.makedirs(validation_dir, exist_ok=True)

# ============================================================================
# SETUP LOGGING WITH RUNTIME TRACKING
# ============================================================================
class Logger:
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.start_time = time.time()
        
        # Write header
        self.write_header()
    
    def write_header(self):
        header = f"""
{'='*80}
DISTRIBUTED GP CONSENSUS EXPERIMENT LOG - FULL COVARIANCE MATRIX
{'='*80}
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Configuration:
  - Number of Agents: {NUM_AGENTS}
  - Initial Training Size: {INIT_TRAIN_SIZE}
  - Additional Data Size: {ADDITIONAL_DATA_SIZE}
  - Training Iterations: {NUM_ITER}
  - Consensus Steps: {CONSENSUS_STEPS}
  - Device: {DEVICE}
  - Consensus Type: FULL COVARIANCE MATRIX
{'='*80}

"""
        self.log_file.write(header)
        self.log_file.flush()
    
    def write(self, message):
        # Add timestamp to each line
        current_time = time.time()
        elapsed = current_time - self.start_time
        timestamp = f"[{elapsed:8.2f}s] "
        
        # Split message into lines and add timestamp to each
        lines = str(message).split('\n')
        timestamped_lines = []
        for line in lines:
            if line.strip():  # Only add timestamp to non-empty lines
                timestamped_lines.append(timestamp + line)
            else:
                timestamped_lines.append(line)
        
        timestamped_message = '\n'.join(timestamped_lines)
        
        # Write to both terminal and file
        self.terminal.write(message)
        self.log_file.write(timestamped_message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        # Write footer
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_elapsed = time.time() - self.start_time
        footer = f"""

{'='*80}
EXPERIMENT COMPLETED - FULL COVARIANCE MATRIX
End Time: {end_time}
Total Runtime: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)
{'='*80}
"""
        self.log_file.write(footer)
        self.log_file.close()

# Setup logging
log_file_path = f'{validation_dir}/experiment_log_full_covariance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
logger = Logger(log_file_path)
sys.stdout = logger

# Add timing decorator for functions
def time_function(func_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"\nStarting {func_name}...")
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func_name} completed in {end_time - start_time:.2f} seconds")
            return result
        return wrapper
    return decorator

# ============================================================================
# EXPERIMENT FUNCTIONS WITH TIMING
# ============================================================================

# Load agent data
def load_agent_data(i):
    df = pd.read_csv(f'project/data/KIN40K_train_agent{i+1}.csv')
    x_cols = [col for col in df.columns if col.startswith('x')]
    x = df[x_cols].values
    y = df['y'].values.reshape(-1, 1)
    return {'x': x, 'y': y}

print("Loading agent data and inducing points...")
agent_data = [load_agent_data(i) for i in range(NUM_AGENTS)]

# Load inducing points and their true y
inducing_df = pd.read_csv('project/data/KIN40K_inducing_all_agents.csv')
inducing_x_cols = [col for col in inducing_df.columns if col.startswith('x')]
inducing_points = inducing_df[inducing_x_cols].values  # shape (100, D)
inducing_y = inducing_df['y'].values  # shape (100,)
inducing_agent_idx = inducing_df['agent_idx'].values  # shape (100,)

print(f"DATA SUMMARY - FULL COVARIANCE CONSENSUS:")
print(f"  - Total inducing points: {inducing_points.shape[0]}")
print(f"  - Initial training size per agent: {INIT_TRAIN_SIZE}")
print(f"  - Additional data size per agent: {ADDITIONAL_DATA_SIZE}")
print(f"  - Training iterations per agent: {NUM_ITER}")
print(f"  - Consensus type: FULL COVARIANCE MATRIX")

# Prepare to store mean histories and covariance histories for all phases
mean_history_initial = [[] for _ in range(NUM_AGENTS)]
mean_history_validation = [[] for _ in range(NUM_AGENTS)]
covariance_history_initial = [[] for _ in range(NUM_AGENTS)]  # NEW: Store covariance evolution
covariance_history_validation = [[] for _ in range(NUM_AGENTS)]  # NEW: Store covariance evolution
hyperparameters_history = []

# Get the true y for each agent's inducing points
print(f"\nAGENT-SPECIFIC INDUCING POINTS:")
for agent_idx in range(NUM_AGENTS):
    agent_inducing_mask = (inducing_agent_idx == agent_idx)
    agent_inducing_y = inducing_y[agent_inducing_mask]
    print(f"  - Agent {agent_idx+1}: {len(agent_inducing_y)} inducing points")

# ============================================================================
# ADD NEW METHOD TO AGENT CLASS
# ============================================================================
def inject_full_consensus_to_variational(self):
    """Inject consensus mean and full Cholesky matrix into variational distribution"""
    if hasattr(self, 'consensus_mean') and hasattr(self, 'consensus_chol_matrix'):
        var_dist = self.model.variational_strategy._variational_distribution
        
        # Update variational mean
        var_dist.variational_mean.data = torch.tensor(
            self.consensus_mean, dtype=torch.float32, device=var_dist.variational_mean.device
        )
        
        # Update full Cholesky matrix
        var_dist.chol_variational_covar.data = torch.tensor(
            self.consensus_chol_matrix, dtype=torch.float32, 
            device=var_dist.chol_variational_covar.device
        )
        
        print(f"Agent {self.id}: Updated FULL mean and covariance from consensus")
    else:
        # Fallback to original method
        self.inject_consensus_to_variational()

# Monkey patch the method to Agent class
Agent.inject_full_consensus_to_variational = inject_full_consensus_to_variational

# ============================================================================
# PHASE 1: INITIAL TRAINING WITH 500 DATA POINTS
# ============================================================================
print("\n" + "=" * 60)
print("PHASE 1: INITIAL TRAINING AND FULL COVARIANCE DAC")
print("=" * 60)

@time_function("Agent Initialization and Training")
def initialize_and_train_agents():
    agents = []
    for i in range(NUM_AGENTS):
        agent_start_time = time.time()
        print(f"\nINITIALIZING Agent {i}...")
        
        train_x = torch.tensor(agent_data[i]['x'][:INIT_TRAIN_SIZE], dtype=torch.float32)
        train_y = torch.tensor(agent_data[i]['y'][:INIT_TRAIN_SIZE], dtype=torch.float32)
        neighbors = [(i-1)%NUM_AGENTS, (i+1)%NUM_AGENTS]
        
        agent = Agent(
            agent_id=i,
            inducing_points=torch.tensor(inducing_points, dtype=torch.float32),
            train_x=train_x,
            train_y=train_y,
            neighbors=neighbors,
            buffer_size=INIT_TRAIN_SIZE + ADDITIONAL_DATA_SIZE,
            device=DEVICE
        )
        
        print(f"TRAINING Agent {i} (initial) with {NUM_ITER} iterations...")
        training_start = time.time()
        agent.train_local(num_iter=NUM_ITER)
        training_time = time.time() - training_start
        
        agent_total_time = time.time() - agent_start_time
        print(f"Agent {i} training completed in {training_time:.2f}s (total: {agent_total_time:.2f}s)")
        
        agents.append(agent)
    return agents

agents = initialize_and_train_agents()

# Store initial hyperparameters
def extract_hyperparameters(agent, phase, step_type):
    """Extract hyperparameters from agent"""
    hyper_data = {
        'agent_id': agent.id,
        'phase': phase,
        'step_type': step_type,
        'timestamp': time.time(),
    }
    
    # Get saved hyperparameters
    if hasattr(agent, 'hypers') and agent.hypers is not None:
        for key, value in agent.hypers.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    hyper_data[key] = value.item()
                else:
                    hyper_data[key] = str(value.detach().cpu().numpy())
            else:
                hyper_data[key] = value
    
    # Get model parameters
    try:
        hyper_data['noise'] = agent.likelihood.noise.item()
    except:
        hyper_data['noise'] = None
    
    # Get variational statistics
    try:
        var_dist = agent.model.variational_strategy._variational_distribution
        var_mean = var_dist.variational_mean.detach().cpu().numpy()
        chol = var_dist.chol_variational_covar.detach().cpu().numpy()
        
        hyper_data['var_mean_min'] = var_mean.min()
        hyper_data['var_mean_max'] = var_mean.max()
        hyper_data['var_mean_std'] = var_mean.std()
        hyper_data['chol_diag_min'] = np.diag(chol).min()
        hyper_data['chol_diag_max'] = np.diag(chol).max()
        
        # NEW: Add covariance matrix statistics
        covar = chol @ chol.T
        hyper_data['covar_trace'] = np.trace(covar)
        hyper_data['covar_det'] = np.linalg.det(covar)
        hyper_data['covar_frobenius_norm'] = np.linalg.norm(covar, 'fro')
        
    except Exception as e:
        print(f"Warning: Could not extract variational parameters for agent {agent.id}: {e}")
    
    return hyper_data

print(f"\nSTORING initial hyperparameters (pre-DAC)...")
for agent in agents:
    hyperparameters_history.append(extract_hyperparameters(agent, 'initial', 'pre_dac'))

# Set up DAC consensus (ring topology)
print(f"\nSETTING UP DAC consensus (ring topology)...")
A = np.zeros((NUM_AGENTS, NUM_AGENTS))
for i in range(NUM_AGENTS):
    A[i, (i-1)%NUM_AGENTS] = 1
    A[i, (i+1)%NUM_AGENTS] = 1
D = np.diag(A.sum(axis=1))
L = D - A
dac = DACConsensus(L, alpha=0.2)
print(f"DAC consensus matrix configured (alpha=0.2)")

@time_function("Initial DAC Consensus with Full Covariance")
def run_initial_dac():
    print(f"\nRUNNING initial DAC consensus with FULL covariance matrix...")
    for step in range(CONSENSUS_STEPS):
        step_start_time = time.time()
        print(f"\nINITIAL DAC Step {step+1}/{CONSENSUS_STEPS}...")
        
        means = []
        chol_matrices = []  # Store full Cholesky matrices
        
        for agent in agents:
            var_dist = agent.model.variational_strategy._variational_distribution
            mean = var_dist.variational_mean.detach().cpu().numpy()
            chol = var_dist.chol_variational_covar.detach().cpu().numpy()
            
            means.append(mean)
            chol_matrices.append(chol)
        
        means = np.stack(means)  # Shape: (num_agents, num_inducing)
        chol_matrices = np.stack(chol_matrices)  # Shape: (num_agents, num_inducing, num_inducing)

        # Record BEFORE consensus
        for i in range(NUM_AGENTS):
            mean_history_initial[i].append(means[i].copy())
            # NEW: Store covariance matrices
            covar = chol_matrices[i] @ chol_matrices[i].T
            covariance_history_initial[i].append(covar.copy())

        # 1. CONSENSUS ON MEANS (precision-weighted)
        precisions = []
        for i in range(NUM_AGENTS):
            covar = chol_matrices[i] @ chol_matrices[i].T
            var = np.diag(covar)
            precision = 1.0 / (var + 1e-6)
            precisions.append(precision)
        
        precisions = np.stack(precisions)
        weighted_means = means * precisions
        
        # Apply DAC to weighted means
        dac.reset(weighted_means)
        for _ in range(1):
            weighted_means = dac.step(weighted_means)
        
        # Apply DAC to precisions
        dac.reset(precisions)
        for _ in range(1):
            precisions = dac.step(precisions)
        
        consensus_means = weighted_means / (precisions + 1e-6)

        # 2. CONSENSUS ON EVERY ELEMENT OF CHOLESKY MATRIX
        consensus_chol_matrices = np.zeros_like(chol_matrices)
        n_inducing = chol_matrices.shape[1]
        
        print(f"  Applying consensus to {n_inducing}×{n_inducing} covariance elements...")
        
        # Count operations for progress tracking
        total_elements = 0
        processed_elements = 0
        
        for i in range(n_inducing):
            for j in range(n_inducing):
                if j <= i:  # Only lower triangular (Cholesky property)
                    total_elements += 1
        
        for i in range(n_inducing):
            for j in range(n_inducing):
                if j <= i:  # Only lower triangular (Cholesky property)
                    # Extract element (i,j) from all agents
                    element_values = chol_matrices[:, i, j]  # Shape: (num_agents,)
                    
                    # Apply consensus to this element
                    dac.reset(element_values.reshape(-1, 1))  # DAC expects 2D
                    for _ in range(1):
                        element_values = dac.step(element_values.reshape(-1, 1)).flatten()
                    
                    # Store consensus element
                    consensus_chol_matrices[:, i, j] = element_values
                    
                    processed_elements += 1
                    if processed_elements % 100 == 0:
                        progress = (processed_elements / total_elements) * 100
                        print(f"    Progress: {processed_elements}/{total_elements} elements ({progress:.1f}%)")
                else:
                    # Upper triangular stays zero (Cholesky property)
                    consensus_chol_matrices[:, i, j] = 0.0

        # 3. ENSURE POSITIVE DEFINITENESS
        for i in range(NUM_AGENTS):
            # Make sure diagonal elements are positive
            diag_indices = np.arange(n_inducing)
            consensus_chol_matrices[i, diag_indices, diag_indices] = np.abs(
                consensus_chol_matrices[i, diag_indices, diag_indices]
            ) + 1e-6

        # 4. INJECT CONSENSUS BACK INTO AGENTS
        for i, agent in enumerate(agents):
            agent.consensus_mean = consensus_means[i]
            agent.consensus_chol_matrix = consensus_chol_matrices[i]
            agent.inject_full_consensus_to_variational()  # Use new method
        
        step_time = time.time() - step_start_time
        print(f"  DAC Step {step+1} completed in {step_time:.2f}s")

run_initial_dac()

print(f"\nSTORING initial hyperparameters (post-DAC)...")
for agent in agents:
    hyperparameters_history.append(extract_hyperparameters(agent, 'initial', 'post_dac'))

# Convert initial histories to numpy arrays
mean_history_initial = [np.stack(agent_means) for agent_means in mean_history_initial]
covariance_history_initial = [np.stack(agent_covars) for agent_covars in covariance_history_initial]
print(f"Initial phase data converted to numpy arrays")

# ============================================================================
# PHASE 2: ONLINE LEARNING WITH STREAMING DATA AND FULL COVARIANCE DAC
# ============================================================================
print("\n" + "=" * 60)
print("PHASE 2: ONLINE LEARNING WITH STREAMING DATA AND FULL COVARIANCE DAC")
print("=" * 60)

# Parameters for online learning
ONLINE_BATCH_SIZE = 50  # Process 50 new points at a time
ONLINE_ITERATIONS = 10  # Quick training iterations for each batch
ONLINE_BATCHES = ADDITIONAL_DATA_SIZE // ONLINE_BATCH_SIZE  # Number of batches

print(f"ONLINE LEARNING CONFIGURATION:")
print(f"  - Batch size: {ONLINE_BATCH_SIZE} points per batch")
print(f"  - Training iterations per batch: {ONLINE_ITERATIONS}")
print(f"  - Total batches: {ONLINE_BATCHES}")
print(f"  - Total new data points: {ONLINE_BATCHES * ONLINE_BATCH_SIZE}")

@time_function("Online Learning with Streaming Data")
def online_learning_with_consensus():
    print(f"\nSTARTING online learning with streaming data...")
    
    # Store online learning evolution
    online_mean_history = [[] for _ in range(NUM_AGENTS)]
    online_covariance_history = [[] for _ in range(NUM_AGENTS)]
    
    for batch_idx in range(ONLINE_BATCHES):
        batch_start_time = time.time()
        print(f"\n" + "=" * 40)
        print(f"ONLINE BATCH {batch_idx+1}/{ONLINE_BATCHES}")
        print("=" * 40)
        
        # Add new data batch to each agent
        for i, agent in enumerate(agents):
            # Calculate data indices for this batch
            start_idx = INIT_TRAIN_SIZE + (batch_idx * ONLINE_BATCH_SIZE)
            end_idx = start_idx + ONLINE_BATCH_SIZE
            
            if end_idx <= len(agent_data[i]['x']):
                # Get new batch data
                new_batch_x = torch.tensor(agent_data[i]['x'][start_idx:end_idx], dtype=torch.float32)
                new_batch_y_raw = agent_data[i]['y'][start_idx:end_idx]
                new_batch_y = torch.tensor(new_batch_y_raw, dtype=torch.float32)
                
                # Handle shape compatibility
                if agent.train_y.dim() == 2 and new_batch_y.dim() == 2:
                    agent.train_y = torch.cat([agent.train_y, new_batch_y], dim=0)
                elif agent.train_y.dim() == 1 and new_batch_y.dim() == 2:
                    agent.train_y = torch.cat([agent.train_y, new_batch_y.flatten()], dim=0)
                elif agent.train_y.dim() == 2 and new_batch_y.dim() == 1:
                    new_batch_y_reshaped = new_batch_y.reshape(-1, 1)
                    agent.train_y = torch.cat([agent.train_y, new_batch_y_reshaped], dim=0)
                else:
                    agent.train_y = torch.cat([agent.train_y, new_batch_y.flatten()], dim=0)
                
                agent.train_x = torch.cat([agent.train_x, new_batch_x], dim=0)
                
                print(f"Agent {i}: Added batch {batch_idx+1} ({len(new_batch_x)} points)")
                print(f"  - Total data points: {len(agent.train_x)}")
                
                # Quick online training on new batch
                online_training_start = time.time()
                agent.train_local(num_iter=ONLINE_ITERATIONS)
                online_training_time = time.time() - online_training_start
                print(f"  - Online training: {online_training_time:.2f}s")
        
        # Run DAC consensus after each batch
        print(f"\nRUNNING DAC consensus after batch {batch_idx+1}...")
        
        # Store hyperparameters before consensus
        for agent in agents:
            hyperparameters_history.append(extract_hyperparameters(agent, 'online', f'batch_{batch_idx+1}_pre_dac'))
        
        # Run full covariance consensus
        for step in range(CONSENSUS_STEPS):
            step_start_time = time.time()
            print(f"  Online DAC Step {step+1}/{CONSENSUS_STEPS}...")
            
            means = []
            chol_matrices = []
            
            for agent in agents:
                var_dist = agent.model.variational_strategy._variational_distribution
                mean = var_dist.variational_mean.detach().cpu().numpy()
                chol = var_dist.chol_variational_covar.detach().cpu().numpy()
                
                means.append(mean)
                chol_matrices.append(chol)
            
            means = np.stack(means)
            chol_matrices = np.stack(chol_matrices)

            # Record BEFORE consensus (only for first step of each batch)
            if step == 0:
                for i in range(NUM_AGENTS):
                    online_mean_history[i].append(means[i].copy())
                    covar = chol_matrices[i] @ chol_matrices[i].T
                    online_covariance_history[i].append(covar.copy())

            # 1. CONSENSUS ON MEANS (precision-weighted)
            precisions = []
            for i in range(NUM_AGENTS):
                covar = chol_matrices[i] @ chol_matrices[i].T
                var = np.diag(covar)
                precision = 1.0 / (var + 1e-6)
                precisions.append(precision)
            
            precisions = np.stack(precisions)
            weighted_means = means * precisions
            
            # Apply DAC to weighted means
            dac.reset(weighted_means)
            for _ in range(1):
                weighted_means = dac.step(weighted_means)
            
            # Apply DAC to precisions
            dac.reset(precisions)
            for _ in range(1):
                precisions = dac.step(precisions)
            
            consensus_means = weighted_means / (precisions + 1e-6)

            # 2. CONSENSUS ON EVERY ELEMENT OF CHOLESKY MATRIX
            consensus_chol_matrices = np.zeros_like(chol_matrices)
            n_inducing = chol_matrices.shape[1]
            
            for i in range(n_inducing):
                for j in range(n_inducing):
                    if j <= i:  # Only lower triangular
                        element_values = chol_matrices[:, i, j]
                        
                        # Apply consensus to this element
                        dac.reset(element_values.reshape(-1, 1))
                        for _ in range(1):
                            element_values = dac.step(element_values.reshape(-1, 1)).flatten()
                        
                        consensus_chol_matrices[:, i, j] = element_values
                    else:
                        consensus_chol_matrices[:, i, j] = 0.0

            # 3. ENSURE POSITIVE DEFINITENESS
            for i in range(NUM_AGENTS):
                diag_indices = np.arange(n_inducing)
                consensus_chol_matrices[i, diag_indices, diag_indices] = np.abs(
                    consensus_chol_matrices[i, diag_indices, diag_indices]
                ) + 1e-6

            # 4. INJECT CONSENSUS BACK INTO AGENTS
            for i, agent in enumerate(agents):
                agent.consensus_mean = consensus_means[i]
                agent.consensus_chol_matrix = consensus_chol_matrices[i]
                agent.inject_full_consensus_to_variational()
            
            step_time = time.time() - step_start_time
            if step % 2 == 0:  # Print every 2nd step to reduce output
                print(f"    Step {step+1} completed in {step_time:.2f}s")
        
        # Store hyperparameters after consensus
        for agent in agents:
            hyperparameters_history.append(extract_hyperparameters(agent, 'online', f'batch_{batch_idx+1}_post_dac'))
        
        batch_time = time.time() - batch_start_time
        print(f"Batch {batch_idx+1} completed in {batch_time:.2f}s")
    
    # Convert online histories to numpy arrays
    online_mean_history = [np.stack(agent_means) for agent_means in online_mean_history]
    online_covariance_history = [np.stack(agent_covars) for agent_covars in online_covariance_history]
    
    return online_mean_history, online_covariance_history

# Run online learning
mean_history_online, covariance_history_online = online_learning_with_consensus()

print(f"\nSTORING final hyperparameters (post-online learning)...")
for agent in agents:
    hyperparameters_history.append(extract_hyperparameters(agent, 'online', 'final'))

print(f"Online learning phase completed with {ONLINE_BATCHES} batches")

# ============================================================================
# SAVE DATA AND CREATE PLOTS
# ============================================================================
@time_function("Data Saving")
def save_all_data():
    print(f"\nSAVING experiment data...")
    
    # Save hyperparameters
    hyperparameters_df = pd.DataFrame(hyperparameters_history)
    hyperparameters_csv_path = f'{validation_dir}/hyperparameters_history_full_covariance.csv'
    hyperparameters_df.to_csv(hyperparameters_csv_path, index=False)
    print(f"Hyperparameters saved to: {hyperparameters_csv_path}")
    
    # Save mean histories (initial + online)
    for phase_name, mean_history in [('initial', mean_history_initial), ('online', mean_history_online)]:
        for agent_idx in range(NUM_AGENTS):
            agent_evolution_data = []
            for step in range(len(mean_history[agent_idx])):
                for param_idx in range(len(mean_history[agent_idx][step])):
                    agent_evolution_data.append({
                        'agent_id': agent_idx,
                        'batch_or_step': step,
                        'parameter_idx': param_idx,
                        'mean_value': mean_history[agent_idx][step][param_idx],
                        'phase': phase_name
                    })
            
            df = pd.DataFrame(agent_evolution_data)
            csv_path = f'{validation_dir}/agent_{agent_idx}_{phase_name}_evolution_full_covariance.csv'
            df.to_csv(csv_path, index=False)
    
    # Save covariance evolution data (initial + online)
    print(f"Saving covariance evolution data...")
    for phase_name, covar_history in [('initial', covariance_history_initial), ('online', covariance_history_online)]:
        for agent_idx in range(NUM_AGENTS):
            covar_evolution_data = []
            for step in range(len(covar_history[agent_idx])):
                covar_matrix = covar_history[agent_idx][step]
                covar_evolution_data.append({
                    'agent_id': agent_idx,
                    'batch_or_step': step,
                    'phase': phase_name,
                    'trace': np.trace(covar_matrix),
                    'determinant': np.linalg.det(covar_matrix + np.eye(covar_matrix.shape[0]) * 1e-6),
                    'frobenius_norm': np.linalg.norm(covar_matrix, 'fro'),
                    'condition_number': np.linalg.cond(covar_matrix),
                    'min_eigenvalue': np.min(np.linalg.eigvals(covar_matrix)),
                    'max_eigenvalue': np.max(np.linalg.eigvals(covar_matrix))
                })
            
            df = pd.DataFrame(covar_evolution_data)
            csv_path = f'{validation_dir}/agent_{agent_idx}_{phase_name}_covariance_evolution.csv'
            df.to_csv(csv_path, index=False)
    
    print(f"Evolution data saved for all agents and phases")

save_all_data()

# Replace the plotting section to show Initial vs Online Learning:

@time_function("Plot Generation")
def create_all_plots():
    print(f"\nCREATING visualization plots for FULL COVARIANCE consensus...")
    
    import matplotlib
    
    plot_count = 0
    
    # Create comparison plots: Initial DAC vs Online Learning
    for agent_idx in range(NUM_AGENTS):
        plot_start_time = time.time()
        print(f"Creating plots for Agent {agent_idx+1}...")
        
        # Collect 4 points from each agent's region
        all_selected_indices = []
        point_labels = []
        
        for source_agent_idx in range(NUM_AGENTS):
            source_agent_mask = (inducing_agent_idx == source_agent_idx)
            source_agent_indices = np.where(source_agent_mask)[0]
            selected_from_source = source_agent_indices[:POINTS_PER_AGENT]
            
            for i, global_idx in enumerate(selected_from_source):
                all_selected_indices.append(global_idx)
                point_labels.append(f'A{source_agent_idx+1}P{i+1} (#{global_idx})')
        
        print(f"Agent {agent_idx+1}: Plotting {len(all_selected_indices)} points total")
        
        if len(all_selected_indices) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 10))
            
            # Define color gradients
            agent_color_schemes = {
                0: ['#000080', '#0000CD', '#4169E1', '#6495ED'],
                1: ['#FF4500', '#FF6347', '#FF7F50', '#FFA07A'],
                2: ['#006400', '#228B22', '#32CD32', '#90EE90'],
                3: ['#8B0000', '#DC143C', '#FF1493', '#FF69B4']
            }
            
            marker_styles = ['o', 's', '^', 'D']
            
            for plot_idx, global_idx in enumerate(all_selected_indices):
                source_agent = plot_idx // 4
                point_in_agent = plot_idx % 4
                
                color = agent_color_schemes[source_agent][point_in_agent]
                marker_style = marker_styles[point_in_agent]
                
                agent_names = ['Agent1', 'Agent2', 'Agent3', 'Agent4']
                point_names = ['P1', 'P2', 'P3', 'P4']
                
                label = f'{agent_names[source_agent]}-{point_names[point_in_agent]} (#{global_idx})'
                
                # Plot initial phase (DAC steps 0-4)
                ax1.plot(range(CONSENSUS_STEPS), mean_history_initial[agent_idx][:, global_idx], 
                        color=color, marker=marker_style, linewidth=2, markersize=6,
                        label=label, alpha=0.9)
                ax1.axhline(inducing_y[global_idx], color=color, linestyle='--', alpha=0.7, linewidth=1.5)
                
                # Plot online learning phase (batch progression)
                ax2.plot(range(len(mean_history_online[agent_idx])), 
                        mean_history_online[agent_idx][:, global_idx], 
                        color=color, marker=marker_style, linewidth=2, markersize=6,
                        label=label, alpha=0.9)
                ax2.axhline(inducing_y[global_idx], color=color, linestyle='--', alpha=0.7, linewidth=1.5)

            # Calculate consistent y-axis limits
            initial_data = mean_history_initial[agent_idx][:, all_selected_indices]
            online_data = mean_history_online[agent_idx][:, all_selected_indices]
            true_values = inducing_y[all_selected_indices]

            all_data = np.concatenate([initial_data.flatten(), online_data.flatten(), true_values])
            y_min = np.min(all_data)
            y_max = np.max(all_data)
            y_range = y_max - y_min
            padding = y_range * 0.05
            y_min_padded = y_min - padding
            y_max_padded = y_max + padding

            ax1.set_ylim(y_min_padded, y_max_padded)
            ax2.set_ylim(y_min_padded, y_max_padded)

            # Format subplots
            ax1.set_xlabel('DAC Step', fontsize=13)
            ax1.set_ylabel('Variational Mean Value', fontsize=13)
            ax1.set_title(f'Agent {agent_idx+1}: Initial Training ({INIT_TRAIN_SIZE} samples)\nFull Covariance Consensus', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small', ncol=1)
            
            ax2.set_xlabel('Online Batch', fontsize=13)
            ax2.set_ylabel('Variational Mean Value', fontsize=13)
            ax2.set_title(f'Agent {agent_idx+1}: Online Learning ({ONLINE_BATCHES} batches)\nStreaming Data + DAC Consensus', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small', ncol=1)
            
            # Create legend
            legend_text = """
FULL COVARIANCE MATRIX CONSENSUS - ONLINE LEARNING
Color Coding by Agent Region:
• Blues: Agent 1's region  • Oranges: Agent 2's region
• Greens: Agent 3's region • Reds: Agent 4's region
Markers: ○=P1, □=P2, △=P3, ◇=P4, Dashed=True values"""
            
            plt.suptitle(f'Agent {agent_idx+1}: Initial DAC vs Online Learning Evolution\n{legend_text}', 
                        fontsize=15, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{validation_dir}/agent_{agent_idx+1}_initial_vs_online_learning.png', 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            plot_time = time.time() - plot_start_time
            plot_count += 1
            print(f"Agent {agent_idx+1} online learning plot saved ({plot_time:.2f}s)")
    
    print(f"All {plot_count} online learning plots generated successfully")

create_all_plots()

# ============================================================================
# FINAL SUMMARY AND CLEANUP
# ============================================================================
print(f"\n" + "=" * 60)
print("EXPERIMENT SUMMARY - FULL COVARIANCE MATRIX CONSENSUS")
print("=" * 60)

print(f"All results saved to: {validation_dir}")
print(f"Files created:")
print(f"  - experiment_log_full_covariance_*.txt (this log file)")
print(f"  - hyperparameters_history_full_covariance.csv")
print(f"  - agent_*_initial_evolution_full_covariance.csv")
print(f"  - agent_*_validation_evolution_full_covariance.csv")
print(f"  - agent_*_initial_covariance_evolution.csv")  # NEW
print(f"  - agent_*_validation_covariance_evolution.csv")  # NEW
print(f"  - agent_*_full_covariance_16points_gradient.png")

print(f"\nFULL COVARIANCE MATRIX CONSENSUS experiment completed successfully!")

# Close logger and restore stdout
sys.stdout = logger.terminal
logger.close()

print(f"\nFull covariance log file saved to: {log_file_path}")