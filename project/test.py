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

# Create validation folder
validation_dir = 'project/train_record/test'
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
DISTRIBUTED GP CONSENSUS EXPERIMENT LOG
{'='*80}
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Configuration:
  - Number of Agents: {NUM_AGENTS}
  - Initial Training Size: {INIT_TRAIN_SIZE}
  - Additional Data Size: {ADDITIONAL_DATA_SIZE}
  - Training Iterations: {NUM_ITER}
  - Consensus Steps: {CONSENSUS_STEPS}
  - Device: {DEVICE}
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
EXPERIMENT COMPLETED
End Time: {end_time}
Total Runtime: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)
{'='*80}
"""
        self.log_file.write(footer)
        self.log_file.close()

# Setup logging
log_file_path = f'{validation_dir}/experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
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

print(f"DATA SUMMARY:")
print(f"  - Total inducing points: {inducing_points.shape[0]}")
print(f"  - Initial training size per agent: {INIT_TRAIN_SIZE}")
print(f"  - Additional data size per agent: {ADDITIONAL_DATA_SIZE}")
print(f"  - Training iterations per agent: {NUM_ITER}")

# Prepare to store mean histories for all phases
mean_history_initial = [[] for _ in range(NUM_AGENTS)]
mean_history_validation = [[] for _ in range(NUM_AGENTS)]
hyperparameters_history = []

# Get the true y for each agent's inducing points
print(f"\nAGENT-SPECIFIC INDUCING POINTS:")
for agent_idx in range(NUM_AGENTS):
    agent_inducing_mask = (inducing_agent_idx == agent_idx)
    agent_inducing_y = inducing_y[agent_inducing_mask]
    print(f"  - Agent {agent_idx+1}: {len(agent_inducing_y)} inducing points")

# ============================================================================
# PHASE 1: INITIAL TRAINING WITH 500 DATA POINTS
# ============================================================================
print("\n" + "=" * 60)
print("PHASE 1: INITIAL TRAINING AND DAC")
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
    except Exception as e:
        print(f"⚠️  Warning: Could not extract variational parameters for agent {agent.id}: {e}")
    
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

@time_function("Initial DAC Consensus")
def run_initial_dac():
    print(f"\nRUNNING initial DAC consensus...")
    for step in range(CONSENSUS_STEPS):
        step_start_time = time.time()
        print(f"\nINITIAL DAC Step {step+1}/{CONSENSUS_STEPS}...")
        
        means = []
        vars = []
        precisions = []
        
        for agent in agents:
            var_dist = agent.model.variational_strategy._variational_distribution
            mean = var_dist.variational_mean.detach().cpu().numpy()
            chol = var_dist.chol_variational_covar.detach().cpu().numpy()
            covar = chol @ chol.T
            var = np.diag(covar)
            precision = 1.0 / (var + 1e-6)
            
            means.append(mean)
            vars.append(var)
            precisions.append(precision)
        
        means = np.stack(means)
        vars = np.stack(vars)
        precisions = np.stack(precisions)

        # Record BEFORE consensus
        for i in range(NUM_AGENTS):
            mean_history_initial[i].append(means[i].copy())

        # Precision-weighted consensus
        weighted_means = means * precisions
        
        dac.reset(weighted_means)
        for _ in range(1):
            weighted_means = dac.step(weighted_means)
        
        dac.reset(precisions)
        for _ in range(1):
            precisions = dac.step(precisions)
        
        # Recover consensus parameters
        consensus_means = weighted_means / (precisions + 1e-6)
        consensus_vars = 1.0 / (precisions + 1e-6)

        # Inject consensus back into agents
        for i, agent in enumerate(agents):
            agent.consensus_mean = consensus_means[i]
            agent.consensus_var = consensus_vars[i]
            agent.inject_consensus_to_variational()
        
        step_time = time.time() - step_start_time
        print(f"DAC Step {step+1} completed in {step_time:.2f}s")

run_initial_dac()

print(f"\nSTORING initial hyperparameters (post-DAC)...")
for agent in agents:
    hyperparameters_history.append(extract_hyperparameters(agent, 'initial', 'post_dac'))

# Convert initial mean_history to numpy arrays
mean_history_initial = [np.stack(agent_means) for agent_means in mean_history_initial]
print(f"Initial phase data converted to numpy arrays")

# ============================================================================
# PHASE 2: ADD NEW DATA, RETRAIN, AND RUN DAC AGAIN
# ============================================================================
print("\n" + "=" * 60)
print("PHASE 2: VALIDATION WITH ADDITIONAL DATA")
print("=" * 60)

@time_function("Data Addition and Validation Training")
def add_data_and_retrain():
    print(f"\nADDING new data to each agent...")
    
    for i, agent in enumerate(agents):
        agent_start_time = time.time()
        print(f"\nPROCESSING Agent {i}...")
        
        # Get additional data
        start_idx = INIT_TRAIN_SIZE
        end_idx = INIT_TRAIN_SIZE + ADDITIONAL_DATA_SIZE
        
        if end_idx <= len(agent_data[i]['x']):
            new_x = torch.tensor(agent_data[i]['x'][start_idx:end_idx], dtype=torch.float32)
            new_y_raw = agent_data[i]['y'][start_idx:end_idx]
            new_y = torch.tensor(new_y_raw, dtype=torch.float32)
            
            print(f"Agent {i} data shapes:")
            print(f"  - Current train_x: {agent.train_x.shape}")
            print(f"  - Current train_y: {agent.train_y.shape}")
            print(f"  - New x: {new_x.shape}")
            print(f"  - New y: {new_y.shape}")
            
            # Handle shape compatibility for concatenation
            if agent.train_y.dim() == 2 and new_y.dim() == 2:
                agent.train_y = torch.cat([agent.train_y, new_y], dim=0)
            elif agent.train_y.dim() == 1 and new_y.dim() == 2:
                agent.train_y = torch.cat([agent.train_y, new_y.flatten()], dim=0)
            elif agent.train_y.dim() == 2 and new_y.dim() == 1:
                new_y_reshaped = new_y.reshape(-1, 1)
                agent.train_y = torch.cat([agent.train_y, new_y_reshaped], dim=0)
            else:
                agent.train_y = torch.cat([agent.train_y, new_y.flatten()], dim=0)
            
            agent.train_x = torch.cat([agent.train_x, new_x], dim=0)
            agent.buffer_size = len(agent.train_x)
            
            print(f"Agent {i}: Added {len(new_x)} new training points")
            print(f"  - Final shapes - X: {agent.train_x.shape}, Y: {agent.train_y.shape}")
            print(f"  - Total training points: {len(agent.train_x)}")
            
        else:
            print(f"Agent {i}: Not enough data available for validation")
        
        agent_data_time = time.time() - agent_start_time
        print(f"Agent {i} data processing: {agent_data_time:.2f}s")

    # Store pre-training hyperparameters
    print(f"\nSTORING pre-validation training hyperparameters...")
    for agent in agents:
        hyperparameters_history.append(extract_hyperparameters(agent, 'validation', 'pre_training'))

    # Retrain agents
    print(f"\nRETRAINING agents with additional data...")
    for i, agent in enumerate(agents):
        training_start = time.time()
        print(f"RETRAINING Agent {i} with {len(agent.train_x)} total data points using {NUM_ITER} iterations...")
        agent.train_local(num_iter=NUM_ITER)
        training_time = time.time() - training_start
        print(f"Agent {i} retraining completed in {training_time:.2f}s")

    # Store post-training hyperparameters
    print(f"\nSTORING post-validation training hyperparameters (pre-DAC)...")
    for agent in agents:
        hyperparameters_history.append(extract_hyperparameters(agent, 'validation', 'post_training_pre_dac'))

add_data_and_retrain()

@time_function("Validation DAC Consensus")
def run_validation_dac():
    print(f"\nRUNNING validation DAC consensus...")
    for step in range(CONSENSUS_STEPS):
        step_start_time = time.time()
        print(f"\nVALIDATION DAC Step {step+1}/{CONSENSUS_STEPS}...")
        
        means = []
        vars = []
        precisions = []
        
        for agent in agents:
            var_dist = agent.model.variational_strategy._variational_distribution
            mean = var_dist.variational_mean.detach().cpu().numpy()
            chol = var_dist.chol_variational_covar.detach().cpu().numpy()
            covar = chol @ chol.T
            var = np.diag(covar)
            precision = 1.0 / (var + 1e-6)
            
            means.append(mean)
            vars.append(var)
            precisions.append(precision)
        
        means = np.stack(means)
        vars = np.stack(vars)
        precisions = np.stack(precisions)

        # Record BEFORE consensus
        for i in range(NUM_AGENTS):
            mean_history_validation[i].append(means[i].copy())

        # Precision-weighted consensus
        weighted_means = means * precisions
        
        dac.reset(weighted_means)
        for _ in range(1):
            weighted_means = dac.step(weighted_means)
        
        dac.reset(precisions)
        for _ in range(1):
            precisions = dac.step(precisions)
        
        # Recover consensus parameters
        consensus_means = weighted_means / (precisions + 1e-6)
        consensus_vars = 1.0 / (precisions + 1e-6)

        # Inject consensus back into agents
        for i, agent in enumerate(agents):
            agent.consensus_mean = consensus_means[i]
            agent.consensus_var = consensus_vars[i]
            agent.inject_consensus_to_variational()
        
        step_time = time.time() - step_start_time
        print(f"Validation DAC Step {step+1} completed in {step_time:.2f}s")

run_validation_dac()

print(f"\nSTORING final hyperparameters (post-DAC)...")
for agent in agents:
    hyperparameters_history.append(extract_hyperparameters(agent, 'validation', 'post_dac'))

# Convert validation mean_history to numpy arrays
mean_history_validation = [np.stack(agent_means) for agent_means in mean_history_validation]
print(f"Validation phase data converted to numpy arrays")

# ============================================================================
# SAVE DATA AND CREATE PLOTS
# ============================================================================
@time_function("Data Saving")
def save_all_data():
    print(f"\nSAVING experiment data...")
    
    # Save hyperparameters
    hyperparameters_df = pd.DataFrame(hyperparameters_history)
    hyperparameters_csv_path = f'{validation_dir}/hyperparameters_history.csv'
    hyperparameters_df.to_csv(hyperparameters_csv_path, index=False)
    print(f"Hyperparameters saved to: {hyperparameters_csv_path}")
    
    # Save mean histories
    for phase_name, mean_history in [('initial', mean_history_initial), ('validation', mean_history_validation)]:
        for agent_idx in range(NUM_AGENTS):
            agent_evolution_data = []
            for step in range(CONSENSUS_STEPS):
                for param_idx in range(len(mean_history[agent_idx][step])):
                    agent_evolution_data.append({
                        'agent_id': agent_idx,
                        'dac_step': step,
                        'parameter_idx': param_idx,
                        'mean_value': mean_history[agent_idx][step][param_idx],
                        'phase': phase_name
                    })
            
            df = pd.DataFrame(agent_evolution_data)
            csv_path = f'{validation_dir}/agent_{agent_idx}_{phase_name}_evolution.csv'
            df.to_csv(csv_path, index=False)
    
    print(f"Evolution data saved for all agents and phases")

save_all_data()

@time_function("Plot Generation")
def create_all_plots():
    print(f"\nCREATING visualization plots...")
    
    # Fix matplotlib deprecation
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    
    plot_count = 0
    
    # Create comparison plots for each agent showing 16 points (4 from each agent region)
    for agent_idx in range(NUM_AGENTS):
        plot_start_time = time.time()
        print(f"Creating plots for Agent {agent_idx+1}...")
        
        # Collect 4 points from each agent's region
        all_selected_indices = []
        point_labels = []
        
        for source_agent_idx in range(NUM_AGENTS):
            # Get inducing points belonging to source_agent_idx
            source_agent_mask = (inducing_agent_idx == source_agent_idx)
            source_agent_indices = np.where(source_agent_mask)[0]
            
            # Select first 4 points from this agent's region
            selected_from_source = source_agent_indices[:POINTS_PER_AGENT]
            
            for i, global_idx in enumerate(selected_from_source):
                all_selected_indices.append(global_idx)
                point_labels.append(f'A{source_agent_idx+1}P{i+1} (#{global_idx})')
        
        print(f"Agent {agent_idx+1}: Plotting {len(all_selected_indices)} points total (4 from each agent region)")
        
        if len(all_selected_indices) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 10))
            
            # Define color gradients for each agent region
            agent_color_schemes = {
                0: ['#000080', '#0000CD', '#4169E1', '#6495ED'],  # Dark blue to light blue
                1: ['#FF4500', '#FF6347', '#FF7F50', '#FFA07A'],  # Dark orange to light orange  
                2: ['#006400', '#228B22', '#32CD32', '#90EE90'],  # Dark green to light green
                3: ['#8B0000', '#DC143C', '#FF1493', '#FF69B4']   # Dark red to light red
            }
            
            # Define marker styles for points within each agent
            marker_styles = ['o', 's', '^', 'D']  # Circle, square, triangle, diamond
            
            # Plot initial phase
            for plot_idx, global_idx in enumerate(all_selected_indices):
                source_agent = plot_idx // 4
                point_in_agent = plot_idx % 4
                
                color = agent_color_schemes[source_agent][point_in_agent]
                marker_style = marker_styles[point_in_agent]
                
                agent_names = ['Agent1', 'Agent2', 'Agent3', 'Agent4']
                point_names = ['P1', 'P2', 'P3', 'P4']
                
                label = f'{agent_names[source_agent]}-{point_names[point_in_agent]} (#{global_idx})'
                
                # Plot initial phase
                ax1.plot(range(CONSENSUS_STEPS), mean_history_initial[agent_idx][:, global_idx], 
                        color=color, marker=marker_style, linewidth=2, markersize=6,
                        label=label, alpha=0.9)
                ax1.axhline(inducing_y[global_idx], color=color, linestyle='--', alpha=0.7, linewidth=1.5)
                
                # Plot validation phase
                ax2.plot(range(CONSENSUS_STEPS), mean_history_validation[agent_idx][:, global_idx], 
                        color=color, marker=marker_style, linewidth=2, markersize=6,
                        label=label, alpha=0.9)
                ax2.axhline(inducing_y[global_idx], color=color, linestyle='--', alpha=0.7, linewidth=1.5)

            # Calculate consistent y-axis limits for both plots
            initial_data = mean_history_initial[agent_idx][:, all_selected_indices]
            validation_data = mean_history_validation[agent_idx][:, all_selected_indices]
            true_values = inducing_y[all_selected_indices]

            # Combine all data to find global min/max
            all_data = np.concatenate([initial_data.flatten(), validation_data.flatten(), true_values])
            y_min = np.min(all_data)
            y_max = np.max(all_data)

            # Add some padding (5% on each side)
            y_range = y_max - y_min
            padding = y_range * 0.05
            y_min_padded = y_min - padding
            y_max_padded = y_max + padding

            # Set identical y-axis limits for both subplots
            ax1.set_ylim(y_min_padded, y_max_padded)
            ax2.set_ylim(y_min_padded, y_max_padded)

            # Format first subplot (Initial phase)
            ax1.set_xlabel('DAC Step', fontsize=13)
            ax1.set_ylabel('Variational Mean Value', fontsize=13)
            ax1.set_title(f'Agent {agent_idx+1}: Initial Training ({INIT_TRAIN_SIZE} samples)\n16 Points: 4 from Each Agent Region', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small', ncol=1)

            # Format second subplot (Validation phase)
            ax2.set_xlabel('DAC Step', fontsize=13)
            ax2.set_ylabel('Variational Mean Value', fontsize=13)
            ax2.set_title(f'Agent {agent_idx+1}: Validation ({INIT_TRAIN_SIZE + ADDITIONAL_DATA_SIZE} samples)\n16 Points: 4 from Each Agent Region', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small', ncol=1)
            
            # Create color legend
            legend_text = """
Color Coding by Agent Region:
• Blues (Dark→Light): Agent 1's region  
• Oranges (Dark→Light): Agent 2's region
• Greens (Dark→Light): Agent 3's region  
• Reds (Dark→Light): Agent 4's region

Markers: ○=P1, □=P2, △=P3, ◇=P4
Dashed lines = True values"""
            
            plt.suptitle(f'Agent {agent_idx+1}: Initial vs Validation DAC Evolution - 16 REPRESENTATIVE POINTS\n{legend_text}', 
                        fontsize=15, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{validation_dir}/agent_{agent_idx+1}_initial_vs_validation_16points_gradient.png', 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            plot_time = time.time() - plot_start_time
            plot_count += 1
            print(f"Agent {agent_idx+1} gradient plot saved with 16 points (4 from each region) ({plot_time:.2f}s)")
    
    print(f"All {plot_count} plots generated successfully")

create_all_plots()

# ============================================================================
# FINAL SUMMARY AND CLEANUP
# ============================================================================
print(f"\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

print(f"All results saved to: {validation_dir}")
print(f"Files created:")
print(f"  - experiment_log_*.txt (this log file)")
print(f"  - hyperparameters_history.csv")
print(f"  - agent_*_initial_evolution.csv")
print(f"  - agent_*_validation_evolution.csv")
print(f"  - agent_*_initial_vs_validation.png")

print(f"\nExperiment completed successfully!")

# Close logger and restore stdout
sys.stdout = logger.terminal
logger.close()

print(f"\nLog file saved to: {log_file_path}")