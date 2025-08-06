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
import torch
import numpy as np
import torch
import numpy as np
import torch
import numpy as np
import torch
import torch
import scipy.io

NUM_AGENTS = 4
INIT_TRAIN_SIZE = 500
DEVICE = 'cpu'
CONSENSUS_STEPS = 5
NUM_ITER = 500 
# Plot 4 inducing points from each agent's domain
POINTS_PER_AGENT = 4  # Keep this as 4 (points per agent region)
TOTAL_POINTS_TO_PLOT = 16  # 4 agents × 4 points each = 16 total points
ADDITIONAL_DATA_SIZE = 200  # NEW: Changed to 200 new data points

# Create validation folder for Optimized Inducing Points experiment
validation_dir = 'project/train_record/test_3'
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
DISTRIBUTED GP CONSENSUS EXPERIMENT LOG - OPTIMIZED INDUCING POINTS
{'='*80}
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Configuration:
  - Number of Agents: {NUM_AGENTS}
  - Initial Training Size: {INIT_TRAIN_SIZE}
  - Additional Data Size: {ADDITIONAL_DATA_SIZE}
  - Training Iterations: {NUM_ITER}
  - Consensus Steps: {CONSENSUS_STEPS}
  - Device: {DEVICE}
  - Inducing Points: OPTIMIZED (128 shared locations per agent)
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
EXPERIMENT COMPLETED - OPTIMIZED INDUCING POINTS
End Time: {end_time}
Total Runtime: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)
{'='*80}
"""
        self.log_file.write(footer)
        self.log_file.close()

# Setup logging
log_file_path = f'{validation_dir}/experiment_log_optimized_inducing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
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
# DATA LOADING FUNCTIONS
# ============================================================================

# Load agent data
def load_agent_data(i):
    df = pd.read_csv(f'project/data/KIN40K_train_agent{i+1}.csv')
    x_cols = [col for col in df.columns if col.startswith('x')]
    x = df[x_cols].values
    y = df['y'].values.reshape(-1, 1)
    return {'x': x, 'y': y}

@time_function("Test Data Loading")
def load_test_data():
    """Load test data from KIN40K_test.mat"""
    print("Loading test data from dataset/KIN40K/KIN40K_test.mat...")
    
    try:
        mat_data = scipy.io.loadmat('dataset/KIN40K/KIN40K_test.mat')
        print(f"Available keys in test data: {list(mat_data.keys())}")
        
        # Extract test data (adjust key names if needed)
        if 'pX' in mat_data and 'pY' in mat_data:
            test_x = mat_data['pX']
            test_y = mat_data['pY']
        elif 'X' in mat_data and 'Y' in mat_data:
            test_x = mat_data['X']
            test_y = mat_data['Y']
        else:
            # Try to find the data arrays
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            print(f"Data keys found: {data_keys}")
            if len(data_keys) >= 2:
                test_x = mat_data[data_keys[0]]
                test_y = mat_data[data_keys[1]]
            else:
                raise KeyError("Could not identify test data arrays")
        
        print(f"Test data loaded successfully:")
        print(f"  - Test X shape: {test_x.shape}")
        print(f"  - Test Y shape: {test_y.shape}")
        print(f"  - Test X range: [{test_x.min():.4f}, {test_x.max():.4f}]")
        print(f"  - Test Y range: [{test_y.min():.4f}, {test_y.max():.4f}]")
        
        return torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)
        
    except FileNotFoundError:
        print("ERROR: Test data file not found at dataset/KIN40K/KIN40K_test.mat")
        print("Please ensure the test data file exists and the path is correct.")
        raise
    except Exception as e:
        print(f"ERROR loading test data: {e}")
        raise

print("Loading agent data and optimized inducing points...")
agent_data = [load_agent_data(i) for i in range(NUM_AGENTS)]

# ============================================================================
# LOAD SHARED OPTIMIZED INDUCING POINTS (SAME LOCATIONS FOR ALL AGENTS)
# ============================================================================
print("Loading OPTIMIZED inducing points from project/data/KIN40K_inducing_optimized.csv...")
try:
    inducing_df = pd.read_csv('project/data/KIN40K_inducing_optimized.csv')
    print(f"Optimized inducing points loaded successfully:")
    print(f"  - Shape: {inducing_df.shape}")
    print(f"  - Columns: {list(inducing_df.columns)}")
    
    inducing_x_cols = [col for col in inducing_df.columns if col.startswith('x')]
    
    # NEW: Use ALL 128 inducing points for each agent (shared locations)
    all_inducing_points = inducing_df[inducing_x_cols].values  # shape (128, D)
    all_inducing_y = inducing_df['y'].values  # shape (128,)
    all_inducing_agent_idx = inducing_df['agent_idx'].values  # shape (128,)
    
    # Create shared inducing points - same 128 locations for all agents
    inducing_points = all_inducing_points  # All agents use the same 128 locations
    inducing_y = all_inducing_y  # Corresponding y values
    inducing_agent_idx = all_inducing_agent_idx  # Original agent ownership (for plotting)
    
    print(f"SHARED INDUCING POINTS CONFIGURATION:")
    print(f"  - Each agent uses ALL {len(inducing_points)} inducing points")
    print(f"  - Same inducing locations shared across all agents")
    print(f"  - Original agent regions preserved for plotting purposes")
    
except FileNotFoundError:
    print("ERROR: Optimized inducing points file not found!")
    print("Please run optimize_inducing_points.py first to generate KIN40K_inducing_optimized.csv")
    raise

# Load test data
test_x, test_y = load_test_data()

print(f"DATA SUMMARY - OPTIMIZED SHARED INDUCING POINTS CONSENSUS:")
print(f"  - Total inducing points per agent: {inducing_points.shape[0]} (SHARED LOCATIONS)")
print(f"  - Initial training size per agent: {INIT_TRAIN_SIZE}")
print(f"  - Additional data size per agent: {ADDITIONAL_DATA_SIZE}")
print(f"  - Training iterations per agent: {NUM_ITER}")
print(f"  - Test data size: {len(test_x)}")
print(f"  - Consensus type: FULL COVARIANCE MATRIX")

# Prepare to store evolution data
mean_history_initial = [[] for _ in range(NUM_AGENTS)]
mean_history_validation = [[] for _ in range(NUM_AGENTS)]
covariance_history_initial = [[] for _ in range(NUM_AGENTS)]
covariance_history_validation = [[] for _ in range(NUM_AGENTS)]
hyperparameters_history = []

# Store all evaluation results for comprehensive tracking
all_evaluation_results = []

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
# COMPREHENSIVE EVALUATION FUNCTIONS
# ============================================================================
@time_function("Model Evaluation")
def evaluate_agent_on_test_data(agent, test_x, test_y, agent_idx, stage_name, stage_number):
    """Comprehensive evaluation of agent model on test data"""
    print(f"\nEvaluating Agent {agent_idx+1} on test data at stage: {stage_name}")
    
    agent.model.eval()
    agent.likelihood.eval()
    
    # Make predictions in batches to avoid memory issues
    batch_size = 1000
    predictions = []
    variances = []
    
    eval_start_time = time.time()
    
    with torch.no_grad():
        for i in range(0, len(test_x), batch_size):
            batch_x = test_x[i:i+batch_size].to(DEVICE)
            
            # Get posterior distribution
            posterior = agent.model(batch_x)
            pred_mean = posterior.mean.cpu()
            pred_var = posterior.variance.cpu()
            
            predictions.append(pred_mean)
            variances.append(pred_var)
    
    # Concatenate all predictions
    all_predictions = torch.cat(predictions, dim=0)
    all_variances = torch.cat(variances, dim=0)
    
    # Calculate comprehensive metrics
    test_y_flat = test_y.flatten()
    pred_flat = all_predictions.flatten()
    
    mse = torch.mean((test_y_flat - pred_flat) ** 2).item()
    mae = torch.mean(torch.abs(test_y_flat - pred_flat)).item()
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = torch.sum((test_y_flat - pred_flat) ** 2).item()
    ss_tot = torch.sum((test_y_flat - torch.mean(test_y_flat)) ** 2).item()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Additional metrics
    mean_uncertainty = torch.mean(torch.sqrt(all_variances)).item()
    max_error = torch.max(torch.abs(test_y_flat - pred_flat)).item()
    
    # FIXED: Prediction intervals coverage (95%) - Convert boolean to float first
    pred_std = torch.sqrt(all_variances).flatten()
    
    # Handle potential NaN/Inf values first
    finite_mask = torch.isfinite(test_y_flat) & torch.isfinite(pred_flat) & torch.isfinite(pred_std)
    
    if finite_mask.any():
        # Use only finite values for coverage calculation
        finite_y = test_y_flat[finite_mask]
        finite_pred = pred_flat[finite_mask]
        finite_std = pred_std[finite_mask]
        
        lower_bound = finite_pred - 1.96 * finite_std
        upper_bound = finite_pred + 1.96 * finite_std
        coverage_mask = (finite_y >= lower_bound) & (finite_y <= upper_bound)
        coverage = torch.mean(coverage_mask.float()).item()  # FIX: Convert boolean to float first
    else:
        coverage = 0.0  # Default if no finite values
    
    eval_time = time.time() - eval_start_time
    
    evaluation_results = {
        'agent_id': agent_idx,
        'agent_name': f'Agent_{agent_idx+1}',
        'stage_number': stage_number,
        'stage_name': stage_name,
        'timestamp': time.time(),
        'evaluation_time': eval_time,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2,
        'max_error': max_error,
        'mean_uncertainty': mean_uncertainty,
        'prediction_coverage_95': coverage,
        'num_test_samples': len(test_y_flat),
        'num_finite_samples': finite_mask.sum().item(),
        'predictions_mean': pred_flat.mean().item(),
        'predictions_std': pred_flat.std().item(),
        'true_values_mean': test_y_flat.mean().item(),
        'true_values_std': test_y_flat.std().item()
    }
    
    print(f"Agent {agent_idx+1} Test Results ({stage_name}):")
    print(f"  - MSE: {mse:.6f}")
    print(f"  - MAE: {mae:.6f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"  - R² Score: {r2:.6f}")
    print(f"  - Max Error: {max_error:.6f}")
    print(f"  - Mean Uncertainty: {mean_uncertainty:.6f}")
    print(f"  - 95% Coverage: {coverage:.4f}")
    print(f"  - Finite Samples: {finite_mask.sum().item()}/{len(test_y_flat)}")
    print(f"  - Evaluation Time: {eval_time:.2f}s")
    
    return evaluation_results

def save_evaluation_results_to_csv(evaluation_results, filename):
    """Save evaluation results to CSV file"""
    df = pd.DataFrame(evaluation_results)
    filepath = f'{validation_dir}/{filename}'
    df.to_csv(filepath, index=False)
    print(f"Evaluation results saved to: {filepath}")
    return filepath

# ============================================================================
# HYPERPARAMETER EXTRACTION
# ============================================================================
def extract_hyperparameters(agent, phase, step_type):
    """Extract comprehensive hyperparameters from agent"""
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
        
        # Add covariance matrix statistics
        covar = chol @ chol.T
        hyper_data['covar_trace'] = np.trace(covar)
        hyper_data['covar_det'] = np.linalg.det(covar)
        hyper_data['covar_frobenius_norm'] = np.linalg.norm(covar, 'fro')
        
    except Exception as e:
        print(f"Warning: Could not extract variational parameters for agent {agent.id}: {e}")
    
    return hyper_data

# ============================================================================
# STAGE 1: INITIAL TRAINING WITH 500 DATA POINTS
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 1: INITIAL TRAINING (500 samples per agent)")
print("=" * 80)

@time_function("Agent Initialization and Training")
def initialize_and_train_agents():
    agents = []
    for i in range(NUM_AGENTS):
        agent_start_time = time.time()
        print(f"\nINITIALIZING Agent {i} with SHARED OPTIMIZED inducing points...")
        
        train_x = torch.tensor(agent_data[i]['x'][:INIT_TRAIN_SIZE], dtype=torch.float32)
        train_y = torch.tensor(agent_data[i]['y'][:INIT_TRAIN_SIZE], dtype=torch.float32)
        neighbors = [(i-1)%NUM_AGENTS, (i+1)%NUM_AGENTS]
        
        # ALL agents use the SAME 128 inducing point locations
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

# Store hyperparameters
print(f"\nSTORING initial hyperparameters...")
for agent in agents:
    hyperparameters_history.append(extract_hyperparameters(agent, 'initial', 'post_training'))

# STAGE 1 EVALUATION: After initial training, before consensus
print(f"\n" + "=" * 60)
print("STAGE 1 EVALUATION: After Initial Training")
print("=" * 60)
stage1_evaluations = []
for i, agent in enumerate(agents):
    eval_results = evaluate_agent_on_test_data(agent, test_x, test_y, i, 
                                              'after_initial_training', 1)
    stage1_evaluations.append(eval_results)
    all_evaluation_results.append(eval_results)

# Save Stage 1 results
save_evaluation_results_to_csv(stage1_evaluations, 'stage1_after_initial_training.csv')

# ============================================================================
# STAGE 2: INITIAL CONSENSUS
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 2: INITIAL CONSENSUS WITH FULL COVARIANCE")
print("=" * 80)

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
        chol_matrices = []
        
        for agent in agents:
            var_dist = agent.model.variational_strategy._variational_distribution
            mean = var_dist.variational_mean.detach().cpu().numpy()
            chol = var_dist.chol_variational_covar.detach().cpu().numpy()
            
            means.append(mean)
            chol_matrices.append(chol)
        
        means = np.stack(means)
        chol_matrices = np.stack(chol_matrices)

        # Record BEFORE consensus
        for i in range(NUM_AGENTS):
            mean_history_initial[i].append(means[i].copy())
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
        
        for i in range(n_inducing):
            for j in range(n_inducing):
                if j <= i:  # Only lower triangular
                    element_values = chol_matrices[:, i, j]
                    
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
        print(f"  DAC Step {step+1} completed in {step_time:.2f}s")

run_initial_dac()

# Store hyperparameters after initial consensus
for agent in agents:
    hyperparameters_history.append(extract_hyperparameters(agent, 'initial', 'post_consensus'))

# STAGE 2 EVALUATION: After initial consensus
print(f"\n" + "=" * 60)
print("STAGE 2 EVALUATION: After Initial Consensus")
print("=" * 60)
stage2_evaluations = []
for i, agent in enumerate(agents):
    eval_results = evaluate_agent_on_test_data(agent, test_x, test_y, i, 
                                              'after_initial_consensus', 2)
    stage2_evaluations.append(eval_results)
    all_evaluation_results.append(eval_results)

# Save Stage 2 results
save_evaluation_results_to_csv(stage2_evaluations, 'stage2_after_initial_consensus.csv')

# Convert initial histories to numpy arrays
mean_history_initial = [np.stack(agent_means) for agent_means in mean_history_initial]
covariance_history_initial = [np.stack(agent_covars) for agent_covars in covariance_history_initial]

# ============================================================================
# STAGE 3: TRAINING WITH ADDITIONAL 200 DATA POINTS
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 3: TRAINING WITH ADDITIONAL 200 DATA POINTS")
print("=" * 80)

@time_function("Additional Data Training")
def run_additional_training():
    print(f"\nADDING additional {ADDITIONAL_DATA_SIZE} data points to each agent...")
    for i, agent in enumerate(agents):
        # Add additional training data
        additional_x = torch.tensor(agent_data[i]['x'][INIT_TRAIN_SIZE:INIT_TRAIN_SIZE+ADDITIONAL_DATA_SIZE], dtype=torch.float32)
        additional_y_raw = agent_data[i]['y'][INIT_TRAIN_SIZE:INIT_TRAIN_SIZE+ADDITIONAL_DATA_SIZE]
        additional_y = torch.tensor(additional_y_raw, dtype=torch.float32)
        
        # Handle shape compatibility
        if agent.train_y.dim() == 2 and additional_y.dim() == 2:
            agent.train_y = torch.cat([agent.train_y, additional_y], dim=0)
        elif agent.train_y.dim() == 1 and additional_y.dim() == 2:
            agent.train_y = torch.cat([agent.train_y, additional_y.flatten()], dim=0)
        elif agent.train_y.dim() == 2 and additional_y.dim() == 1:
            additional_y_reshaped = additional_y.reshape(-1, 1)
            agent.train_y = torch.cat([agent.train_y, additional_y_reshaped], dim=0)
        else:
            agent.train_y = torch.cat([agent.train_y, additional_y.flatten()], dim=0)
        
        agent.train_x = torch.cat([agent.train_x, additional_x], dim=0)
        
        print(f"Agent {i}: Added {len(additional_x)} additional points (total: {len(agent.train_x)})")
        
        # Retrain with additional data
        print(f"RETRAINING Agent {i} with {NUM_ITER} iterations...")
        training_start = time.time()
        agent.train_local(num_iter=NUM_ITER)
        training_time = time.time() - training_start
        print(f"Agent {i} retraining completed in {training_time:.2f}s")

run_additional_training()

# Store hyperparameters after additional training
for agent in agents:
    hyperparameters_history.append(extract_hyperparameters(agent, 'additional', 'post_training'))

# STAGE 3 EVALUATION: After training with additional data
print(f"\n" + "=" * 60)
print("STAGE 3 EVALUATION: After Training with Additional 200 Points")
print("=" * 60)
stage3_evaluations = []
for i, agent in enumerate(agents):
    eval_results = evaluate_agent_on_test_data(agent, test_x, test_y, i, 
                                              'after_additional_training', 3)
    stage3_evaluations.append(eval_results)
    all_evaluation_results.append(eval_results)

# Save Stage 3 results
save_evaluation_results_to_csv(stage3_evaluations, 'stage3_after_additional_training.csv')

# ============================================================================
# STAGE 4: FINAL CONSENSUS AFTER ADDITIONAL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 4: FINAL CONSENSUS AFTER ADDITIONAL TRAINING")
print("=" * 80)

@time_function("Final DAC Consensus with Full Covariance")
def run_final_dac():
    print(f"\nRUNNING final DAC consensus with FULL covariance matrix...")
    for step in range(CONSENSUS_STEPS):
        step_start_time = time.time()
        print(f"\nFINAL DAC Step {step+1}/{CONSENSUS_STEPS}...")
        
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

        # Record BEFORE consensus
        for i in range(NUM_AGENTS):
            mean_history_validation[i].append(means[i].copy())
            covar = chol_matrices[i] @ chol_matrices[i].T
            covariance_history_validation[i].append(covar.copy())

        # Apply same consensus algorithm as initial phase
        # 1. CONSENSUS ON MEANS (precision-weighted)
        precisions = []
        for i in range(NUM_AGENTS):
            covar = chol_matrices[i] @ chol_matrices[i].T
            var = np.diag(covar)
            precision = 1.0 / (var + 1e-6)
            precisions.append(precision)
        
        precisions = np.stack(precisions)
        weighted_means = means * precisions
        
        dac.reset(weighted_means)
        for _ in range(1):
            weighted_means = dac.step(weighted_means)
        
        dac.reset(precisions)
        for _ in range(1):
            precisions = dac.step(precisions)
        
        consensus_means = weighted_means / (precisions + 1e-6)

        # 2. CONSENSUS ON EVERY ELEMENT OF CHOLESKY MATRIX
        consensus_chol_matrices = np.zeros_like(chol_matrices)
        n_inducing = chol_matrices.shape[1]
        
        for i in range(n_inducing):
            for j in range(n_inducing):
                if j <= i:
                    element_values = chol_matrices[:, i, j]
                    
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
        print(f"  DAC Step {step+1} completed in {step_time:.2f}s")

run_final_dac()

# Store final hyperparameters
for agent in agents:
    hyperparameters_history.append(extract_hyperparameters(agent, 'final', 'post_consensus'))

# STAGE 4 EVALUATION: After final consensus
print(f"\n" + "=" * 60)
print("STAGE 4 EVALUATION: After Final Consensus")
print("=" * 60)
stage4_evaluations = []
for i, agent in enumerate(agents):
    eval_results = evaluate_agent_on_test_data(agent, test_x, test_y, i, 
                                              'after_final_consensus', 4)
    stage4_evaluations.append(eval_results)
    all_evaluation_results.append(eval_results)

# Save Stage 4 results
save_evaluation_results_to_csv(stage4_evaluations, 'stage4_after_final_consensus.csv')

# Convert validation histories to numpy arrays
mean_history_validation = [np.stack(agent_means) for agent_means in mean_history_validation]
covariance_history_validation = [np.stack(agent_covars) for agent_covars in covariance_history_validation]

# ============================================================================
# COMPREHENSIVE DATA SAVING AND ANALYSIS
# ============================================================================
@time_function("Comprehensive Data Saving and Analysis")
def save_all_data_and_analysis():
    print(f"\nSAVING comprehensive experiment data...")
    
    # 1. Save all evaluation results
    all_eval_df = pd.DataFrame(all_evaluation_results)
    all_eval_path = f'{validation_dir}/complete_evaluation_results.csv'
    all_eval_df.to_csv(all_eval_path, index=False)
    print(f"Complete evaluation results saved to: {all_eval_path}")
    
    # 2. Save hyperparameters
    hyperparameters_df = pd.DataFrame(hyperparameters_history)
    hyper_path = f'{validation_dir}/hyperparameters_history_optimized_inducing.csv'
    hyperparameters_df.to_csv(hyper_path, index=False)
    print(f"Hyperparameters saved to: {hyper_path}")
    
    # 3. Create performance improvement analysis
    performance_analysis = []
    
    stages = ['after_initial_training', 'after_initial_consensus', 
              'after_additional_training', 'after_final_consensus']
    
    for agent_id in range(NUM_AGENTS):
        agent_analysis = {'agent_id': agent_id}
        
        def test_complete_evaluation_function_fix():
            """Test the complete fix for evaluate_agent_on_test_data function"""
            
            # Fix the coverage calculation in evaluate_agent_on_test_data
            def evaluate_agent_on_test_data_fixed(agent, test_x, test_y, agent_idx, stage_name, stage_number):
                """FIXED VERSION: Comprehensive evaluation of agent model on test data"""
                print(f"\nEvaluating Agent {agent_idx+1} on test data at stage: {stage_name}")
                
                agent.model.eval()
                agent.likelihood.eval()
                
                # Make predictions in batches to avoid memory issues
                batch_size = 1000
                predictions = []
                variances = []
                
                eval_start_time = time.time()
                
                with torch.no_grad():
                    for i in range(0, len(test_x), batch_size):
                        batch_x = test_x[i:i+batch_size].to(DEVICE)
                        
                        # Get posterior distribution
                        posterior = agent.model(batch_x)
                        pred_mean = posterior.mean.cpu()
                        pred_var = posterior.variance.cpu()
                        
                        predictions.append(pred_mean)
                        variances.append(pred_var)
                
                # Concatenate all predictions
                all_predictions = torch.cat(predictions, dim=0)
                all_variances = torch.cat(variances, dim=0)
                
                # Calculate comprehensive metrics
                test_y_flat = test_y.flatten()
                pred_flat = all_predictions.flatten()
                
                mse = torch.mean((test_y_flat - pred_flat) ** 2).item()
                mae = torch.mean(torch.abs(test_y_flat - pred_flat)).item()
                rmse = np.sqrt(mse)
                
                # R² score
                ss_res = torch.sum((test_y_flat - pred_flat) ** 2).item()
                ss_tot = torch.sum((test_y_flat - torch.mean(test_y_flat)) ** 2).item()
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                
                # Additional metrics
                mean_uncertainty = torch.mean(torch.sqrt(all_variances)).item()
                max_error = torch.max(torch.abs(test_y_flat - pred_flat)).item()
                
                # FIXED: Prediction intervals coverage (95%) - Convert boolean to float
                pred_std = torch.sqrt(all_variances).flatten()
                
                # Handle potential NaN/Inf values first
                finite_mask = torch.isfinite(test_y_flat) & torch.isfinite(pred_flat) & torch.isfinite(pred_std)
                
                if finite_mask.any():
                    # Use only finite values for coverage calculation
                    finite_y = test_y_flat[finite_mask]
                    finite_pred = pred_flat[finite_mask]
                    finite_std = pred_std[finite_mask]
                    
                    lower_bound = finite_pred - 1.96 * finite_std
                    upper_bound = finite_pred + 1.96 * finite_std
                    coverage_mask = (finite_y >= lower_bound) & (finite_y <= upper_bound)
                    coverage = torch.mean(coverage_mask.float()).item()  # FIX: Convert boolean to float first
                else:
                    coverage = 0.0  # Default if no finite values
                
                eval_time = time.time() - eval_start_time
                
                evaluation_results = {
                    'agent_id': agent_idx,
                    'agent_name': f'Agent_{agent_idx+1}',
                    'stage_number': stage_number,
                    'stage_name': stage_name,
                    'timestamp': time.time(),
                    'evaluation_time': eval_time,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2_score': r2,
                    'max_error': max_error,
                    'mean_uncertainty': mean_uncertainty,
                    'prediction_coverage_95': coverage,
                    'num_test_samples': len(test_y_flat),
                    'num_finite_samples': finite_mask.sum().item(),
                    'predictions_mean': pred_flat.mean().item(),
                    'predictions_std': pred_flat.std().item(),
                    'true_values_mean': test_y_flat.mean().item(),
                    'true_values_std': test_y_flat.std().item()
                }
                
                print(f"Agent {agent_idx+1} Test Results ({stage_name}):")
                print(f"  - MSE: {mse:.6f}")
                print(f"  - MAE: {mae:.6f}")
                print(f"  - RMSE: {rmse:.6f}")
                print(f"  - R² Score: {r2:.6f}")
                print(f"  - Max Error: {max_error:.6f}")
                print(f"  - Mean Uncertainty: {mean_uncertainty:.6f}")
                print(f"  - 95% Coverage: {coverage:.4f}")
                print(f"  - Finite Samples: {finite_mask.sum().item()}/{len(test_y_flat)}")
                print(f"  - Evaluation Time: {eval_time:.2f}s")
                
                return evaluation_results
            
            print("Fixed evaluation function created successfully")


        def test_boolean_tensor_conversion_methods():
            """Test different methods to convert boolean tensor to float for mean calculation"""
            
            # Create test boolean tensor
            test_tensor = torch.tensor([True, False, True, True, False])
            
            # Method 1: Convert to float then mean (RECOMMENDED)
            result1 = torch.mean(test_tensor.float()).item()
            
            # Method 2: Alternative syntax
            result2 = test_tensor.float().mean().item()
            
            # Method 3: Using sum and division
            result3 = (test_tensor.sum().float() / len(test_tensor)).item()
            
            # Method 4: Using type casting
            result4 = torch.mean(test_tensor.type(torch.float32)).item()
            
            # All methods should give same result
            expected = 3/5  # 3 True values out of 5
            
            assert abs(result1 - expected) < 1e-6, f"Method 1 failed: {result1} vs {expected}"
            assert abs(result2 - expected) < 1e-6, f"Method 2 failed: {result2} vs {expected}"
            assert abs(result3 - expected) < 1e-6, f"Method 3 failed: {result3} vs {expected}"
            assert abs(result4 - expected) < 1e-6, f"Method 4 failed: {result4} vs {expected}"
            
            print(f"All boolean conversion methods passed: {expected}")
            print(f"  Method 1 (recommended): {result1}")
            print(f"  Method 2 (alternative): {result2}")
            print(f"  Method 3 (manual): {result3}")
            print(f"  Method 4 (type cast): {result4}")


        def test_coverage_calculation_edge_cases():
            """Test coverage calculation with various edge cases"""
            
            # Test case 1: Empty tensors
            try:
                empty_y = torch.tensor([])
                empty_pred = torch.tensor([])
                empty_std = torch.tensor([])
                
                lower_bound = empty_pred - 1.96 * empty_std
                upper_bound = empty_pred + 1.96 * empty_std
                coverage_mask = (empty_y >= lower_bound) & (empty_y <= upper_bound)
                
                # This should handle empty tensors gracefully
                if len(coverage_mask) == 0:
                    coverage = 0.0  # Default for empty
                else:
                    coverage = torch.mean(coverage_mask.float()).item()
                
                print(f"Test 1 passed: empty tensors handled, coverage = {coverage}")
            except Exception as e:
                print(f"Test 1 info: empty tensors raise {type(e).__name__}: {e}")
            
            # Test case 2: NaN values in predictions
            test_y_flat = torch.tensor([1.0, 2.0, 3.0])
            pred_flat = torch.tensor([1.0, float('nan'), 3.0])
            pred_std = torch.tensor([0.1, 0.1, 0.1])
            
            # Handle NaN values
            valid_mask = ~torch.isnan(pred_flat) & ~torch.isnan(test_y_flat) & ~torch.isnan(pred_std)
            if valid_mask.any():
                valid_y = test_y_flat[valid_mask]
                valid_pred = pred_flat[valid_mask]
                valid_std = pred_std[valid_mask]
                
                lower_bound = valid_pred - 1.96 * valid_std
                upper_bound = valid_pred + 1.96 * valid_std
                coverage_mask = (valid_y >= lower_bound) & (valid_y <= upper_bound)
                coverage = torch.mean(coverage_mask.float()).item()
                
                print(f"Test 2 passed: NaN handling, coverage = {coverage}")
            else:
                print("Test 2 passed: All NaN values detected and handled")
            
            # Test case 3: Infinite values
            test_y_flat = torch.tensor([1.0, 2.0, 3.0])
            pred_flat = torch.tensor([1.0, float('inf'), 3.0])
            pred_std = torch.tensor([0.1, 0.1, 0.1])
            
            # Handle infinite values
            finite_mask = torch.isfinite(pred_flat) & torch.isfinite(test_y_flat) & torch.isfinite(pred_std)
            if finite_mask.any():
                finite_y = test_y_flat[finite_mask]
                finite_pred = pred_flat[finite_mask]
                finite_std = pred_std[finite_mask]
                
                lower_bound = finite_pred - 1.96 * finite_std
                upper_bound = finite_pred + 1.96 * finite_std
                coverage_mask = (finite_y >= lower_bound) & (finite_y <= upper_bound)
                coverage = torch.mean(coverage_mask.float()).item()
                
                print(f"Test 3 passed: infinite value handling, coverage = {coverage}")
            
            # Test case 4: Zero standard deviation
            test_y_flat = torch.tensor([1.0, 2.0, 3.0])
            pred_flat = torch.tensor([1.0, 2.0, 3.0])
            pred_std = torch.tensor([0.0, 0.0, 0.0])
            
            # Add small epsilon to avoid zero std
            pred_std_safe = pred_std + 1e-8
            lower_bound = pred_flat - 1.96 * pred_std_safe
            upper_bound = pred_flat + 1.96 * pred_std_safe
            coverage_mask = (test_y_flat >= lower_bound) & (test_y_flat <= upper_bound)
            coverage = torch.mean(coverage_mask.float()).item()
            
            assert coverage == 1.0, f"Expected 1.0 for perfect predictions, got {coverage}"
            print(f"Test 4 passed: zero std deviation handling, coverage = {coverage}")


        def test_tensor_device_compatibility():
            """Test coverage calculation with different devices and dtypes"""
            
            # Test case 1: Different dtypes
            dtypes = [torch.float32, torch.float64]
            
            for dtype in dtypes:
                test_y_flat = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
                pred_flat = torch.tensor([1.1, 1.9, 3.1], dtype=dtype)
                pred_std = torch.tensor([0.2, 0.2, 0.2], dtype=dtype)
                
                lower_bound = pred_flat - 1.96 * pred_std
                upper_bound = pred_flat + 1.96 * pred_std
                coverage_mask = (test_y_flat >= lower_bound) & (test_y_flat <= upper_bound)
                coverage = torch.mean(coverage_mask.float()).item()
                
                assert isinstance(coverage, float)
                assert 0.0 <= coverage <= 1.0
                print(f"Test passed for dtype {dtype}: coverage = {coverage}")
            
            # Test case 2: CPU vs GPU compatibility (if available)
            if torch.cuda.is_available():
                test_y_gpu = torch.tensor([1.0, 2.0, 3.0]).cuda()
                pred_gpu = torch.tensor([1.1, 1.9, 3.1]).cuda()
                pred_std_gpu = torch.tensor([0.2, 0.2, 0.2]).cuda()
                
                lower_bound = pred_gpu - 1.96 * pred_std_gpu
                upper_bound = pred_gpu + 1.96 * pred_std_gpu
                coverage_mask = (test_y_gpu >= lower_bound) & (test_y_gpu <= upper_bound)
                coverage = torch.mean(coverage_mask.float()).item()
                
                assert isinstance(coverage, float)
                print(f"GPU test passed: coverage = {coverage}")
            else:
                print("GPU not available, skipping GPU tests")


        def test_batch_processing_simulation():
            """Test coverage calculation in batch processing scenario"""
            
            # Simulate batch processing like in the actual function
            total_samples = 5000
            batch_size = 1000
            
            # Generate synthetic test data
            np.random.seed(42)
            torch.manual_seed(42)
            
            test_y_full = torch.randn(total_samples)
            pred_full = test_y_full + 0.1 * torch.randn(total_samples) # Add some noise
            var_full = torch.abs(torch.randn(total_samples)) * 0.1 + 0.01  # Positive variances
            
            # Process in batches (simulating the actual function behavior)
            all_predictions = []
            all_variances = []
            
            for i in range(0, total_samples, batch_size):
                batch_pred = pred_full[i:i+batch_size]
                batch_var = var_full[i:i+batch_size]
                
                all_predictions.append(batch_pred)
                all_variances.append(batch_var)
            
            # Concatenate all predictions (as in actual function)
            concatenated_pred = torch.cat(all_predictions, dim=0)
            concatenated_var = torch.cat(all_variances, dim=0)
            
            # Calculate coverage
            test_y_flat = test_y_full.flatten()
            pred_flat = concatenated_pred.flatten()
            pred_std = torch.sqrt(concatenated_var).flatten()
            
            lower_bound = pred_flat - 1.96 * pred_std
            upper_bound = pred_flat + 1.96 * pred_std
            coverage_mask = (test_y_flat >= lower_bound) & (test_y_flat <= upper_bound)
            coverage = torch.mean(coverage_mask.float()).item()
            
            # Verify results
            assert isinstance(coverage, float)
            assert 0.0 <= coverage <= 1.0
            assert len(test_y_flat) == total_samples
            assert len(pred_flat) == total_samples
            
            print(f"Batch processing test passed:")
            print(f"  - Total samples: {total_samples}")
            print(f"  - Batch size: {batch_size}")
            print(f"  - Final coverage: {coverage:.4f}")
            print(f"  - Expected coverage ~95%: {abs(coverage - 0.95) < 0.1}")


        def test_memory_efficient_coverage():
            """Test memory-efficient coverage calculation for large datasets"""
            
            # Test with large dataset
            large_size = 10000
            chunk_size = 1000
            
            # Generate large synthetic data
            torch.manual_seed(123)
            test_y_large = torch.randn(large_size)
            pred_large = test_y_large + 0.05 * torch.randn(large_size)
            std_large = torch.ones(large_size) * 0.1
            
            # Method 1: All at once (memory intensive)
            lower_bound_full = pred_large - 1.96 * std_large
            upper_bound_full = pred_large + 1.96 * std_large
            coverage_mask_full = (test_y_large >= lower_bound_full) & (test_y_large <= upper_bound_full)
            coverage_full = torch.mean(coverage_mask_full.float()).item()
            
            # Method 2: Chunk-wise processing (memory efficient)
            total_correct = 0
            total_samples = 0
            
            for i in range(0, large_size, chunk_size):
                end_idx = min(i + chunk_size, large_size)
                
                chunk_y = test_y_large[i:end_idx]
                chunk_pred = pred_large[i:end_idx]
                chunk_std = std_large[i:end_idx]
                
                chunk_lower = chunk_pred - 1.96 * chunk_std
                chunk_upper = chunk_pred + 1.96 * chunk_std
                chunk_mask = (chunk_y >= chunk_lower) & (chunk_y <= chunk_upper)
                
                total_correct += torch.sum(chunk_mask).item()
                total_samples += len(chunk_mask)
            
            coverage_chunked = total_correct / total_samples
            
            # Both methods should give same result
            assert abs(coverage_full - coverage_chunked) < 1e-6, f"Coverage mismatch: {coverage_full} vs {coverage_chunked}"
            
            print(f"Memory efficient test passed:")
            print(f"  - Large dataset size: {large_size}")
            print(f"  - Chunk size: {chunk_size}")
            print(f"  - Full calculation: {coverage_full:.6f}")
            print(f"  - Chunked calculation: {coverage_chunked:.6f}")
            print(f"  - Difference: {abs(coverage_full - coverage_chunked):.8f}")


        def test_real_world_scenario_simulation():
            """Test with realistic GP prediction scenarios"""
            
            # Simulate realistic GP outputs
            n_samples = 1000
            
            # Scenario 1: Good predictions (high coverage expected)
            torch.manual_seed(456)
            true_values = torch.linspace(-2, 2, n_samples)
            predictions = true_values + 0.1 * torch.randn(n_samples)  # Small noise
            uncertainties = torch.ones(n_samples) * 0.2  # Reasonable uncertainty
            
            lower_bound = predictions - 1.96 * uncertainties
            upper_bound = predictions + 1.96 * uncertainties
            coverage_mask = (true_values >= lower_bound) & (true_values <= upper_bound)
            coverage_good = torch.mean(coverage_mask.float()).item()
            
            assert coverage_good > 0.8, f"Expected high coverage, got {coverage_good}"
            print(f"Scenario 1 (good predictions): coverage = {coverage_good:.4f}")
            
            # Scenario 2: Poor predictions (low coverage expected)
            bad_predictions = true_values + 2.0 * torch.randn(n_samples)  # Large noise
            small_uncertainties = torch.ones(n_samples) * 0.05  # Too confident
            
            lower_bound_bad = bad_predictions - 1.96 * small_uncertainties
            upper_bound_bad = bad_predictions + 1.96 * small_uncertainties
            coverage_mask_bad = (true_values >= lower_bound_bad) & (true_values <= upper_bound_bad)
            coverage_bad = torch.mean(coverage_mask_bad.float()).item()
            
            assert coverage_bad < 0.6, f"Expected low coverage, got {coverage_bad}"
            print(f"Scenario 2 (poor predictions): coverage = {coverage_bad:.4f}")
            
            # Scenario 3: Overconfident model (very low coverage)
            overconf_uncertainties = torch.ones(n_samples) * 0.01  # Very small uncertainty
            
            lower_bound_over = predictions - 1.96 * overconf_uncertainties
            upper_bound_over = predictions + 1.96 * overconf_uncertainties
            coverage_mask_over = (true_values >= lower_bound_over) & (true_values <= upper_bound_over)
            coverage_over = torch.mean(coverage_mask_over.float()).item()
            
            print(f"Scenario 3 (overconfident): coverage = {coverage_over:.4f}")
            
            # All scenarios should produce valid coverage values
            for name, cov in [("good", coverage_good), ("bad", coverage_bad), ("overconfident", coverage_over)]:
                assert isinstance(cov, float), f"{name} coverage not float: {type(cov)}"
                assert 0.0 <= cov <= 1.0, f"{name} coverage out of range: {cov}"


        # Run all new tests
        if __name__ == "__main__":
            print("="*80)
            print("RUNNING COMPREHENSIVE COVERAGE CALCULATION TESTS")
            print("="*80)
            
            print("\n1. Testing complete evaluation function fix...")
            test_complete_evaluation_function_fix()
            
            print("\n2. Testing boolean tensor conversion methods...")
            test_boolean_tensor_conversion_methods()
            
            print("\n3. Testing coverage calculation edge cases...")
            test_coverage_calculation_edge_cases()
            
            print("\n4. Testing tensor device compatibility...")
            test_tensor_device_compatibility()
            
            print("\n5. Testing batch processing simulation...")
            test_batch_processing_simulation()
            
            print("\n6. Testing memory efficient coverage...")
            test_memory_efficient_coverage()
            
            print("\n7. Testing real-world scenario simulation...")
            test_real_world_scenario_simulation()
            
            print("\n" + "="*80)
            print("ALL COMPREHENSIVE TESTS PASSED!")
            print("="*80)
            
            print("\nTo fix your error, replace line 312 in evaluate_agent_on_test_data:")
            print("FROM: coverage = torch.mean((test_y_flat >= lower_bound) & (test_y_flat <= upper_bound)).item()")
            print("TO:   coverage = torch.mean(((test_y_flat >= lower_bound) & (test_y_flat <= upper_bound)).float()).item()")
    perf_df = pd.DataFrame(performance_analysis)
    perf_path = f'{validation_dir}/performance_improvement_analysis.csv'
    perf_df.to_csv(perf_path, index=False)
    print(f"Performance improvement analysis saved to: {perf_path}")
    
    # 4. Create summary statistics
    summary_stats = []
    
    for stage in stages:
        stage_data = all_eval_df[all_eval_df['stage_name'] == stage]
        
        if len(stage_data) > 0:
            summary_stats.append({
                'stage_name': stage,
                'stage_number': stage_data['stage_number'].iloc[0],
                'avg_mse': stage_data['mse'].mean(),
                'std_mse': stage_data['mse'].std(),
                'avg_mae': stage_data['mae'].mean(),
                'std_mae': stage_data['mae'].std(),
                'avg_rmse': stage_data['rmse'].mean(),
                'std_rmse': stage_data['rmse'].std(),
                'avg_r2': stage_data['r2_score'].mean(),
                'std_r2': stage_data['r2_score'].std(),
                'avg_uncertainty': stage_data['mean_uncertainty'].mean(),
                'std_uncertainty': stage_data['mean_uncertainty'].std(),
                'avg_coverage': stage_data['prediction_coverage_95'].mean()
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = f'{validation_dir}/summary_statistics.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    # 5. Print comprehensive analysis
    print(f"\n{'='*100}")
    print("COMPREHENSIVE PERFORMANCE ANALYSIS - OPTIMIZED SHARED INDUCING POINTS")
    print(f"{'='*100}")
    
    for _, row in summary_df.iterrows():
        print(f"\n{row['stage_name'].upper().replace('_', ' ')}:")
        print(f"  MSE:  {row['avg_mse']:.6f} ± {row['std_mse']:.6f}")
        print(f"  MAE:  {row['avg_mae']:.6f} ± {row['std_mae']:.6f}")
        print(f"  RMSE: {row['avg_rmse']:.6f} ± {row['std_rmse']:.6f}")
        print(f"  R²:   {row['avg_r2']:.6f} ± {row['std_r2']:.6f}")
        print(f"  Uncertainty: {row['avg_uncertainty']:.6f} ± {row['std_uncertainty']:.6f}")
        print(f"  Coverage: {row['avg_coverage']:.4f}")
    
    # Overall improvement analysis
    initial_stats = summary_df[summary_df['stage_name'] == stages[0]].iloc[0]
    final_stats = summary_df[summary_df['stage_name'] == stages[-1]].iloc[0]
    
    mse_improvement = ((initial_stats['avg_mse'] - final_stats['avg_mse']) / initial_stats['avg_mse']) * 100
    r2_improvement = ((final_stats['avg_r2'] - initial_stats['avg_r2']) / abs(initial_stats['avg_r2'])) * 100
    
    print(f"\n{'='*80}")
    print("OVERALL IMPROVEMENT SUMMARY:")
    print(f"{'='*80}")
    print(f"MSE Improvement:  {mse_improvement:+.2f}%")
    print(f"R² Improvement:   {r2_improvement:+.2f}%")
    print(f"Final Avg MSE:    {final_stats['avg_mse']:.6f}")
    print(f"Final Avg R²:     {final_stats['avg_r2']:.6f}")
    
    return all_eval_path, hyper_path, perf_path, summary_path

all_eval_path, hyper_path, perf_path, summary_path = save_all_data_and_analysis()

# ============================================================================
# CREATE ENHANCED VISUALIZATION PLOTS
# ============================================================================
@time_function("Enhanced Plot Generation")
def create_comprehensive_plots():
    print(f"\nCREATING comprehensive evaluation plots...")
    
    # Load data
    eval_df = pd.read_csv(all_eval_path)
    
    # 1. Multi-stage performance evolution plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    stages = ['after_initial_training', 'after_initial_consensus', 
              'after_additional_training', 'after_final_consensus']
    stage_labels = ['Initial\nTraining', 'Initial\nConsensus', 
                   'Additional\nTraining', 'Final\nConsensus']
    
    metrics = ['mse', 'r2_score', 'rmse', 'mean_uncertainty']
    metric_labels = ['Mean Squared Error', 'R² Score', 'Root Mean Squared Error', 'Mean Uncertainty']
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[metric_idx // 2, metric_idx % 2]
        
        # Prepare data
        stage_means = []
        stage_stds = []
        agent_data = {i: [] for i in range(NUM_AGENTS)}
        
        for stage in stages:
            stage_results = eval_df[eval_df['stage_name'] == stage]
            stage_means.append(stage_results[metric].mean())
            stage_stds.append(stage_results[metric].std())
            
            for agent_id in range(NUM_AGENTS):
                agent_result = stage_results[stage_results['agent_id'] == agent_id]
                if len(agent_result) > 0:
                    agent_data[agent_id].append(agent_result[metric].iloc[0])
                else:
                    agent_data[agent_id].append(np.nan)
        
        # Plot individual agent lines
        for agent_id in range(NUM_AGENTS):
            ax.plot(range(len(stages)), agent_data[agent_id], 
                   marker='o', linewidth=2, markersize=8, alpha=0.7,
                   label=f'Agent {agent_id+1}', color=colors[agent_id])
        
        # Plot average with error bars
        ax.errorbar(range(len(stages)), stage_means, yerr=stage_stds,
                   marker='s', linewidth=3, markersize=10, 
                   color='black', label='Average', capsize=5, capthick=2)
        
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} Evolution\nShared Optimized Inducing Points (128 per agent)', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stage_labels, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{validation_dir}/comprehensive_evaluation_evolution.png', 
               bbox_inches='tight', dpi=150)
    plt.close()
    
    # 2. Performance improvement heatmap
    perf_df = pd.read_csv(perf_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # MSE across all stages
    mse_data = []
    for agent_id in range(NUM_AGENTS):
        agent_row = []
        for stage in stages:
            col_name = f'{stage}_mse'
            if col_name in perf_df.columns:
                value = perf_df[perf_df['agent_id'] == agent_id][col_name].iloc[0]
                agent_row.append(value)
            else:
                agent_row.append(np.nan)
        mse_data.append(agent_row)
    
    im1 = ax1.imshow(mse_data, cmap='RdYlBu_r', aspect='auto')
    ax1.set_title('MSE Across All Stages', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Stage', fontsize=12)
    ax1.set_ylabel('Agent', fontsize=12)
    ax1.set_xticks(range(len(stages)))
    ax1.set_xticklabels(stage_labels, rotation=45)
    ax1.set_yticks(range(NUM_AGENTS))
    ax1.set_yticklabels([f'Agent {i+1}' for i in range(NUM_AGENTS)])
    
    # Add text annotations
    for i in range(NUM_AGENTS):
        for j in range(len(stages)):
            if not np.isnan(mse_data[i][j]):
                ax1.text(j, i, f'{mse_data[i][j]:.4f}', 
                        ha='center', va='center', fontsize=9, color='white')
    
    plt.colorbar(im1, ax=ax1, label='MSE')
    
    # R² Score heatmap
    r2_data = []
    for agent_id in range(NUM_AGENTS):
        agent_row = []
        for stage in stages:
            col_name = f'{stage}_r2_score'
            if col_name in perf_df.columns:
                value = perf_df[perf_df['agent_id'] == agent_id][col_name].iloc[0]
                agent_row.append(value)
            else:
                agent_row.append(np.nan)
        r2_data.append(agent_row)
    
    im2 = ax2.imshow(r2_data, cmap='RdYlBu_r', aspect='auto')
    ax2.set_title('R² Score Across All Stages', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Stage', fontsize=12)
    ax2.set_ylabel('Agent', fontsize=12)
    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels(stage_labels, rotation=45)
    ax2.set_yticks(range(NUM_AGENTS))
    ax2.set_yticklabels([f'Agent {i+1}' for i in range(NUM_AGENTS)])
    
    # Add text annotations
    for i in range(NUM_AGENTS):
        for j in range(len(stages)):
            if not np.isnan(r2_data[i][j]):
                ax2.text(j, i, f'{r2_data[i][j]:.4f}', 
                        ha='center', va='center', fontsize=9, color='white')
    
    plt.colorbar(im2, ax=ax2, label='R² Score')
    
    plt.tight_layout()
    plt.savefig(f'{validation_dir}/performance_heatmap.png', 
               bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Comprehensive plots saved")

create_comprehensive_plots()

# ============================================================================
# FINAL SUMMARY AND CLEANUP
# ============================================================================
print(f"\n" + "=" * 100)
print("EXPERIMENT COMPLETED - OPTIMIZED SHARED INDUCING POINTS WITH COMPREHENSIVE EVALUATION")
print("=" * 100)

print(f"\nAll results saved to: {validation_dir}")
print(f"\nFiles created:")
print(f"  - experiment_log_optimized_inducing_*.txt")
print(f"  - complete_evaluation_results.csv")
print(f"  - stage1_after_initial_training.csv")
print(f"  - stage2_after_initial_consensus.csv")
print(f"  - stage3_after_additional_training.csv")
print(f"  - stage4_after_final_consensus.csv")
print(f"  - performance_improvement_analysis.csv")
print(f"  - summary_statistics.csv")
print(f"  - hyperparameters_history_optimized_inducing.csv")
print(f"  - comprehensive_evaluation_evolution.png")
print(f"  - performance_heatmap.png")

print(f"\nKey Experiment Features Implemented:")
print(f"  ✓ Shared optimized inducing points (128 same locations for all agents)")
print(f"  ✓ 4-stage comprehensive evaluation:")
print(f"    • Stage 1: After initial training (500 samples)")
print(f"    • Stage 2: After initial consensus")
print(f"    • Stage 3: After additional training (200 more samples)")
print(f"    • Stage 4: After final consensus")
print(f"  ✓ Advanced metrics: MSE, MAE, RMSE, R², uncertainty, coverage")
print(f"  ✓ Performance improvement tracking across all stages")
print(f"  ✓ Comprehensive CSV data recording")
print(f"  ✓ Full covariance matrix consensus")

# Close logger and restore stdout
sys.stdout = logger.terminal
logger.close()

print(f"\nOptimized shared inducing points experiment log saved to: {log_file_path}")