import threading
import numpy as np
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import itertools

# Configuration
NUM_AGENTS = 4
BUFFER_SIZE = 500  # Match VSGP
INIT_TRAIN_SIZE = 800  # Match VSGP
STREAMING_BATCH_SIZE = 50  # Match VSGP
DEVICE = 'cpu'

# TEST ALL FUNCTION TYPES
FUNCTION_TYPES = ['multimodal', 'sinusoidal', 'polynomial', 'rbf_mixture']

class FullGPModel(gpytorch.models.ExactGP):
    """Full Gaussian Process Model (no sparse approximation)"""
    
    def __init__(self, train_x, train_y, likelihood):
        super(FullGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Use RBF kernel with ARD (different lengthscales for each dimension)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class FullGPAgent:
    """Agent using Full Gaussian Process without DAC consensus"""
    
    def __init__(self, agent_id, train_x, train_y, neighbors, buffer_size=500, device='cpu'):
        self.agent_id = agent_id
        self.neighbors = neighbors
        self.device = device
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        
        # Initialize data
        self.train_x = train_x.to(device)
        self.train_y = train_y.squeeze(-1).to(device)  # Full GP expects 1D targets
        
        # Initialize GP model and likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = FullGPModel(self.train_x, self.train_y, self.likelihood)
        
        # Move to device
        self.model = self.model.to(device)
        self.likelihood = self.likelihood.to(device)
        
        # Training setup
        self.model.train()
        self.likelihood.train()
        
        # FIXED: Standard GPyTorch approach - use itertools.chain without parameter groups
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.model.parameters(), self.likelihood.parameters()),
            lr=0.1
        )
        
        # Loss function
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        print(f"Agent {agent_id+1} initialized with Full GP:")
        print(f"  Training data: {self.train_x.shape}")
        print(f"  Device: {device}")
        print(f"  Buffer size: {buffer_size}")

    def train_local(self, num_iter=100):
        """Train the Full GP model"""
        print(f"Agent {self.agent_id+1}: Training Full GP for {num_iter} iterations...")
        
        self.model.train()
        self.likelihood.train()
        
        # Create data loader for mini-batch training (if dataset is large)
        if len(self.train_x) > 1000:
            dataset = TensorDataset(self.train_x, self.train_y)
            train_loader = DataLoader(dataset, batch_size=min(200, len(self.train_x)), shuffle=True)
            use_batches = True
        else:
            use_batches = False
        
        training_losses = []
        
        for i in range(num_iter):
            if use_batches:
                # Mini-batch training for large datasets
                batch_losses = []
                for batch_x, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    output = self.model(batch_x)
                    loss = -self.mll(output, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    batch_losses.append(loss.item())
                
                avg_loss = np.mean(batch_losses)
            else:
                # Full batch training for smaller datasets
                self.optimizer.zero_grad()
                output = self.model(self.train_x)
                loss = -self.mll(output, self.train_y)
                loss.backward()
                self.optimizer.step()
                avg_loss = loss.item()
            
            training_losses.append(avg_loss)
            
            if (i + 1) % 20 == 0:
                print(f"  Agent {self.agent_id+1} Iter {i+1:3d}/{num_iter}: Loss = {avg_loss:.4f}")
        
        print(f"Agent {self.agent_id+1}: Full GP training completed!")
        return training_losses
    
    def update_data(self, new_x, new_y):
        """Add new data without unnecessary pruning"""
        with self.lock:
            # Accumulate ALL data - Full GP should use everything
            self.train_x = torch.cat([self.train_x, new_x.to(self.device)], dim=0)
            self.train_y = torch.cat([self.train_y, new_y.squeeze(-1).to(self.device)], dim=0)
            
            # Update model with ALL accumulated data
            self.model.set_train_data(self.train_x, self.train_y, strict=False)
            
            # Recreate optimizer
            self.optimizer = torch.optim.Adam(
                itertools.chain(self.model.parameters(), self.likelihood.parameters()),
                lr=0.1
            )
            
            print(f"Agent {self.agent_id+1}: Updated with {len(new_x)} samples, total: {len(self.train_x)}")
    
    def predict(self, test_x, return_std=True):
        """Make predictions"""
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Make sure test_x is on the right device
            test_x = test_x.to(self.device)
            
            # Get model predictions
            f_preds = self.model(test_x)
            y_preds = self.likelihood(f_preds)
            
            # Extract mean and variance
            mean = y_preds.mean.cpu().numpy()
            variance = y_preds.variance.cpu().numpy()
            std = np.sqrt(variance)
            
            if return_std:
                return mean, std
            else:
                return mean
    
    def get_training_data_size(self):
        """Get current training data size"""
        with self.lock:
            return len(self.train_x)

class FullGPTestEvaluator:
    """Simplified test evaluator for Full GP agents"""
    
    def __init__(self, test_data_path, results_dir, experiment_name):
        self.test_data_path = test_data_path
        self.results_dir = results_dir
        self.experiment_name = experiment_name
        self.evaluation_results = []
        
        # Load test data
        self.test_df = pd.read_csv(test_data_path)
        x_cols = [col for col in self.test_df.columns if col.startswith('x')]
        self.test_x = torch.tensor(self.test_df[x_cols].values, dtype=torch.float32)
        self.test_y = self.test_df['y'].values
        
        print(f"Test evaluator initialized:")
        print(f"  Test data: {len(self.test_y)} samples")
        print(f"  Input dimensions: {len(x_cols)}")
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # CSV file for detailed results
        self.csv_path = os.path.join(results_dir, f"{experiment_name}_evaluation_results.csv")
    
    def evaluate_agents(self, agents, stage_name, stage_number, additional_info=None):
        """Evaluate all agents on test data - FIXED Unicode issue"""
        print(f"\nEvaluating Full GP agents at stage {stage_number}: {stage_name}")
        
        stage_results = []
        
        for agent in agents:
            # Make predictions
            predictions, uncertainties = agent.predict(self.test_x, return_std=True)
            
            # Calculate metrics
            mse = np.mean((predictions - self.test_y)**2)
            mae = np.mean(np.abs(predictions - self.test_y))
            rmse = np.sqrt(mse)
            
            # R² score (FIXED: Use R2 instead of R²)
            ss_res = np.sum((self.test_y - predictions)**2)
            ss_tot = np.sum((self.test_y - np.mean(self.test_y))**2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Uncertainty metrics
            mean_uncertainty = np.mean(uncertainties)
            std_uncertainty = np.std(uncertainties)
            
            # Coverage metrics (95% confidence intervals)
            lower_bound = predictions - 1.96 * uncertainties
            upper_bound = predictions + 1.96 * uncertainties
            coverage_95 = np.mean((self.test_y >= lower_bound) & (self.test_y <= upper_bound))
            
            # Get training data size
            training_data_size = agent.get_training_data_size()
            
            # Compile results
            result = {
                'timestamp': time.time(),
                'experiment_name': self.experiment_name,
                'stage_number': stage_number,
                'stage_name': stage_name,
                'agent_id': agent.agent_id,
                'agent_name': f'Agent_{agent.agent_id+1}',
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2_score,
                'mean_uncertainty': mean_uncertainty,
                'std_uncertainty': std_uncertainty,
                'coverage_95': coverage_95,
                'training_data_size': training_data_size,
                'test_data_size': len(self.test_y),
                'prediction_samples': len(predictions)
            }
            
            # Add additional info
            if additional_info:
                result.update(additional_info)
            
            stage_results.append(result)
            self.evaluation_results.append(result)
            
            # FIXED: Use R2 instead of R² to avoid Unicode issues
            print(f"  Agent {agent.agent_id+1}: MSE={mse:.6f}, R2={r2_score:.4f}, "
                  f"Coverage={coverage_95:.3f}, Data={training_data_size}")
        
        # Save results to CSV
        self._save_to_csv()
        
        return stage_results
    
    def _save_to_csv(self):
        """Save evaluation results to CSV"""
        df = pd.DataFrame(self.evaluation_results)
        df.to_csv(self.csv_path, index=False)

def run_full_gp_experiment_for_function_type(function_type):
    """Run complete Full GP streaming experiment with stage-by-stage performance output"""
    
    print(f"""
{'='*100}
FULL GP STREAMING EXPERIMENT - FUNCTION TYPE: {function_type.upper()}
{'='*100}
Configuration:
  - Function Type: {function_type}
  - Number of Agents: {NUM_AGENTS}
  - Buffer Size: {BUFFER_SIZE}
  - Initial Training Size: {INIT_TRAIN_SIZE}
  - Streaming Batch Size: {STREAMING_BATCH_SIZE}
  - Model Type: Full Gaussian Process (Exact GP)
  - Pattern: Online retraining with streaming data
{'='*100}
""")

    # Set up directories
    save_dir = f"project/train_record/full_gp_{function_type}_streaming"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize test evaluator
    test_data_path = f'project/data/synthetic/{function_type}/synthetic_test.csv'
    
    # Check if data exists
    if not os.path.exists(test_data_path):
        print(f"ERROR: Test data not found for {function_type}: {test_data_path}")
        print("Please run generate_synthetic_data.py first!")
        return None
    
    evaluator = FullGPTestEvaluator(
        test_data_path=test_data_path,
        results_dir=save_dir,
        experiment_name=f"full_gp_{function_type}_streaming_experiment"
    )

    def load_synthetic_agent_data(agent_idx, function_type):
        """Load synthetic training data for a specific agent"""
        data_path = f'project/data/synthetic/{function_type}/synthetic_train_agent{agent_idx+1}.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Synthetic data not found: {data_path}")
        
        df = pd.read_csv(data_path)
        x_cols = [col for col in df.columns if col.startswith('x')]
        x = df[x_cols].values
        y = df['y'].values.reshape(-1, 1)
        
        print(f"  Agent {agent_idx+1}: {len(x)} total training samples")
        return {'x': x, 'y': y}

    # Load data
    print(f"Loading {function_type} training data...")
    agent_data = []
    total_training_samples = 0

    for i in range(NUM_AGENTS):
        data = load_synthetic_agent_data(i, function_type)
        agent_data.append(data)
        total_training_samples += len(data['x'])

    print(f"\n{function_type.upper()} Training Data Summary:")
    print(f"  - Total training samples across all agents: {total_training_samples}")
    print(f"  - Average samples per agent: {total_training_samples // NUM_AGENTS}")

    # Initialize Full GP agents with initial portion of data
    print(f"\nInitializing Full GP agents with {INIT_TRAIN_SIZE} samples each...")
    agents = []
    for i in range(NUM_AGENTS):
        # Start with initial portion
        available_samples = len(agent_data[i]['x'])
        init_size = min(INIT_TRAIN_SIZE, available_samples)
        
        train_x = torch.tensor(agent_data[i]['x'][:init_size], dtype=torch.float32)
        train_y = torch.tensor(agent_data[i]['y'][:init_size], dtype=torch.float32)
        neighbors = [(i-1) % NUM_AGENTS, (i+1) % NUM_AGENTS]
        
        print(f"  Agent {i+1}: Starting with {init_size}/{available_samples} samples")
        
        agent = FullGPAgent(
            agent_id=i,
            train_x=train_x,
            train_y=train_y,
            neighbors=neighbors,
            buffer_size=BUFFER_SIZE,
            device=DEVICE
        )
        agents.append(agent)

    # ============================================================================
    # STAGE 1: INITIAL TRAINING
    # ============================================================================
    print(f"\n[{function_type.upper()}] STAGE 1: INITIAL FULL GP TRAINING")
    print(f"{'='*80}")

    # Train all agents initially
    train_threads = []
    for i, agent in enumerate(agents):
        def train_agent(agent, idx, func_type):
            start_time = time.time()
            agent.train_local(num_iter=200)  # Match VSGP iterations
            training_time = time.time() - start_time
            print(f"[{func_type}] Agent {idx+1} Full GP training completed in {training_time:.2f}s")
        
        t = threading.Thread(target=train_agent, args=(agent, i, function_type))
        t.start()
        train_threads.append(t)

    for t in train_threads:
        t.join()

    # EVALUATE STAGE 1
    stage_results = evaluator.evaluate_agents(
        agents, 
        stage_name="initial_training", 
        stage_number=1,
        additional_info={
            'model_type': 'full_gp',
            'training_iterations': 200,
            'samples_per_agent': INIT_TRAIN_SIZE,
            'total_training_samples': NUM_AGENTS * INIT_TRAIN_SIZE,
            'function_type': function_type
        }
    )

    # ============================================================================
    # STREAMING DATA SIMULATION WITH ONLINE RETRAINING
    # ============================================================================
    print(f"\n[{function_type.upper()}] FULL GP ONLINE STREAMING SIMULATION")
    print(f"{'='*80}")

    # Track data indices for each agent
    current_data_indices = [INIT_TRAIN_SIZE] * NUM_AGENTS
    stage_number = 2
    
    # Track last retrain size for each agent
    last_retrain_size = [INIT_TRAIN_SIZE] * NUM_AGENTS  # Track when each agent last retrained

    # Continue until ALL data is consumed
    max_cycles = 50  # Enough to handle all data
    
    # Alternative: Retrain every N cycles instead of buffer size
    RETRAIN_EVERY_N_CYCLES = 10  # Retrain every 10 cycles

    for cycle in range(max_cycles):
        print(f"\n[{function_type}] === STREAMING CYCLE {cycle+1}/{max_cycles} ===")
        
        # Check if any agent has more data to stream
        agents_with_available_data = []
        for i in range(NUM_AGENTS):
            if current_data_indices[i] < len(agent_data[i]['x']):
                agents_with_available_data.append(i)
        
        if not agents_with_available_data:
            print(f"[{function_type}] All data consumed. Ending simulation.")
            break
        
        # Add new data batch to each agent (if available)
        agents_with_new_data = []
        
        for i in agents_with_available_data:
            current_idx = current_data_indices[i]
            end_idx = min(current_idx + STREAMING_BATCH_SIZE, len(agent_data[i]['x']))
            
            if current_idx < end_idx:  # Still has data to add
                batch_x = torch.tensor(agent_data[i]['x'][current_idx:end_idx], dtype=torch.float32)
                batch_y = torch.tensor(agent_data[i]['y'][current_idx:end_idx], dtype=torch.float32)
                
                agent = agents[i]
                agent.update_data(batch_x, batch_y)
                current_data_indices[i] = end_idx
                agents_with_new_data.append(i)
                
                print(f"  Agent {i+1}: Added {len(batch_x)} samples, "
                      f"total: {agent.get_training_data_size()}, "
                      f"remaining: {len(agent_data[i]['x']) - end_idx}")
        
        # FIXED: Check if any agent needs retraining based on BUFFER_SIZE trigger
        agents_to_retrain = []
        for i in agents_with_new_data:
            current_size = agents[i].get_training_data_size()
            samples_since_last_retrain = current_size - last_retrain_size[i]
            
            # Retrain if we've added BUFFER_SIZE samples since last retrain
            if samples_since_last_retrain >= BUFFER_SIZE:
                agents_to_retrain.append(i)
                print(f"  Agent {i+1}: Triggering retrain ({samples_since_last_retrain} samples since last retrain)")
        
        # Simple periodic retraining
        if (cycle + 1) % RETRAIN_EVERY_N_CYCLES == 0 and agents_with_new_data:
            agents_to_retrain = agents_with_new_data  # Retrain all agents that got new data
            print(f"[{function_type}] Cycle {cycle+1}: Periodic retrain trigger (every {RETRAIN_EVERY_N_CYCLES} cycles)")
        
        # Online retraining for agents that hit the buffer trigger
        if agents_to_retrain:
            print(f"[{function_type}] Cycle {cycle+1}: Online retraining {len(agents_to_retrain)} Full GP agents")
            
            train_threads = []
            for agent_id in agents_to_retrain:
                def retrain_agent(agent, idx):
                    retrain_start = time.time()
                    agent.train_local(num_iter=50)  # Quick online retraining
                    retrain_time = time.time() - retrain_start
                    print(f"    Agent {idx+1} online retrained in {retrain_time:.2f}s")
                
                t = threading.Thread(target=retrain_agent, args=(agents[agent_id], agent_id))
                t.start()
                train_threads.append(t)
            
            for t in train_threads:
                t.join()
            
            # Update last retrain size for retrained agents
            for agent_id in agents_to_retrain:
                last_retrain_size[agent_id] = agents[agent_id].get_training_data_size()
            
            # Evaluate after online retraining
            stage_results = evaluator.evaluate_agents(
                agents, 
                stage_name=f"online_retrain_cycle_{cycle+1}", 
                stage_number=stage_number,
                additional_info={
                    'model_type': 'full_gp',
                    'cycle': cycle + 1,
                    'agents_retrained': len(agents_to_retrain),
                    'retraining_type': 'online',
                    'function_type': function_type,
                    'buffer_trigger': BUFFER_SIZE
                }
            )
            stage_number += 1
        
        # Always evaluate after data update (even without retraining)
        else:
            stage_results = evaluator.evaluate_agents(
                agents, 
                stage_name=f"data_update_cycle_{cycle+1}", 
                stage_number=stage_number,
                additional_info={
                    'model_type': 'full_gp',
                    'cycle': cycle + 1,
                    'agents_updated': len(agents_with_new_data),
                    'retraining_type': 'none',
                    'function_type': function_type
                }
            )
            stage_number += 1
        
        # Print progress every 5 cycles
        if (cycle + 1) % 5 == 0:
            total_remaining = sum(len(agent_data[i]['x']) - current_data_indices[i] for i in range(NUM_AGENTS))
            total_consumed = sum(current_data_indices[i] - INIT_TRAIN_SIZE for i in range(NUM_AGENTS))
            progress_pct = (total_consumed / (total_consumed + total_remaining)) * 100 if (total_consumed + total_remaining) > 0 else 0
            print(f"[{function_type}] Progress: {total_consumed} samples consumed, {total_remaining} remaining ({progress_pct:.1f}%)")

    # ============================================================================
    # FINAL COMPREHENSIVE TRAINING
    # ============================================================================
    print(f"\n[{function_type.upper()}] FINAL COMPREHENSIVE TRAINING")
    print(f"{'='*80}")
    
    # Final training with all accumulated data
    print("Final comprehensive training with all accumulated data...")
    train_threads = []
    for i, agent in enumerate(agents):
        def final_train_agent(agent, idx):
            final_start = time.time()
            agent.train_local(num_iter=100)  # More thorough final training
            final_time = time.time() - final_start
            print(f"  Agent {idx+1} final training completed in {final_time:.2f}s")
        
        t = threading.Thread(target=final_train_agent, args=(agent, i))
        t.start()
        train_threads.append(t)

    for t in train_threads:
        t.join()

    # Final evaluation
    final_results = evaluator.evaluate_agents(
        agents, 
        stage_name="final_comprehensive", 
        stage_number=stage_number,
        additional_info={
            'model_type': 'full_gp',
            'experiment_type': 'full_gp_streaming',
            'retraining_type': 'comprehensive',
            'function_type': function_type,
            'total_data_consumed': sum(current_data_indices[i] - INIT_TRAIN_SIZE for i in range(NUM_AGENTS))
        }
    )

    # Performance summary
    all_results = evaluator.evaluation_results
    if all_results:
        df = pd.DataFrame(all_results)
        
        initial_results = df[df['stage_number'] == 1]
        final_results = df[df['stage_number'] == stage_number]
        
        if len(initial_results) > 0 and len(final_results) > 0:
            initial_mse = initial_results['mse'].mean()
            final_mse = final_results['mse'].mean()
            initial_r2 = initial_results['r2_score'].mean()
            final_r2 = final_results['r2_score'].mean()
            
            mse_improvement = ((initial_mse - final_mse) / initial_mse) * 100
            r2_improvement = ((final_r2 - initial_r2) / abs(initial_r2 + 1e-8)) * 100
            
            print(f"\n[{function_type.upper()}] FULL GP ONLINE STREAMING RESULTS:")
            print("="*70)
            print(f"Total Stages: {stage_number}")
            print(f"Initial MSE: {initial_mse:.6f} -> Final MSE: {final_mse:.6f}")
            print(f"MSE Improvement: {mse_improvement:+.2f}%")
            print(f"Initial R2: {initial_r2:.6f} -> Final R2: {final_r2:.6f}")
            print(f"R2 Improvement: {r2_improvement:+.2f}%")
            print(f"Results saved to: {evaluator.csv_path}")
            
            return {
                'function_type': function_type,
                'model_type': 'full_gp',
                'initial_mse': initial_mse,
                'final_mse': final_mse,
                'mse_improvement': mse_improvement,
                'initial_r2': initial_r2,
                'final_r2': final_r2,
                'r2_improvement': r2_improvement,
                'total_stages': stage_number,
                'csv_path': evaluator.csv_path
            }
    
    return None

def main_full_gp_multi_function_experiment():
    """Run Full GP experiments for all function types - Simple output"""
    
    print(f"""
{'='*120}
FULL GP MULTI-FUNCTION ONLINE STREAMING EXPERIMENT
Testing Function Types: {', '.join(FUNCTION_TYPES)}
Model: Full Gaussian Process (Exact GP) - Online Retraining
{'='*120}
""")
    
    all_results = {}
    
    # Run experiment for each function type
    for i, function_type in enumerate(FUNCTION_TYPES):
        print(f"\n{'*'*100}")
        print(f"FULL GP EXPERIMENT {i+1}/{len(FUNCTION_TYPES)}: {function_type.upper()}")
        print(f"{'*'*100}")
        
        try:
            result = run_full_gp_experiment_for_function_type(function_type)
            if result:
                all_results[function_type] = result
                print(f"SUCCESS: {function_type} Full GP experiment completed!")
            else:
                print(f"FAILED: {function_type} Full GP experiment failed!")
        except Exception as e:
            print(f"ERROR in {function_type} Full GP experiment: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"""
{'='*120}
FULL GP MULTI-FUNCTION EXPERIMENT COMPLETED!
{'='*120}
Successful experiments: {len(all_results)}/{len(FUNCTION_TYPES)}
""")
    
    for func_type, result in all_results.items():
        print(f"  SUCCESS {func_type}: MSE improved by {result['mse_improvement']:+.2f}%, "
              f"R2 improved by {result['r2_improvement']:+.2f}%, Total stages: {result['total_stages']}")
    
    return all_results

if __name__ == "__main__":
    # Run the comprehensive Full GP experiment
    results = main_full_gp_multi_function_experiment()
    
    print(f"\n{'='*120}")
    print("FULL GP ONLINE STREAMING EXPERIMENT COMPLETE!")
    print("Stage-by-stage performance with online retraining recorded.")
    print(f"{'='*120}")