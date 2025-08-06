import threading
import numpy as np
import torch
from agent import Agent
from dac import DACConsensus
from test_evaluator import TestEvaluator
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# FIXED: Import only the existing function, remove the missing one
try:
    from agent_performance_analyzer import analyze_single_experiment
    ANALYZER_AVAILABLE = True
except ImportError:
    print("Warning: agent_performance_analyzer not available, using built-in analysis")
    ANALYZER_AVAILABLE = False
    
    def analyze_single_experiment(csv_path):
        """Fallback analyzer function"""
        return "Built-in analysis complete"

# NEW: Add VSGP imports for baseline comparison
import gpytorch
from gpytorch.models import VariationalGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood

# UPDATED Configuration to match new dataset sizes
NUM_AGENTS = 4

# Dynamic configuration based on function type
FUNCTION_CONFIGS = {
    'sinusoidal': {
        'buffer_size': 30,      # Reduced for smaller dataset
        'init_train_size': 150,  # ~40% of agent data (375 * 0.4)
        'streaming_batch_size': 30,
        'max_cycles': 50        # Fewer cycles needed
    },
    'polynomial': {
        'buffer_size': 50,      # Moderate
        'init_train_size': 250,  # ~40% of agent data (625 * 0.4)
        'streaming_batch_size': 50,
        'max_cycles': 50
    },
    'multimodal': {
        'buffer_size': 80,      # Larger dataset allows larger buffer
        'init_train_size': 400,  # ~40% of agent data (1000 * 0.4)
        'streaming_batch_size': 80,
        'max_cycles': 50
    },
    'rbf_mixture': {
        'buffer_size': 60,      # Medium-large
        'init_train_size': 300,  # ~40% of agent data (750 * 0.4)
        'streaming_batch_size': 60,
        'max_cycles': 50
    }
}

DEVICE = 'cpu'

# TEST ALL FUNCTION TYPES
FUNCTION_TYPES = ['multimodal', 'sinusoidal', 'polynomial', 'rbf_mixture']

# NEW: Weighted DAC with PoE (Product of Experts)
class WeightedDACConsensus:
    """Weighted DAC using Product of Experts for uncertainty-based weighting"""
    
    def __init__(self, L, alpha=0.2, use_uncertainty_weights=True, weighting_scheme="enhanced_poe"):
        self.L = L
        self.alpha = alpha
        self.use_uncertainty_weights = use_uncertainty_weights
        self.weighting_scheme = weighting_scheme  # "standard_poe", "enhanced_poe", "softmax", "power"
        self.n_agents = L.shape[0]
        
    def compute_weights(self, means, variances):
        """Compute PoE-based weights using agent uncertainties with enhanced differentiation"""
        if not self.use_uncertainty_weights:
            # Uniform weights for standard DAC
            return np.ones((self.n_agents, means.shape[1])) / self.n_agents
        
        epsilon = 1e-8
        
        if self.weighting_scheme == "standard_poe":
            # Original PoE: Weight inversely proportional to variance
            precisions = 1.0 / (variances + epsilon)
            weights = precisions / (precisions.sum(axis=0, keepdims=True) + epsilon)
            
        elif self.weighting_scheme == "enhanced_poe":
            # Enhanced PoE: Amplify differences using power law + relative scaling
            precisions = 1.0 / (variances + epsilon)
            
            # Apply power law to amplify differences (power=2.5)
            enhanced_precisions = precisions ** 2.5
            
            # Add relative scaling based on variance ratios
            var_ratios = variances / (variances.mean(axis=0, keepdims=True) + epsilon)
            confidence_bonus = np.exp(-2.0 * (var_ratios - 1.0))  # Bonus for below-average variance
            
            final_precisions = enhanced_precisions * confidence_bonus
            weights = final_precisions / (final_precisions.sum(axis=0, keepdims=True) + epsilon)
            
        elif self.weighting_scheme == "softmax":
            # Softmax weighting with low temperature for sharp differences
            precisions = 1.0 / (variances + epsilon)
            temperature = 0.2  # Low temperature = sharper distinctions
            log_precisions = np.log(precisions + epsilon)
            weights = np.exp(log_precisions / temperature)
            weights = weights / (weights.sum(axis=0, keepdims=True) + epsilon)
            
        elif self.weighting_scheme == "power":
            # Power law weighting
            precisions = 1.0 / (variances + epsilon)
            power = 3.0
            powered_precisions = precisions ** power
            weights = powered_precisions / (powered_precisions.sum(axis=0, keepdims=True) + epsilon)
            
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting_scheme}")
        
        return weights
    
    def weighted_consensus_step(self, means, variances):
        """Perform one weighted consensus step using enhanced PoE"""
        
        # Compute enhanced PoE-based weights
        weights = self.compute_weights(means, variances)
        
        # DEBUG: Print weight statistics for first few steps
        weight_std = np.std(weights, axis=0).mean()
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
            
        if self._step_count <= 3:  # Only print for first 3 steps
            print(f"    [WeightedDAC] Step {self._step_count}: Weight std = {weight_std:.6f}, "
                  f"Scheme = {self.weighting_scheme}")
            print(f"    [WeightedDAC] Agent weights (inducing 0): {weights[:, 0]}")
        
        # Weighted mean computation (PoE for means)
        weighted_means = np.sum(weights * means, axis=0, keepdims=True)
        weighted_means = np.repeat(weighted_means, self.n_agents, axis=0)
        
        # PoE for variances: 1/var_combined = sum(1/var_i) 
        epsilon = 1e-8
        precisions = 1.0 / (variances + epsilon)
        combined_precision = precisions.sum(axis=0, keepdims=True)
        combined_variance = 1.0 / (combined_precision + epsilon)
        weighted_variances = np.repeat(combined_variance, self.n_agents, axis=0)
        
        # Apply DAC mixing with weighted targets
        new_means = means - self.alpha * (self.L @ (means - weighted_means))
        new_variances = variances - self.alpha * (self.L @ (variances - weighted_variances))
        
        # Ensure variances remain positive
        new_variances = np.maximum(new_variances, epsilon)
        
        return new_means, new_variances

# NEW: VSGP Baseline Model for whole dataset comparison
class BaselineVSGP(VariationalGP):
    """Baseline VSGP model trained on entire dataset"""
    
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(BaselineVSGP, self).__init__(variational_strategy)
        
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_baseline_vsgp(train_x, train_y, inducing_points, num_iter=200):
    """Train baseline VSGP on entire dataset"""
    
    print(f"Training baseline VSGP on {len(train_x)} samples...")
    
    # Initialize model and likelihood
    model = BaselineVSGP(inducing_points)
    likelihood = GaussianLikelihood()
    
    # Set to training mode
    model.train()
    likelihood.train()
    
    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)
    
    # Loss function
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))
    
    # Training loop
    start_time = time.time()
    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y.squeeze())
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 50 == 0:
            print(f"  Iteration {i+1}/{num_iter}, Loss: {loss.item():.4f}")
    
    training_time = time.time() - start_time
    print(f"Baseline VSGP training completed in {training_time:.2f}s")
    
    return model, likelihood

def load_synthetic_agent_data(agent_idx, function_type):
    """Load synthetic training data for a specific agent"""
    data_path = f'project/data/synthetic/{function_type}/agent{agent_idx+1}.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Synthetic data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    x_cols = [col for col in df.columns if col.startswith('x')]
    x = df[x_cols].values
    
    # FIXED: Use noisy training data (y column should contain noisy observations)
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

def load_synthetic_inducing_points(function_type):
    """Load synthetic inducing points"""
    inducing_path = f'project/data/synthetic/{function_type}/inducing.csv'
    if not os.path.exists(inducing_path):
        raise FileNotFoundError(f"Inducing points not found: {inducing_path}")
    
    inducing_df = pd.read_csv(inducing_path)
    inducing_x_cols = [col for col in inducing_df.columns if col.startswith('x')]
    inducing_points = inducing_df[inducing_x_cols].values
    
    print(f"  Inducing points: {inducing_points.shape[0]} points")
    return inducing_points

def load_full_training_data(function_type):
    """Load complete training dataset for baseline comparison (using noisy training data)"""
    
    print(f"Loading full training data for {function_type}...")
    
    # Load all agent data and combine
    all_x = []
    all_y = []
    
    for i in range(NUM_AGENTS):
        agent_data = load_synthetic_agent_data(i, function_type)
        all_x.append(agent_data['x'])
        all_y.append(agent_data['y'])  # This should already be noisy training data
    
    # Combine all agent data
    full_x = np.vstack(all_x)
    full_y = np.vstack(all_y)
    
    print(f"  Total training samples: {len(full_x)} (noisy training data)")
    print(f"  X range: [{full_x[:, 0].min():.2f}, {full_x[:, 0].max():.2f}] × [{full_x[:, 1].min():.2f}, {full_x[:, 1].max():.2f}]")
    print(f"  Y range: [{full_y.min():.2f}, {full_y.max():.2f}] (noisy observations)")
    
    return torch.tensor(full_x, dtype=torch.float32), torch.tensor(full_y, dtype=torch.float32)

def run_dac_consensus(agents, dac, consensus_steps=5, consensus_type="standard", convergence_threshold=1e-4):
    """Run DAC consensus (standard or weighted) with convergence detection"""
    print(f"Running {consensus_type} DAC consensus for up to {consensus_steps} steps...")
    
    consensus_history = []
    converged = False
    
    for step in range(consensus_steps):
        print(f"  DAC step {step+1}/{consensus_steps}")
        
        # Collect agent states
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
        
        # Calculate convergence metrics
        mean_convergence = np.std(means, axis=0).mean()
        var_convergence = np.std(vars, axis=0).mean()
        
        # NEW: Calculate additional convergence metrics
        max_mean_diff = np.max(np.std(means, axis=0))    # Maximum disagreement
        relative_mean_change = mean_convergence / (np.abs(means).mean() + 1e-8)  # Relative convergence
        
        # NEW: Pairwise agent disagreement
        pairwise_distances = []
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                mean_dist = np.linalg.norm(means[i] - means[j])
                var_dist = np.linalg.norm(vars[i] - vars[j])
                pairwise_distances.append(mean_dist + var_dist)
        
        avg_pairwise_distance = np.mean(pairwise_distances) if pairwise_distances else 0
        
        consensus_info = {
            'step': step,
            'mean_convergence': mean_convergence,
            'var_convergence': var_convergence,
            'max_mean_diff': max_mean_diff,
            'relative_mean_change': relative_mean_change,
            'avg_pairwise_distance': avg_pairwise_distance,
            'consensus_type': consensus_type
        }
        
        consensus_history.append(consensus_info)
        
        # ENHANCED: Check for convergence (multiple criteria)
        convergence_checks = [
            mean_convergence < convergence_threshold,           # Overall mean agreement
            var_convergence < convergence_threshold,            # Overall variance agreement  
            max_mean_diff < convergence_threshold * 10,         # No outlier agents
            relative_mean_change < 0.01                         # Relative change small
        ]
        
        if all(convergence_checks) and step >= 2:  # Minimum 3 steps
            converged = True
            print(f"  ✓ DAC CONVERGED at step {step+1}!")
            print(f"    Mean convergence: {mean_convergence:.6f}")
            print(f"    Var convergence: {var_convergence:.6f}")
            print(f"    Max disagreement: {max_mean_diff:.6f}")
            print(f"    Relative change: {relative_mean_change:.6f}")
            break
        
        # Print convergence progress
        print(f"    Mean conv: {mean_convergence:.6f}, Var conv: {var_convergence:.6f}, Max diff: {max_mean_diff:.6f}")
        
        # Apply appropriate consensus method
        if isinstance(dac, WeightedDACConsensus):
            # Weighted DAC with PoE
            means, vars = dac.weighted_consensus_step(means, vars)
        else:
            # Standard DAC
            dac.reset(means)
            for _ in range(1):
                means = dac.step(means)
            
            dac.reset(vars)
            for _ in range(1):
                vars = dac.step(vars)
        
        # Update agents
        for i, agent in enumerate(agents):
            with agent.lock:
                agent.consensus_mean = means[i] 
                agent.consensus_var = vars[i]
                agent.inject_consensus_to_variational()
    
    # Final convergence status
    actual_steps = len(consensus_history)
    final_info = consensus_history[-1]
    
    if converged:
        print(f"  DAC converged in {actual_steps} steps (threshold: {convergence_threshold})")
    else:
        print(f"  DAC reached max steps ({consensus_steps}) without full convergence")
        
        # Check if we're close to convergence
        final_conv = final_info['mean_convergence']
        if final_conv < convergence_threshold * 10:
            print(f"  Close to convergence (final: {final_conv:.6f})")
        else:
            print(f"  Still divergent (final: {final_conv:.6f})")
    
    print(f"  Final metrics: Mean={final_info['mean_convergence']:.6f}, Var={final_info['var_convergence']:.6f}")
    
    return consensus_history, converged, actual_steps

# REMOVE the duplicate adaptive_dac_consensus function at the bottom of the file

def plot_dac_convergence(consensus_history, save_path=None):
    """Plot DAC convergence metrics"""
    
    if not consensus_history:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    steps = [h['step'] for h in consensus_history]
    mean_conv = [h['mean_convergence'] for h in consensus_history]
    var_conv = [h['var_convergence'] for h in consensus_history]
    max_diff = [h['max_mean_diff'] for h in consensus_history]
    pairwise_dist = [h['avg_pairwise_distance'] for h in consensus_history]
    
    # Mean convergence
    ax1 = axes[0, 0]
    ax1.semilogy(steps, mean_conv, 'b-o', label='Mean Convergence')
    ax1.set_xlabel('DAC Step')
    ax1.set_ylabel('Mean Convergence (log scale)')
    ax1.set_title('Mean Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Variance convergence
    ax2 = axes[0, 1]
    ax2.semilogy(steps, var_conv, 'r-o', label='Variance Convergence')
    ax2.set_xlabel('DAC Step')
    ax2.set_ylabel('Variance Convergence (log scale)')
    ax2.set_title('Variance Convergence')
    ax2.grid(True, alpha=0.3)
    
    # Maximum disagreement
    ax3 = axes[1, 0]
    ax3.semilogy(steps, max_diff, 'g-o', label='Max Disagreement')
    ax3.set_xlabel('DAC Step')
    ax3.set_ylabel('Max Agent Disagreement (log scale)')
    ax3.set_title('Maximum Agent Disagreement')
    ax3.grid(True, alpha=0.3)
    
    # Pairwise distances
    ax4 = axes[1, 1]
    ax4.semilogy(steps, pairwise_dist, 'm-o', label='Avg Pairwise Distance')
    ax4.set_xlabel('DAC Step')
    ax4.set_ylabel('Average Pairwise Distance (log scale)')
    ax4.set_title('Average Pairwise Agent Distance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")
    
    plt.show()
    return fig

class SimpleProgressTracker:
    """Track stage-by-stage progress with simplified naming and agent-level details"""
    
    def __init__(self, save_dir, experiment_name):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.stage_results = []
    
    def add_stage(self, stage_number, stage_name, agents_results):
        """Add stage results with agent-level tracking"""
        
        # Record each agent individually with R2 focus
        for agent_result in agents_results:
            agent_record = {
                'stage_number': stage_number,
                'stage_name': stage_name,
                'agent_id': agent_result['agent_id'],
                'r2_score': agent_result['r2_score'],  # PRIMARY METRIC
                'mse': agent_result['mse'],
                'rmse': agent_result['rmse'], 
                'mae': agent_result['mae'],
                'avg_uncertainty': agent_result.get('avg_prediction_uncertainty', 0),
                'coverage': agent_result.get('coverage_95', 0),
                'training_data_size': agent_result['training_data_size']
            }
            self.stage_results.append(agent_record)
        
        # Print stage summary focused on R2
        all_r2 = [result['r2_score'] for result in agents_results]
        all_mse = [result['mse'] for result in agents_results]
        avg_r2 = np.mean(all_r2)
        avg_mse = np.mean(all_mse)
        
        print(f"Stage {stage_number} ({stage_name}): R2={avg_r2:.4f}, MSE={avg_mse:.6f} (avg across {len(agents_results)} agents)")
    
    def save_progress_csv(self):
        """Save progress as simple CSV with agent-level details"""
        
        if not self.stage_results:
            return
        
        # Create DataFrame with agent-level records
        progress_df = pd.DataFrame(self.stage_results)
        
        # Save with clean experiment names
        if 'Standard_DAC' in self.experiment_name:
            filename = 'standard_dac.csv'
        elif 'Weighted_DAC_PoE' in self.experiment_name:
            filename = 'weighted_dac_poe.csv'
        else:
            filename = 'progress.csv'
        
        csv_path = os.path.join(self.save_dir, filename)
        progress_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"Stage-by-stage progress saved to: {csv_path}")
        
        # Print R2 improvement summary
        stage_summary = progress_df.groupby('stage_number')['r2_score'].mean()
        initial_r2 = stage_summary.iloc[0]
        final_r2 = stage_summary.iloc[-1]
        improvement = final_r2 - initial_r2
        improvement_pct = (improvement / max(abs(initial_r2), 0.001)) * 100
        
        print(f"R2 Score Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        print(f"Initial R2: {initial_r2:.4f} → Final R2: {final_r2:.4f}")
        print(f"Total records: {len(progress_df)} (stages × agents)")
        
        return csv_path

def evaluate_baseline_vsgp(model, likelihood, test_evaluator, function_type):
    """Evaluate baseline VSGP model - focused on R2 score"""
    
    print("Evaluating baseline VSGP...")
    
    # Load test data
    test_df = pd.read_csv(test_evaluator.test_data_path)
    test_x_cols = [col for col in test_df.columns if col.startswith('x')]
    test_x = torch.tensor(test_df[test_x_cols].values, dtype=torch.float32)
    
    # Use noisy observed values for evaluation
    if 'y_noisy' in test_df.columns:
        test_y_observed = test_df['y_noisy'].values
        print("Using noisy observed values for evaluation (realistic scenario)")
    elif 'y' in test_df.columns:
        test_y_observed = test_df['y'].values
        print("Using y column for evaluation")
    else:
        test_y_observed = test_df['y_true'].values
        print("Warning: Using y_true for evaluation (unrealistic scenario)")
    
    # Set to eval mode
    model.eval()
    likelihood.eval()
    
    # Make predictions
    with torch.no_grad():
        pred_dist = likelihood(model(test_x))
        pred_mean = pred_dist.mean.cpu().numpy()
        pred_var = pred_dist.variance.cpu().numpy()
    
    # Calculate metrics - PRIMARY FOCUS ON R2
    mse = np.mean((pred_mean - test_y_observed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_mean - test_y_observed))
    
    # R² score against observed values (PRIMARY METRIC)
    ss_res = np.sum((test_y_observed - pred_mean) ** 2)
    ss_tot = np.sum((test_y_observed - test_y_observed.mean()) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Supporting metrics
    avg_uncertainty = np.mean(pred_var)
    
    # Calculate coverage
    lower_bound = pred_mean - 1.96 * np.sqrt(pred_var)
    upper_bound = pred_mean + 1.96 * np.sqrt(pred_var)
    coverage = np.mean((test_y_observed >= lower_bound) & (test_y_observed <= upper_bound))
    
    baseline_results = {
        'method': 'Baseline_VSGP',
        'r2_score': r2_score,  # PRIMARY METRIC
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'avg_uncertainty': avg_uncertainty,
        'coverage': coverage,
        'evaluation_type': 'noisy_observed'
    }
    
    # Save baseline results
    baseline_record = {
        'stage_number': 1,
        'stage_name': 'baseline',
        'agent_id': 'baseline_vsgp',
        'r2_score': r2_score,  # PRIMARY METRIC
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'avg_uncertainty': avg_uncertainty,
        'coverage': coverage,
        'training_data_size': 'all_data',
        'evaluation_type': 'noisy_observed'
    }
    
    # Save to simple CSV file
    baseline_dir = f"project/results/{function_type}"
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_csv_path = os.path.join(baseline_dir, f'baseline_vsgp.csv')
    
    baseline_df = pd.DataFrame([baseline_record])
    baseline_df.to_csv(baseline_csv_path, index=False, encoding='utf-8')
    print(f"Baseline VSGP results saved to: {baseline_csv_path}")
    
    # Print results - R2 FOCUS
    print(f"Baseline VSGP Results:")
    print(f"  R2 Score: {r2_score:.6f} (PRIMARY METRIC)")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Coverage 95%: {coverage:.4f}")
    
    return baseline_results

def create_comparison_visualization(comparison_df, save_dir, function_type):
    """Create visualization comparing different methods - R2 FOCUSED"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # R2 Comparison (PRIMARY)
    ax1 = axes[0]
    methods = comparison_df['Method']
    r2_values = comparison_df['R2']
    colors = ['blue', 'red', 'green'][:len(methods)]
    
    bars1 = ax1.bar(methods, r2_values, color=colors, alpha=0.8)
    ax1.set_ylabel('R2 Score')
    ax1.set_title(f'{function_type.title()} - R2 Score Comparison (PRIMARY)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, r2_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # MSE Comparison (SECONDARY)
    ax2 = axes[1]
    mse_values = comparison_df['MSE']
    
    bars2 = ax2.bar(methods, mse_values, color=colors, alpha=0.7)
    ax2.set_ylabel('MSE (log scale)')
    ax2.set_title(f'{function_type.title()} - MSE Comparison (SECONDARY)')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, mse_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.2, 
                f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'{function_type}_r2_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"R2-focused comparison visualization saved to: {plot_path}")

def run_experiment_for_function_type(function_type):
    """Run complete streaming experiment with R2-focused tracking"""
    
    # Get function-specific configuration
    config = FUNCTION_CONFIGS[function_type]
    buffer_size = config['buffer_size']
    init_train_size = config['init_train_size']
    streaming_batch_size = config['streaming_batch_size']
    max_cycles = config['max_cycles']
    
    print(f"""
{'='*100}
TESTING FUNCTION TYPE: {function_type.upper()}
Methods: Standard_DAC vs Weighted_DAC_PoE vs Baseline_VSGP
Stage Pattern: initial -> dac -> retrain -> dac -> retrain -> dac -> ... -> final
PRIMARY METRIC: R2 Score
{'='*100}
""")

    # SIMPLIFIED: Create simple results directory
    save_dir = f"project/results/{function_type}"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print(f"Loading {function_type} training data...")
    agent_data = []
    total_training_samples = 0

    for i in range(NUM_AGENTS):
        data = load_synthetic_agent_data(i, function_type)
        agent_data.append(data)
        total_training_samples += len(data['x'])

    inducing_points = load_synthetic_inducing_points(function_type)
    inducing_points_tensor = torch.tensor(inducing_points, dtype=torch.float32)

    print(f"\n{function_type.upper()} Training Data Summary:")
    print(f"  - Total training samples across all agents: {total_training_samples}")
    print(f"  - Average samples per agent: {total_training_samples // NUM_AGENTS}")
    print(f"  - Inducing points: {len(inducing_points)}")

    # ============================================================================
    # BASELINE VSGP EXPERIMENT
    # ============================================================================
    print(f"\n[{function_type.upper()}] BASELINE VSGP TRAINING")
    print(f"{'='*80}")
    
    # Load full training data
    full_train_x, full_train_y = load_full_training_data(function_type)
    
    # FIXED: Get actual sample count for baseline
    actual_training_samples = len(full_train_x)
    print(f"Baseline VSGP will train on {actual_training_samples} total samples")
    
    # Train baseline VSGP
    baseline_model, baseline_likelihood = train_baseline_vsgp(
        full_train_x, full_train_y, inducing_points_tensor, 
        num_iter=300
    )
    
    # Test data path for baseline evaluation
    test_data_path = f'project/data/synthetic/{function_type}/test.csv'
    if not os.path.exists(test_data_path):
        print(f"ERROR: Test data not found for {function_type}: {test_data_path}")
        return None
    
    # Create dummy evaluator for baseline
    dummy_evaluator = type('obj', (object,), {'test_data_path': test_data_path})()
    
    # Evaluate baseline
    baseline_results = evaluate_baseline_vsgp(baseline_model, baseline_likelihood, dummy_evaluator, function_type)
    
    # FIXED: Add actual sample count to baseline results
    baseline_results['actual_samples_used'] = actual_training_samples
    
    # ============================================================================
    # DISTRIBUTED EXPERIMENTS: Standard DAC vs Weighted DAC
    # ============================================================================
    
    results_comparison = {'baseline_vsgp': baseline_results}
    
    # Clean experiment names matching your requirements
    for experiment_name, use_weighted_dac in [("Standard_DAC", False), ("Weighted_DAC_PoE", True)]:
        
        print(f"\n[{function_type.upper()}] {experiment_name.upper()} EXPERIMENT")
        print(f"{'='*80}")
        
        # Initialize progress tracker with clean naming
        progress_tracker = SimpleProgressTracker(save_dir, experiment_name)
        
        # Initialize test evaluator for this experiment
        evaluator = TestEvaluator(
            test_data_path=test_data_path,
            results_dir=save_dir,
            experiment_name=f"{experiment_name}_{function_type}"
        )
        
        # Initialize agents
        print(f"Initializing agents for {experiment_name}...")
        agents = []
        for i in range(NUM_AGENTS):
            available_samples = len(agent_data[i]['x'])
            actual_init_size = min(init_train_size, available_samples)
            
            train_x = torch.tensor(agent_data[i]['x'][:actual_init_size], dtype=torch.float32)
            train_y = torch.tensor(agent_data[i]['y'][:actual_init_size], dtype=torch.float32)
            neighbors = [(i-1) % NUM_AGENTS, (i+1) % NUM_AGENTS]
            
            agent = Agent(
                agent_id=i,
                inducing_points=inducing_points_tensor,
                train_x=train_x,
                train_y=train_y,
                neighbors=neighbors,
                buffer_size=buffer_size,
                device=DEVICE
            )
            agents.append(agent)

        # Set up DAC (ring topology)
        A = np.zeros((NUM_AGENTS, NUM_AGENTS))
        for i in range(NUM_AGENTS):
            A[i, (i-1) % NUM_AGENTS] = 1
            A[i, (i+1) % NUM_AGENTS] = 1
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        if use_weighted_dac:
            dac = WeightedDACConsensus(L, alpha=0.2, use_uncertainty_weights=True, 
                                     weighting_scheme="enhanced_poe")
            consensus_method = "weighted_poe_enhanced"
        else:
            dac = DACConsensus(L, alpha=0.2)
            consensus_method = "standard"

        # ========================================================================
        # STAGE 1: INITIAL TRAINING
        # ========================================================================
        print(f"\n[{experiment_name}] STAGE 1: INITIAL TRAINING")
        print(f"{'='*60}")

        train_threads = []
        for i, agent in enumerate(agents):
            def train_agent(agent, idx):
                start_time = time.time()
                iterations = 150 if function_type == 'sinusoidal' else 200
                agent.train_local(num_iter=iterations)
                training_time = time.time() - start_time
                print(f"[{experiment_name}] Agent {idx+1} initial training completed in {training_time:.2f}s")
            
            t = threading.Thread(target=train_agent, args=(agent, i))
            t.start()
            train_threads.append(t)

        for t in train_threads:
            t.join()

        # Evaluate Stage 1
        stage_results = evaluator.evaluate_agents(
            agents, 
            stage_name="initial",
            stage_number=1,
            additional_info={
                'experiment_type': experiment_name,
                'consensus_method': consensus_method,
                'function_type': function_type,
                'stage_type': 'initial'
            }
        )
        
        progress_tracker.add_stage(1, "initial", stage_results)

        # ========================================================================
        # STAGE 2: FIRST DAC CONSENSUS
        # ========================================================================
        print(f"\n[{experiment_name}] STAGE 2: FIRST DAC CONSENSUS")
        print(f"{'='*60}")

        consensus_history, converged, actual_steps = run_dac_consensus(
            agents, dac, consensus_steps=10, consensus_type=consensus_method
        )

        stage_results = evaluator.evaluate_agents(
            agents, 
            stage_name="dac",
            stage_number=2,
            additional_info={
                'experiment_type': experiment_name,
                'consensus_method': consensus_method,
                'function_type': function_type,
                'stage_type': 'dac'
            }
        )
        
        progress_tracker.add_stage(2, "dac", stage_results)

        # ========================================================================
        # STREAMING CYCLES: retrain -> dac -> retrain -> dac ...
        # ========================================================================
        print(f"\n[{experiment_name}] STREAMING SIMULATION")
        print(f"Stage pattern: retrain -> dac -> retrain -> dac -> ... -> final")
        print(f"{'='*60}")

        current_data_indices = [init_train_size] * NUM_AGENTS
        stage_number = 3

        for cycle in range(max_cycles):
            print(f"\n[{experiment_name}] === STREAMING CYCLE {cycle+1}/{max_cycles} ===")
            
            # Add new data and check for retraining
            agents_with_available_data = []
            for i in range(NUM_AGENTS):
                if current_data_indices[i] < len(agent_data[i]['x']):
                    agents_with_available_data.append(i)
            
            if not agents_with_available_data:
                print(f"[{experiment_name}] All data consumed early at cycle {cycle+1}.")
                break
            
            # Add new data batch
            agents_with_new_data = []
            for i in agents_with_available_data:
                current_idx = current_data_indices[i]
                end_idx = min(current_idx + streaming_batch_size, len(agent_data[i]['x']))
                
                if current_idx < end_idx:
                    batch_x = torch.tensor(agent_data[i]['x'][current_idx:end_idx], dtype=torch.float32)
                    batch_y = torch.tensor(agent_data[i]['y'][current_idx:end_idx], dtype=torch.float32)
                    
                    with agents[i].lock:
                        agents[i].update_data(batch_x, batch_y)
                    
                    current_data_indices[i] = end_idx
                    agents_with_new_data.append(i)
            
            if not agents_with_new_data:
                break
            
            # Check for retraining
            agents_to_retrain = []
            for i in agents_with_new_data:
                with agents[i].lock:
                    current_size = agents[i].train_x.shape[0]
                    if current_size >= buffer_size:
                        agents_to_retrain.append(i)
            
            # RETRAIN and DAC cycle
            if agents_to_retrain:
                print(f"[{experiment_name}] Cycle {cycle+1}: Retraining {len(agents_to_retrain)} agents")
                
                # RETRAIN STAGE
                train_threads = []
                for agent_id in agents_to_retrain:
                    def retrain_agent(agent, idx):
                        with agent.lock:
                            iterations = 30 if function_type == 'sinusoidal' else 50
                            agent.train_local(num_iter=iterations)
                    
                    t = threading.Thread(target=retrain_agent, args=(agents[agent_id], agent_id))
                    t.start()
                    train_threads.append(t)
                
                for t in train_threads:
                    t.join()
                
                # Evaluate after retraining
                stage_results = evaluator.evaluate_agents(
                    agents, 
                    stage_name="retrain",
                    stage_number=stage_number,
                    additional_info={
                        'experiment_type': experiment_name,
                        'consensus_method': consensus_method,
                        'cycle': cycle + 1,
                        'function_type': function_type,
                        'stage_type': 'retrain'
                    }
                )
                
                progress_tracker.add_stage(stage_number, "retrain", stage_results)
                stage_number += 1
                
                # DAC CONSENSUS STAGE
                consensus_info = run_dac_consensus(agents, dac, consensus_steps=3, consensus_type=consensus_method)
                
                stage_results = evaluator.evaluate_agents(
                    agents, 
                    stage_name="dac",
                    stage_number=stage_number,
                    additional_info={
                        'experiment_type': experiment_name,
                        'consensus_method': consensus_method,
                        'cycle': cycle + 1,
                        'function_type': function_type,
                        'stage_type': 'dac'
                    }
                )
                
                progress_tracker.add_stage(stage_number, "dac", stage_results)
                stage_number += 1

        # ========================================================================
        # FINAL STAGE
        # ========================================================================
        print(f"\n[{experiment_name}] FINAL EVALUATION")
        print(f"{'='*60}")

        final_consensus_info = run_dac_consensus(agents, dac, consensus_steps=10, consensus_type=consensus_method)

        final_results = evaluator.evaluate_agents(
            agents, 
            stage_name="final",
            stage_number=stage_number,
            additional_info={
                'experiment_type': experiment_name,
                'consensus_method': consensus_method,
                'function_type': function_type,
                'stage_type': 'final'
            }
        )
        
        progress_tracker.add_stage(stage_number, "final", final_results)

        # Save progress CSV with clean naming
        csv_path = progress_tracker.save_progress_csv()

        # Store results for comparison - R2 FOCUSED
        all_results = evaluator.evaluation_results
        if all_results:
            df = pd.DataFrame(all_results)
            final_stage_results = df[df['stage_number'] == stage_number]
            
            if len(final_stage_results) > 0:
                avg_r2 = final_stage_results['r2_score'].mean()
                avg_mse = final_stage_results['mse'].mean()
                
                results_comparison[experiment_name.lower()] = {
                    'method': experiment_name,
                    'final_r2': avg_r2,  # PRIMARY METRIC
                    'final_mse': avg_mse,
                    'total_stages': stage_number,
                    'csv_path': csv_path
                }

        evaluator.evaluation_results = []

    # ============================================================================
    # R2-FOCUSED COMPARISON SUMMARY
    # ============================================================================
    print(f"\n[{function_type.upper()}] R2-FOCUSED COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    comparison_df = pd.DataFrame([
        {
            'Method': 'Baseline_VSGP',
            'R2': baseline_results['r2_score'],  # PRIMARY
            'MSE': baseline_results['mse'],       # SECONDARY
            'Coverage': baseline_results['coverage'],
            'Type': 'Centralized'
        }
    ])
    
    for exp_name, results in results_comparison.items():
        if exp_name not in ['baseline_vsgp']:
            comparison_df = pd.concat([comparison_df, pd.DataFrame([{
                'Method': results['method'],
                'R2': results['final_r2'],    # PRIMARY
                'MSE': results['final_mse'],  # SECONDARY
                'Coverage': 'N/A',
                'Type': 'Distributed'
            }])], ignore_index=True)
    
    # Print R2-focused comparison results
    print("R2-Focused Method Comparison Results:")
    print("=" * 50)
    for _, row in comparison_df.iterrows():
        print(f"  {row['Method']}:")
        print(f"    R2 Score: {row['R2']:.6f} (PRIMARY)")
        print(f"    MSE: {row['MSE']:.6f} (secondary)")
        print(f"    Type: {row['Type']}")
        print()
    
    # Save comparison results with R2 focus
    comparison_path = os.path.join(save_dir, f'r2_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
    print(f"R2-focused comparison results saved to: {comparison_path}")
    
    # Create R2-focused visualization
    create_comparison_visualization(comparison_df, save_dir, function_type)
    
    return {
        'function_type': function_type,
        'comparison_results': results_comparison,
        'comparison_df': comparison_df,
        'save_dir': save_dir
    }

def create_overall_comparison(all_results):
    """Create overall R2-focused comparison across all function types"""
    
    print(f"\n{'='*120}")
    print("OVERALL R2-FOCUSED COMPARISON ACROSS ALL FUNCTION TYPES")
    print(f"{'='*120}")
    
    # Aggregate results with R2 focus
    overall_df = []
    
    for func_type, result in all_results.items():
        comparison_df = result['comparison_df']
        
        for _, row in comparison_df.iterrows():
            overall_df.append({
                'Function': func_type.title(),
                'Method': row['Method'],
                'R2': row['R2'],    # PRIMARY METRIC
                'MSE': row['MSE'],  # SECONDARY
                'Type': row['Type']
            })
    
    overall_df = pd.DataFrame(overall_df)
    
    # Print R2-focused summary
    print("\nOverall R2-Focused Results Summary:")
    print("=" * 80)
    
    for method in overall_df['Method'].unique():
        method_data = overall_df[overall_df['Method'] == method]
        avg_r2 = method_data['R2'].mean()
        avg_mse = method_data['MSE'].mean()
        
        print(f"{method}:")
        print(f"  Average R2: {avg_r2:.6f} (PRIMARY)")
        print(f"  Average MSE: {avg_mse:.6f} (secondary)")
        print()
    
    # Save overall results
    overall_path = "project/results/overall_r2_comparison.csv"
    os.makedirs(os.path.dirname(overall_path), exist_ok=True)
    overall_df.to_csv(overall_path, index=False, encoding='utf-8')
    print(f"Overall R2-focused comparison saved to: {overall_path}")
    
    # Create overall R2-focused visualization
    create_overall_r2_visualization(overall_df)

def create_overall_r2_visualization(overall_df):
    """Create R2-focused visualization for overall comparison"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # R2 by function and method (PRIMARY)
    ax1 = axes[0]
    r2_pivot = overall_df.pivot(index='Function', columns='Method', values='R2')
    r2_pivot.plot(kind='bar', ax=ax1, color=['blue', 'red', 'green'])
    ax1.set_title('R2 Score Comparison by Function Type (PRIMARY)')
    ax1.set_ylabel('R2 Score')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Average R2 comparison (PRIMARY)
    ax2 = axes[1]
    avg_performance = overall_df.groupby('Method').agg({'R2': 'mean', 'MSE': 'mean'}).reset_index()
    
    x_pos = np.arange(len(avg_performance))
    bars = ax2.bar(x_pos, avg_performance['R2'], color=['blue', 'red', 'green'][:len(avg_performance)])
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Average R2 Score')
    ax2.set_title('Average R2 Across All Functions (PRIMARY)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(avg_performance['Method'], rotation=45)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, avg_performance['R2']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # MSE comparison (SECONDARY)
    ax3 = axes[2]
    bars = ax3.bar(x_pos, avg_performance['MSE'], color=['blue', 'red', 'green'][:len(avg_performance)])
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Average MSE (log scale)')
    ax3.set_title('Average MSE Across All Functions (SECONDARY)')
    ax3.set_yscale('log')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(avg_performance['Method'], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, avg_performance['MSE']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.2, 
                f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = "project/results/overall_r2_focused_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Overall R2-focused comparison visualization saved to: {plot_path}")

def main_multi_function_experiment():
    """Run comprehensive R2-focused experiments"""
    
    print(f"""
{'='*120}
COMPREHENSIVE R2-FOCUSED DAC COMPARISON EXPERIMENT
Methods: Standard_DAC vs Weighted_DAC_PoE vs Baseline_VSGP
Function Types: {', '.join(FUNCTION_TYPES)}
Stage Pattern: initial -> dac -> retrain -> dac -> retrain -> dac -> ... -> final
PRIMARY METRIC: R2 Score
Saved Files: standard_dac.csv, weighted_dac_poe.csv, baseline_vsgp.csv, r2_comparison.csv
{'='*120}
""")
    
    all_results = {}
    
    for i, function_type in enumerate(FUNCTION_TYPES):
        print(f"\n{'*'*100}")
        print(f"EXPERIMENT {i+1}/{len(FUNCTION_TYPES)}: {function_type.upper()}")
        print(f"{'*'*100}")
        
        try:
            result = run_experiment_for_function_type(function_type)
            if result:
                all_results[function_type] = result
                print(f"SUCCESS: {function_type} experiment completed!")
                
                # Print R2 summary for this function
                comparison_df = result['comparison_df']
                print(f"\n{function_type.upper()} R2 RESULTS:")
                for _, row in comparison_df.iterrows():
                    print(f"  {row['Method']}: R2 = {row['R2']:.6f}")
                    
            else:
                print(f"FAILED: {function_type} experiment failed!")
        except Exception as e:
            print(f"ERROR in {function_type} experiment: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Overall R2-focused comparison across all functions
    if len(all_results) > 1:
        create_overall_comparison(all_results)
    
    # Print final R2 summary
    print(f"\n{'='*120}")
    print("FINAL R2 SUMMARY ACROSS ALL FUNCTIONS")
    print(f"{'='*120}")
    
    all_r2_results = []
    for func_type, result in all_results.items():
        comparison_df = result['comparison_df']
        for _, row in comparison_df.iterrows():
            all_r2_results.append({
                'Function': func_type,
                'Method': row['Method'],
                'R2': row['R2']
            })
    
    if all_r2_results:
        r2_df = pd.DataFrame(all_r2_results)
        method_avg = r2_df.groupby('Method')['R2'].mean().sort_values(ascending=False)
        
        print("Average R2 Scores (Ranked):")
        for method, avg_r2 in method_avg.items():
            print(f"  {method}: {avg_r2:.6f}")
    
    return all_results

if __name__ == "__main__":
    # Set console encoding for Windows
    import sys
    if sys.platform == 'win32':
        import locale
        try:
            import os
            os.environ['PYTHONIOENCODING'] = 'utf-8'
        except:
            pass
    
    # Run the comprehensive R2-focused comparison experiment
    results = main_multi_function_experiment()
    
    print(f"\n{'='*120}")
    print("R2-FOCUSED COMPREHENSIVE EXPERIMENT COMPLETE!")
    print(f"Successful experiments: {len(results)}/{len(FUNCTION_TYPES)}")
    print("Files generated per function:")
    print("  - standard_dac.csv (Standard_DAC stage-by-stage results)")
    print("  - weighted_dac_poe.csv (Weighted_DAC_PoE stage-by-stage results)")
    print("  - baseline_vsgp.csv (Baseline VSGP results)")
    print("  - r2_comparison.csv (R2-focused method comparison)")
    print("  - r2_focused_comparison.png (R2 visualization)")
    print(f"{'='*120}")