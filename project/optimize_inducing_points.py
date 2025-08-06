import numpy as np
import torch
import pandas as pd
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Configuration
NUM_AGENTS = 4
NUM_INDUCING_POINTS = 32  # 32 inducing points per agent
DEVICE = 'cpu'
NUM_ITER = 1000
LEARNING_RATE = 0.01

# CONVERGENCE MONITORING PARAMETERS
CONVERGENCE_WINDOW = 50         # Window size for convergence check
CONVERGENCE_THRESHOLD = 1e-6    # Relative change threshold for convergence
PATIENCE = 150                  # Early stopping patience
MIN_ITERATIONS = 200            # Minimum iterations before checking convergence

print(f"""
{'='*80}
OPTIMIZING INDUCING POINT LOCATIONS WITH VSGP - ENHANCED CONVERGENCE MONITORING
{'='*80}
Configuration:
  - Number of Agents: {NUM_AGENTS}
  - Inducing Points per Agent: {NUM_INDUCING_POINTS}
  - Total Inducing Points: {NUM_AGENTS * NUM_INDUCING_POINTS}
  - Training Iterations: {NUM_ITER}
  - Learning Rate: {LEARNING_RATE}
  - Device: {DEVICE}
  - Learn Inducing Locations: True
  
Convergence Settings:
  - Convergence Window: {CONVERGENCE_WINDOW}
  - Convergence Threshold: {CONVERGENCE_THRESHOLD}
  - Early Stopping Patience: {PATIENCE}
  - Minimum Iterations: {MIN_ITERATIONS}
{'='*80}
""")

# ============================================================================
# VSGP MODEL WITH LEARNABLE INDUCING POINTS
# ============================================================================
class VariationalGPModel(ApproximateGP):
    def __init__(self, inducing_points, learn_inducing_locations=True):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, 
            inducing_points, 
            variational_distribution, 
            learn_inducing_locations=learn_inducing_locations
        )
        super(VariationalGPModel, self).__init__(variational_strategy)
        
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ============================================================================
# CONVERGENCE MONITORING FUNCTIONS
# ============================================================================
def check_convergence(losses, window_size=CONVERGENCE_WINDOW, threshold=CONVERGENCE_THRESHOLD):
    """Check if training has converged based on loss history"""
    if len(losses) < window_size:
        return False, 0.0
    
    recent_losses = losses[-window_size:]
    
    # Method 1: Relative change between first and second half of window
    mid_point = window_size // 2
    first_half_avg = np.mean(recent_losses[:mid_point])
    second_half_avg = np.mean(recent_losses[mid_point:])
    
    if abs(first_half_avg) > 1e-10:
        relative_change = abs(second_half_avg - first_half_avg) / abs(first_half_avg)
    else:
        relative_change = abs(second_half_avg - first_half_avg)
    
    # Method 2: Standard deviation of recent losses
    loss_std = np.std(recent_losses)
    loss_mean = abs(np.mean(recent_losses))
    normalized_std = loss_std / (loss_mean + 1e-10)
    
    # Method 3: Slope of recent trend
    x = np.arange(len(recent_losses))
    slope = np.polyfit(x, recent_losses, 1)[0]
    normalized_slope = abs(slope) / (loss_mean + 1e-10)
    
    # Combined convergence criteria
    converged = (relative_change < threshold and 
                normalized_std < threshold * 10 and 
                normalized_slope < threshold * 5)
    
    return converged, relative_change

def track_inducing_point_movement(model, iteration, tracking_data):
    """Track movement of inducing points during optimization"""
    current_inducing = model.variational_strategy.inducing_points.detach().cpu().numpy()
    
    if iteration == 0:
        tracking_data['initial_inducing'] = current_inducing.copy()
        tracking_data['inducing_history'] = [current_inducing.copy()]
        tracking_data['movement_distances'] = []
    else:
        # Calculate movement from previous iteration
        prev_inducing = tracking_data['inducing_history'][-1]
        distances = np.linalg.norm(current_inducing - prev_inducing, axis=1)
        avg_movement = np.mean(distances)
        max_movement = np.max(distances)
        
        tracking_data['movement_distances'].append({
            'iteration': iteration,
            'avg_movement': avg_movement,
            'max_movement': max_movement,
            'total_movement': np.sum(distances)
        })
        
        # Store current positions (every 10 iterations to save memory)
        if iteration % 10 == 0:
            tracking_data['inducing_history'].append(current_inducing.copy())

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_agent_data(agent_idx):
    """Load training data for a specific agent"""
    df = pd.read_csv(f'project/data/KIN40K_train_agent{agent_idx+1}.csv')
    x_cols = [col for col in df.columns if col.startswith('x')]
    x = df[x_cols].values
    y = df['y'].values
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_initial_inducing_points(train_x, num_inducing):
    """Initialize inducing points using K-means clustering"""
    from sklearn.cluster import KMeans
    
    # Use K-means to find good initial inducing point locations
    kmeans = KMeans(n_clusters=num_inducing, random_state=42, n_init=10)
    kmeans.fit(train_x.numpy())
    inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    
    return inducing_points

# ============================================================================
# ENHANCED TRAINING FUNCTION WITH CONVERGENCE MONITORING
# ============================================================================
def train_vsgp_for_agent(agent_idx, train_x, train_y, num_inducing, num_iter):
    """Train VSGP model with comprehensive convergence monitoring"""
    
    print(f"\nTraining VSGP for Agent {agent_idx+1}...")
    print(f"  - Training data: {len(train_x)} samples")
    print(f"  - Input dimension: {train_x.shape[1]}")
    print(f"  - Inducing points: {num_inducing}")
    
    # Initialize inducing points
    initial_inducing = get_initial_inducing_points(train_x, num_inducing)
    print(f"  - Initial inducing points shape: {initial_inducing.shape}")
    
    # Create model and likelihood
    model = VariationalGPModel(initial_inducing, learn_inducing_locations=True)
    likelihood = GaussianLikelihood()
    
    # Move to device
    model = model.to(DEVICE)
    likelihood = likelihood.to(DEVICE)
    train_x = train_x.to(DEVICE)
    train_y = train_y.to(DEVICE)
    
    # Set to training mode
    model.train()
    likelihood.train()
    
    # Set up optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=LEARNING_RATE)
    
    # Set up loss function
    mll = VariationalELBO(likelihood, model, num_data=train_y.size(0))
    
    # Initialize tracking variables
    losses = []
    hyperparameter_history = []
    inducing_tracking = {}
    convergence_info = {
        'converged': False,
        'convergence_iteration': None,
        'convergence_reason': 'max_iterations_reached',
        'final_relative_change': None
    }
    
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"  - Starting training with convergence monitoring...")
    training_start_time = time.time()
    
    for i in range(num_iter):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(train_x)
        loss = -mll(output, train_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        losses.append(current_loss)
        
        # Track inducing point movement
        track_inducing_point_movement(model, i, inducing_tracking)
        
        # Track hyperparameters every 50 iterations
        if i % 50 == 0:
            with torch.no_grad():
                lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().item()
                outputscale = model.covar_module.outputscale.detach().cpu().item()
                noise = likelihood.noise.detach().cpu().item()
                
                hyperparameter_history.append({
                    'iteration': i,
                    'lengthscale': lengthscale,
                    'outputscale': outputscale,
                    'noise': noise,
                    'loss': current_loss
                })
        
        # Check for improvement (for early stopping)
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Check convergence (after minimum iterations)
        if i >= MIN_ITERATIONS:
            converged, relative_change = check_convergence(losses)
            
            if converged:
                convergence_info.update({
                    'converged': True,
                    'convergence_iteration': i,
                    'convergence_reason': 'loss_converged',
                    'final_relative_change': relative_change
                })
                print(f"    CONVERGED at iteration {i+1}!")
                print(f"    Relative change: {relative_change:.2e} < {CONVERGENCE_THRESHOLD:.2e}")
                break
        
        # Early stopping check
        if patience_counter >= PATIENCE:
            convergence_info.update({
                'converged': False,
                'convergence_iteration': i,
                'convergence_reason': 'early_stopping',
                'final_relative_change': relative_change if i >= MIN_ITERATIONS else None
            })
            print(f"    EARLY STOPPING at iteration {i+1}")
            print(f"    No improvement for {PATIENCE} iterations")
            break
        
        # Print progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - training_start_time
            if len(losses) >= CONVERGENCE_WINDOW:
                _, relative_change = check_convergence(losses)
                avg_movement = np.mean([d['avg_movement'] for d in inducing_tracking['movement_distances'][-10:]])
                print(f"    Iter {i+1:4d}/{num_iter} | Loss: {current_loss:.6f} | "
                      f"Change: {relative_change:.2e} | Movement: {avg_movement:.4f} | Time: {elapsed:.1f}s")
            else:
                print(f"    Iter {i+1:4d}/{num_iter} | Loss: {current_loss:.6f} | "
                      f"Time: {elapsed:.1f}s")
    
    training_time = time.time() - training_start_time
    final_iterations = len(losses)
    
    # Final convergence assessment
    if not convergence_info['converged'] and convergence_info['convergence_reason'] == 'max_iterations_reached':
        _, final_relative_change = check_convergence(losses)
        convergence_info['final_relative_change'] = final_relative_change
        
        if final_relative_change < CONVERGENCE_THRESHOLD * 5:  # Relaxed threshold
            print(f"  - Training completed - LIKELY CONVERGED")
            print(f"    Final relative change: {final_relative_change:.2e}")
        else:
            print(f"  - Training completed - MAY NOT BE CONVERGED!")
            print(f"    Final relative change: {final_relative_change:.2e}")
            print(f"    Consider increasing NUM_ITER or decreasing LEARNING_RATE")
    
    print(f"  - Training completed in {training_time:.2f}s ({final_iterations} iterations)")
    print(f"  - Final loss: {losses[-1]:.6f}")
    print(f"  - Best loss: {best_loss:.6f}")
    
    # Extract optimized inducing points
    optimized_inducing = model.variational_strategy.inducing_points.detach().cpu()
    
    # Get final model statistics
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        inducing_pred = model(model.variational_strategy.inducing_points)
        inducing_mean = inducing_pred.mean.detach().cpu().numpy()
        inducing_variance = inducing_pred.variance.detach().cpu().numpy()
        
        lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        outputscale = model.covar_module.outputscale.detach().cpu().numpy()
        noise = likelihood.noise.detach().cpu().numpy()
    
    # Calculate inducing point movement statistics
    if inducing_tracking['movement_distances']:
        total_movement = sum(d['total_movement'] for d in inducing_tracking['movement_distances'])
        avg_total_movement = total_movement / len(inducing_tracking['movement_distances'])
        max_single_movement = max(d['max_movement'] for d in inducing_tracking['movement_distances'])
    else:
        avg_total_movement = 0.0
        max_single_movement = 0.0
    
    training_stats = {
        'agent_id': agent_idx,
        'final_loss': losses[-1],
        'best_loss': best_loss,
        'training_time': training_time,
        'iterations_used': final_iterations,
        'converged': convergence_info['converged'],
        'convergence_iteration': convergence_info.get('convergence_iteration'),
        'convergence_reason': convergence_info['convergence_reason'],
        'final_relative_change': convergence_info.get('final_relative_change'),
        'lengthscale': lengthscale,
        'outputscale': outputscale,
        'noise': noise,
        'inducing_mean_min': inducing_mean.min(),
        'inducing_mean_max': inducing_mean.max(),
        'inducing_mean_std': inducing_mean.std(),
        'inducing_var_min': inducing_variance.min(),
        'inducing_var_max': inducing_variance.max(),
        'inducing_var_mean': inducing_variance.mean(),
        'avg_inducing_movement': avg_total_movement,
        'max_inducing_movement': max_single_movement
    }
    
    # Store detailed tracking data for plotting
    detailed_tracking = {
        'losses': losses,
        'hyperparameter_history': hyperparameter_history,
        'inducing_movement': inducing_tracking['movement_distances'],
        'convergence_info': convergence_info
    }
    
    return optimized_inducing, inducing_mean, training_stats, detailed_tracking

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def create_convergence_plots(agent_idx, detailed_tracking, save_dir):
    """Create comprehensive convergence plots for an agent"""
    
    losses = detailed_tracking['losses']
    hyperparams = detailed_tracking['hyperparameter_history']
    movements = detailed_tracking['inducing_movement']
    convergence_info = detailed_tracking['convergence_info']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Agent {agent_idx+1} - VSGP Optimization Convergence Analysis', fontsize=16, fontweight='bold')
    
    # 1. Loss convergence
    ax1 = axes[0, 0]
    iterations = range(len(losses))
    ax1.plot(iterations, losses, 'b-', linewidth=1, alpha=0.7, label='Loss')
    
    # Add smoothed trend line
    if len(losses) > 50:
        smoothed = savgol_filter(losses, min(51, len(losses)//4*2+1), 3)
        ax1.plot(iterations, smoothed, 'r-', linewidth=2, label='Smoothed Trend')
    
    # Mark convergence point
    if convergence_info['converged'] and convergence_info['convergence_iteration']:
        conv_iter = convergence_info['convergence_iteration']
        ax1.axvline(conv_iter, color='green', linestyle='--', linewidth=2, 
                   label=f'Converged at iter {conv_iter}')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Negative ELBO (Loss)')
    ax1.set_title('Loss Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss convergence (last 200 iterations for detail)
    ax2 = axes[0, 1]
    start_idx = max(0, len(losses) - 200)
    recent_losses = losses[start_idx:]
    recent_iterations = range(start_idx, len(losses))
    
    ax2.plot(recent_iterations, recent_losses, 'b-', linewidth=1)
    if len(recent_losses) > 20:
        smoothed_recent = savgol_filter(recent_losses, min(21, len(recent_losses)//4*2+1), 3)
        ax2.plot(recent_iterations, smoothed_recent, 'r-', linewidth=2)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Negative ELBO (Loss)')
    ax2.set_title('Loss Convergence (Last 200 Iterations)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Hyperparameter evolution
    ax3 = axes[0, 2]
    if hyperparams:
        hyper_iters = [h['iteration'] for h in hyperparams]
        lengthscales = [h['lengthscale'] for h in hyperparams]
        outputscales = [h['outputscale'] for h in hyperparams]
        noises = [h['noise'] for h in hyperparams]
        
        ax3_twin1 = ax3.twinx()
        ax3_twin2 = ax3.twinx()
        ax3_twin2.spines['right'].set_position(('outward', 60))
        
        line1 = ax3.plot(hyper_iters, lengthscales, 'b-', linewidth=2, label='Lengthscale')
        line2 = ax3_twin1.plot(hyper_iters, outputscales, 'r-', linewidth=2, label='Outputscale')
        line3 = ax3_twin2.plot(hyper_iters, noises, 'g-', linewidth=2, label='Noise')
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Lengthscale', color='b')
        ax3_twin1.set_ylabel('Outputscale', color='r')
        ax3_twin2.set_ylabel('Noise', color='g')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper right')
    
    ax3.set_title('Hyperparameter Evolution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Inducing point movement
    ax4 = axes[1, 0]
    if movements:
        move_iters = [m['iteration'] for m in movements]
        avg_movements = [m['avg_movement'] for m in movements]
        max_movements = [m['max_movement'] for m in movements]
        
        ax4.plot(move_iters, avg_movements, 'b-', linewidth=2, label='Average Movement')
        ax4.plot(move_iters, max_movements, 'r-', linewidth=2, label='Max Movement')
        ax4.set_yscale('log')
        ax4.legend()
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Inducing Point Movement (log scale)')
    ax4.set_title('Inducing Point Movement During Optimization')
    ax4.grid(True, alpha=0.3)
    
    # 5. Convergence metrics
    ax5 = axes[1, 1]
    if len(losses) >= CONVERGENCE_WINDOW:
        conv_checks = []
        relative_changes = []
        
        for i in range(CONVERGENCE_WINDOW, len(losses)):
            _, rel_change = check_convergence(losses[:i+1])
            conv_checks.append(i)
            relative_changes.append(rel_change)
        
        ax5.semilogy(conv_checks, relative_changes, 'b-', linewidth=2, label='Relative Change')
        ax5.axhline(CONVERGENCE_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                   label=f'Convergence Threshold ({CONVERGENCE_THRESHOLD:.1e})')
        ax5.legend()
    
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Relative Change (log scale)')
    ax5.set_title('Convergence Metric Evolution')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
CONVERGENCE SUMMARY:

Status: {'CONVERGED' if convergence_info['converged'] else 'NOT CONVERGED'}
Reason: {convergence_info['convergence_reason']}
Final Iteration: {len(losses)}
"""
    
    if convergence_info['convergence_iteration']:
        summary_text += f"Convergence Iteration: {convergence_info['convergence_iteration']}\n"
    
    if convergence_info['final_relative_change']:
        summary_text += f"Final Relative Change: {convergence_info['final_relative_change']:.2e}\n"
    
    summary_text += f"""
Final Loss: {losses[-1]:.6f}
Loss Improvement: {((losses[0] - losses[-1]) / abs(losses[0]) * 100):.2f}%

HYPERPARAMETERS:
"""
    if hyperparams:
        final_hyper = hyperparams[-1]
        summary_text += f"Lengthscale: {final_hyper['lengthscale']:.4f}\n"
        summary_text += f"Outputscale: {final_hyper['outputscale']:.4f}\n"
        summary_text += f"Noise: {final_hyper['noise']:.6f}\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f'{save_dir}/agent_{agent_idx+1}_convergence_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Convergence plot saved: {plot_path}")
    return plot_path

def create_summary_convergence_plot(all_detailed_tracking, save_dir):
    """Create summary plot comparing convergence across all agents"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Agent VSGP Optimization - Convergence Comparison', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # 1. Loss evolution comparison
    ax1 = axes[0, 0]
    for agent_idx, tracking in enumerate(all_detailed_tracking):
        losses = tracking['losses']
        ax1.plot(range(len(losses)), losses, color=colors[agent_idx], 
                linewidth=2, label=f'Agent {agent_idx+1}', alpha=0.8)
        
        # Mark convergence
        conv_info = tracking['convergence_info']
        if conv_info['converged'] and conv_info['convergence_iteration']:
            ax1.axvline(conv_info['convergence_iteration'], color=colors[agent_idx], 
                       linestyle='--', alpha=0.6)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Negative ELBO (Loss)')
    ax1.set_title('Loss Evolution - All Agents')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Convergence status summary
    ax2 = axes[0, 1]
    convergence_status = []
    final_iterations = []
    
    for agent_idx, tracking in enumerate(all_detailed_tracking):
        conv_info = tracking['convergence_info']
        convergence_status.append(1 if conv_info['converged'] else 0)
        final_iterations.append(len(tracking['losses']))
    
    agents = [f'Agent {i+1}' for i in range(len(all_detailed_tracking))]
    bars = ax2.bar(agents, convergence_status, color=['green' if s else 'red' for s in convergence_status])
    
    ax2.set_ylabel('Converged (1) / Not Converged (0)')
    ax2.set_title('Convergence Status by Agent')
    ax2.set_ylim(0, 1.2)
    
    # Add iteration counts on bars
    for i, (bar, iterations) in enumerate(zip(bars, final_iterations)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{iterations} iters', ha='center', va='bottom', fontsize=10)
    
    # 3. Final loss comparison
    ax3 = axes[1, 0]
    final_losses = [tracking['losses'][-1] for tracking in all_detailed_tracking]
    bars = ax3.bar(agents, final_losses, color=colors[:len(all_detailed_tracking)])
    
    ax3.set_ylabel('Final Loss (Negative ELBO)')
    ax3.set_title('Final Loss by Agent')
    
    # Add values on bars
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Inducing point movement summary
    ax4 = axes[1, 1]
    avg_movements = []
    
    for tracking in all_detailed_tracking:
        movements = tracking['inducing_movement']
        if movements:
            avg_movement = np.mean([m['avg_movement'] for m in movements])
            avg_movements.append(avg_movement)
        else:
            avg_movements.append(0)
    
    bars = ax4.bar(agents, avg_movements, color=colors[:len(all_detailed_tracking)])
    ax4.set_ylabel('Average Inducing Point Movement')
    ax4.set_title('Average Inducing Point Movement by Agent')
    
    # Add values on bars
    for bar, movement in zip(bars, avg_movements):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{movement:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f'{save_dir}/summary_convergence_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary convergence plot saved: {plot_path}")
    return plot_path

# ============================================================================
# MAIN OPTIMIZATION PROCESS
# ============================================================================
def optimize_all_inducing_points():
    """Optimize inducing points for all agents with convergence monitoring"""
    
    print(f"\nStarting inducing point optimization for all {NUM_AGENTS} agents...")
    
    all_optimized_data = []
    all_training_stats = []
    all_detailed_tracking = []
    
    # Create plots directory
    plots_dir = 'project/data/optimization_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    for agent_idx in range(NUM_AGENTS):
        agent_start_time = time.time()
        
        # Load agent data
        print(f"\n{'='*50}")
        print(f"PROCESSING AGENT {agent_idx+1}/{NUM_AGENTS}")
        print(f"{'='*50}")
        
        train_x, train_y = load_agent_data(agent_idx)
        print(f"Loaded {len(train_x)} training samples for Agent {agent_idx+1}")
        
        # Train VSGP and get optimized inducing points
        optimized_inducing, inducing_predictions, stats, detailed_tracking = train_vsgp_for_agent(
            agent_idx, train_x, train_y, NUM_INDUCING_POINTS, NUM_ITER
        )
        
        # Create convergence plots for this agent
        print(f"  Creating convergence plots for Agent {agent_idx+1}...")
        create_convergence_plots(agent_idx, detailed_tracking, plots_dir)
        
        # Store results
        for i in range(NUM_INDUCING_POINTS):
            inducing_point_data = {
                'agent_idx': agent_idx,
                'inducing_idx': i,
                'y': inducing_predictions[i]
            }
            
            # Add x coordinates
            for dim_idx in range(optimized_inducing.shape[1]):
                inducing_point_data[f'x{dim_idx}'] = optimized_inducing[i, dim_idx].item()
            
            all_optimized_data.append(inducing_point_data)
        
        all_training_stats.append(stats)
        all_detailed_tracking.append(detailed_tracking)
        
        agent_time = time.time() - agent_start_time
        print(f"Agent {agent_idx+1} completed in {agent_time:.2f}s")
    
    # Create summary convergence plot
    print(f"\nCreating summary convergence analysis...")
    create_summary_convergence_plot(all_detailed_tracking, plots_dir)
    
    return all_optimized_data, all_training_stats, all_detailed_tracking

# ============================================================================
# ENHANCED SAVE RESULTS WITH CONVERGENCE DATA
# ============================================================================
def save_optimized_results(optimized_data, training_stats, detailed_tracking):
    """Save optimized inducing points and comprehensive training statistics"""
    
    # Create data directory if it doesn't exist
    os.makedirs('project/data', exist_ok=True)
    
    # Save optimized inducing points
    inducing_df = pd.DataFrame(optimized_data)
    
    # Reorder columns to match original format
    x_cols = [col for col in inducing_df.columns if col.startswith('x')]
    x_cols.sort(key=lambda x: int(x[1:]))
    
    column_order = ['agent_idx', 'inducing_idx'] + x_cols + ['y']
    inducing_df = inducing_df[column_order]
    
    # Save to CSV
    output_path = 'project/data/KIN40K_inducing_optimized.csv'
    inducing_df.to_csv(output_path, index=False)
    
    print(f"\nOptimized inducing points saved to: {output_path}")
    print(f"Shape: {inducing_df.shape}")
    print(f"Columns: {list(inducing_df.columns)}")
    
    # Save enhanced training statistics
    stats_df = pd.DataFrame(training_stats)
    stats_path = 'project/data/KIN40K_inducing_optimization_stats.csv'
    stats_df.to_csv(stats_path, index=False)
    
    print(f"Training statistics saved to: {stats_path}")
    
    # Save detailed convergence data
    convergence_summary = []
    for agent_idx, tracking in enumerate(detailed_tracking):
        conv_info = tracking['convergence_info']
        summary = {
            'agent_id': agent_idx,
            'converged': conv_info['converged'],
            'convergence_iteration': conv_info.get('convergence_iteration'),
            'convergence_reason': conv_info['convergence_reason'],
            'final_relative_change': conv_info.get('final_relative_change'),
            'total_iterations': len(tracking['losses']),
            'initial_loss': tracking['losses'][0],
            'final_loss': tracking['losses'][-1],
            'loss_improvement_percent': ((tracking['losses'][0] - tracking['losses'][-1]) / abs(tracking['losses'][0])) * 100
        }
        convergence_summary.append(summary)
    
    conv_df = pd.DataFrame(convergence_summary)
    conv_path = 'project/data/KIN40K_convergence_summary.csv'
    conv_df.to_csv(conv_path, index=False)
    
    print(f"Convergence summary saved to: {conv_path}")
    
    # Print comprehensive analysis
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY WITH CONVERGENCE ANALYSIS")
    print(f"{'='*80}")
    
    print(f"Total inducing points optimized: {len(optimized_data)}")
    print(f"Inducing points per agent: {NUM_INDUCING_POINTS}")
    
    # Convergence statistics
    converged_agents = conv_df['converged'].sum()
    print(f"\nCONVERGENCE RESULTS:")
    print(f"  Agents converged: {converged_agents}/{NUM_AGENTS}")
    print(f"  Overall convergence rate: {converged_agents/NUM_AGENTS*100:.1f}%")
    
    if converged_agents > 0:
        converged_stats = conv_df[conv_df['converged']]
        avg_conv_iter = converged_stats['convergence_iteration'].mean()
        print(f"  Average convergence iteration: {avg_conv_iter:.1f}")
    
    # Loss improvement statistics
    avg_improvement = conv_df['loss_improvement_percent'].mean()
    print(f"  Average loss improvement: {avg_improvement:.2f}%")
    
    # Statistics by agent
    for agent_idx in range(NUM_AGENTS):
        agent_data = inducing_df[inducing_df['agent_idx'] == agent_idx]
        agent_stats = stats_df[stats_df['agent_id'] == agent_idx].iloc[0]
        agent_conv = conv_df[conv_df['agent_id'] == agent_idx].iloc[0]
        
        print(f"\nAgent {agent_idx+1}:")
        print(f"  - Converged: {'YES' if agent_conv['converged'] else 'NO'}")
        if agent_conv['converged']:
            print(f"  - Convergence iteration: {agent_conv['convergence_iteration']}")
        print(f"  - Convergence reason: {agent_conv['convergence_reason']}")
        print(f"  - Total iterations: {agent_conv['total_iterations']}")
        print(f"  - Loss improvement: {agent_conv['loss_improvement_percent']:.2f}%")
        print(f"  - Final loss: {agent_stats['final_loss']:.6f}")
        print(f"  - Training time: {agent_stats['training_time']:.2f}s")
        print(f"  - Final lengthscale: {agent_stats['lengthscale']}")
        print(f"  - Final outputscale: {agent_stats['outputscale']:.4f}")
        print(f"  - Final noise: {agent_stats['noise']:.6f}")
        
        # Inducing point statistics
        print(f"  - Y range: [{agent_data['y'].min():.4f}, {agent_data['y'].max():.4f}]")
        if agent_stats['avg_inducing_movement'] > 0:
            print(f"  - Avg inducing movement: {agent_stats['avg_inducing_movement']:.6f}")
            print(f"  - Max inducing movement: {agent_stats['max_inducing_movement']:.6f}")
    
    return output_path, stats_path, conv_path

# ============================================================================
# COMPARISON WITH ORIGINAL INDUCING POINTS
# ============================================================================
def compare_with_original():
    """Compare optimized inducing points with original ones"""
    
    try:
        original_df = pd.read_csv('project/data/KIN40K_inducing_all_agents.csv')
        optimized_df = pd.read_csv('project/data/KIN40K_inducing_optimized.csv')
        
        print(f"\n{'='*60}")
        print("COMPARISON WITH ORIGINAL INDUCING POINTS")
        print(f"{'='*60}")
        
        print(f"Original inducing points: {len(original_df)}")
        print(f"Optimized inducing points: {len(optimized_df)}")
        
        # Compare by agent
        for agent_idx in range(NUM_AGENTS):
            original_agent = original_df[original_df['agent_idx'] == agent_idx]
            optimized_agent = optimized_df[optimized_df['agent_idx'] == agent_idx]
            
            print(f"\nAgent {agent_idx+1}:")
            print(f"  Original: {len(original_agent)} points")
            print(f"  Optimized: {len(optimized_agent)} points")
            
            if len(original_agent) > 0:
                print(f"  Original Y range: [{original_agent['y'].min():.4f}, {original_agent['y'].max():.4f}]")
                print(f"  Optimized Y range: [{optimized_agent['y'].min():.4f}, {optimized_agent['y'].max():.4f}]")
                print(f"  Original Y mean: {original_agent['y'].mean():.4f}")
                print(f"  Optimized Y mean: {optimized_agent['y'].mean():.4f}")
        
    except FileNotFoundError:
        print("\nOriginal inducing points file not found. Skipping comparison.")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    print(f"Starting optimization at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run optimization with convergence monitoring
    optimized_data, training_stats, detailed_tracking = optimize_all_inducing_points()
    
    # Save results with convergence data
    output_path, stats_path, conv_path = save_optimized_results(optimized_data, training_stats, detailed_tracking)
    
    # Compare with original
    compare_with_original()
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("INDUCING POINT OPTIMIZATION COMPLETED WITH CONVERGENCE MONITORING")
    print(f"{'='*80}")
    print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Optimized inducing points saved to: {output_path}")
    print(f"Training statistics saved to: {stats_path}")
    print(f"Convergence summary saved to: {conv_path}")
    print(f"Convergence plots saved to: project/data/optimization_plots/")
    print(f"Optimization completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")