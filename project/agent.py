import threading
import torch
import gpytorch
import numpy as np
from gp_model import VSGPRegressionModel
from train import train_gp

class Agent:
    def __init__(self, agent_id, inducing_points, train_x, train_y, neighbors, buffer_size=1000, device='cpu'):
        self.id = agent_id
        self.device = device
        # Pass inducing_points as torch tensor directly
        self.model = VSGPRegressionModel(inducing_points=inducing_points, device=device).to(device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.train_x = train_x.to(device)
        self.train_y = train_y.to(device)
        self.neighbors = neighbors
        self.lock = threading.Lock()
        self.ready_for_dac = threading.Event()
        self.consensus_mean = None
        self.consensus_var = None
        self.buffer_size = buffer_size
        self.data_buffer_x = [self.train_x]
        self.data_buffer_y = [self.train_y]
        self.loss_history = []
        self.hypers = None
        self.evaluation_history = []  # Store evaluation results
        self.test_data = None  # Will be loaded when needed

    def train_local(self, num_iter=100):
        # Use the updated train_gp function
        self.model, self.likelihood, self.loss_history, self.hypers = train_gp(
            self.model, self.likelihood, self.train_x, self.train_y, num_iter, self.device, 
        )

    def predict_mean_and_var(self, test_x):
        # Always inject consensus before prediction if available
        if self.consensus_mean is not None and self.consensus_var is not None:
            self.inject_consensus_to_variational()
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            pred = self.model(test_x.to(self.device))
            mean = pred.mean.cpu().numpy()
            var = pred.variance.cpu().numpy()
        return mean, var

    def update_data(self, new_x, new_y):
        self.data_buffer_x.append(new_x.to(self.device))
        self.data_buffer_y.append(new_y.to(self.device))
        self.train_x = torch.cat(self.data_buffer_x)
        self.train_y = torch.cat(self.data_buffer_y)
        # If buffer is full, retrain local model and signal ready for DAC
        if self.train_x.shape[0] >= self.buffer_size:
            self.train_local()
            self.ready_for_dac.set()

    def inject_consensus_to_variational(self):
        M = self.model.variational_strategy.inducing_points.size(0)
        device = self.device
        # Ensure consensus_mean is shape (M,)
        mean = np.asarray(self.consensus_mean).reshape(-1)
        assert mean.shape[0] == M, f"consensus_mean shape {mean.shape} does not match #inducing {M}"
        self.model.variational_strategy._variational_distribution.variational_mean.data = \
            torch.tensor(mean, dtype=torch.float32, device=device)
        # Ensure consensus_var is (M,) or (M, M)
        var = np.asarray(self.consensus_var)
        if var.ndim == 1:
            assert var.shape[0] == M, f"consensus_var shape {var.shape} does not match #inducing {M}"
            chol = torch.diag(torch.sqrt(torch.tensor(var, dtype=torch.float32, device=device)))
        else:
            assert var.shape == (M, M), f"consensus_var shape {var.shape} does not match ({M},{M})"
            chol = torch.linalg.cholesky(torch.tensor(var, dtype=torch.float32, device=device))
        self.model.variational_strategy._variational_distribution.chol_variational_covar.data = chol

    def evaluate_rmse(self, test_x, test_y):
        return self.model.evaluate_rmse(self.likelihood, test_x, test_y)

    def evaluate_nll(self, test_x, test_y):
        return self.model.evaluate_nll(self.likelihood, test_x, test_y)
    
    def load_test_data(self, test_data_path='dataset/KIN40K/KIN40K_test.mat'):
        """Load test data for evaluation (shared across all agents)"""
        if self.test_data is None:
            try:
                import scipy.io
                mat_data = scipy.io.loadmat(test_data_path)
                
                print(f"Agent {self.id}: Loading test data from {test_data_path}")
                print(f"Available keys in .mat file: {[k for k in mat_data.keys() if not k.startswith('__')]}")
                
                # Extract test data - Updated for KIN40K_test.mat structure
                if 'x' in mat_data and 'y' in mat_data:
                    test_x = mat_data['x']  # Shape: (10000, 8)
                    test_y = mat_data['y']  # Shape: (10000, 1)
                    print(f"Agent {self.id}: Found 'x' and 'y' keys")
                else:
                    # Fallback: try to find the largest arrays
                    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                    print(f"Agent {self.id}: Fallback mode - available keys: {data_keys}")
                    
                    if len(data_keys) >= 2:
                        # Sort by array size to get the main data arrays
                        key_sizes = [(k, mat_data[k].size if hasattr(mat_data[k], 'size') else 0) for k in data_keys]
                        key_sizes.sort(key=lambda x: x[1], reverse=True)
                        
                        test_x = mat_data[key_sizes[0][0]]
                        test_y = mat_data[key_sizes[1][0]]
                        print(f"Agent {self.id}: Using keys '{key_sizes[0][0]}' and '{key_sizes[1][0]}'")
                    else:
                        raise KeyError("Could not identify test data arrays")
                
                # Ensure correct shapes and types
                test_x = np.asarray(test_x, dtype=np.float32)
                test_y = np.asarray(test_y, dtype=np.float32)
                
                # Handle y dimension - flatten if it's (N, 1)
                if test_y.ndim == 2 and test_y.shape[1] == 1:
                    test_y = test_y.flatten()
                
                self.test_data = {
                    'x': torch.tensor(test_x, dtype=torch.float32),
                    'y': torch.tensor(test_y, dtype=torch.float32)
                }
                
                print(f"Agent {self.id}: Test data loaded - X: {test_x.shape}, Y: {test_y.shape}")
                
            except Exception as e:
                print(f"Agent {self.id}: Error loading test data: {e}")
                raise
        
        return self.test_data
    
    def evaluate_on_test_data(self, stage_name, stage_number, batch_size=1000):
        """Comprehensive evaluation on test data"""
        import time
        import numpy as np
        
        # Load test data if not already loaded
        if self.test_data is None:
            self.load_test_data()
        
        test_x = self.test_data['x']
        test_y = self.test_data['y']
        
        print(f"Agent {self.id+1}: Evaluating on {len(test_x)} test samples at stage: {stage_name}")
        
        self.model.eval()
        self.likelihood.eval()
        
        predictions = []
        variances = []
        eval_start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, len(test_x), batch_size):
                batch_x = test_x[i:i+batch_size].to(self.device)
                
                # Get posterior distribution
                posterior = self.model(batch_x)
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
        
        # Basic metrics
        mse = torch.mean((test_y_flat - pred_flat) ** 2).item()
        mae = torch.mean(torch.abs(test_y_flat - pred_flat)).item()
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = torch.sum((test_y_flat - pred_flat) ** 2).item()
        ss_tot = torch.sum((test_y_flat - torch.mean(test_y_flat)) ** 2).item()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Uncertainty metrics
        mean_uncertainty = torch.mean(torch.sqrt(all_variances)).item()
        max_error = torch.max(torch.abs(test_y_flat - pred_flat)).item()
        
        # Prediction intervals coverage (95%) - FIXED
        pred_std = torch.sqrt(all_variances).flatten()
        
        # Handle potential NaN/Inf values
        finite_mask = torch.isfinite(test_y_flat) & torch.isfinite(pred_flat) & torch.isfinite(pred_std)
        
        if finite_mask.any():
            finite_y = test_y_flat[finite_mask]
            finite_pred = pred_flat[finite_mask]
            finite_std = pred_std[finite_mask]
            
            lower_bound = finite_pred - 1.96 * finite_std
            upper_bound = finite_pred + 1.96 * finite_std
            coverage_mask = (finite_y >= lower_bound) & (finite_y <= upper_bound)
            coverage = torch.mean(coverage_mask.float()).item()  # FIXED: Convert to float first
        else:
            coverage = 0.0
        
        eval_time = time.time() - eval_start_time
        
        # Store evaluation results
        evaluation_results = {
            'agent_id': self.id,
            'agent_name': f'Agent_{self.id+1}',
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
        
        # Add to evaluation history
        self.evaluation_history.append(evaluation_results)
        
        # Print results
        print(f"Agent {self.id+1} Test Results ({stage_name}):")
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
    
    def get_evaluation_summary(self):
        """Get summary of all evaluations for this agent"""
        if not self.evaluation_history:
            return None
        
        import pandas as pd
        df = pd.DataFrame(self.evaluation_history)
        
        summary = {
            'agent_id': self.id,
            'total_evaluations': len(self.evaluation_history),
            'stages_evaluated': df['stage_name'].unique().tolist(),
            'best_mse': df['mse'].min(),
            'best_r2': df['r2_score'].max(),
            'final_mse': df['mse'].iloc[-1] if len(df) > 0 else None,
            'final_r2': df['r2_score'].iloc[-1] if len(df) > 0 else None,
            'mse_improvement': ((df['mse'].iloc[0] - df['mse'].iloc[-1]) / df['mse'].iloc[0] * 100) if len(df) > 1 else 0,
            'r2_improvement': ((df['r2_score'].iloc[-1] - df['r2_score'].iloc[0]) / abs(df['r2_score'].iloc[0]) * 100) if len(df) > 1 else 0
        }
        
        return summary
    
    def save_evaluation_history(self, filepath):
        """Save evaluation history to CSV"""
        if not self.evaluation_history:
            print(f"Agent {self.id}: No evaluation history to save")
            return None
        
        import pandas as pd
        df = pd.DataFrame(self.evaluation_history)
        df.to_csv(filepath, index=False)
        print(f"Agent {self.id}: Evaluation history saved to {filepath}")
        return filepath
    
    def compare_stages(self, stage1, stage2):
        """Compare performance between two stages"""
        stage1_results = [r for r in self.evaluation_history if r['stage_name'] == stage1]
        stage2_results = [r for r in self.evaluation_history if r['stage_name'] == stage2]
        
        if not stage1_results or not stage2_results:
            print(f"Agent {self.id}: Cannot compare - missing data for {stage1} or {stage2}")
            return None
        
        s1 = stage1_results[-1]  # Most recent result for stage1
        s2 = stage2_results[-1]  # Most recent result for stage2
        
        comparison = {
            'agent_id': self.id,
            'stage1': stage1,
            'stage2': stage2,
            'mse_change': s2['mse'] - s1['mse'],
            'mse_change_percent': ((s2['mse'] - s1['mse']) / s1['mse']) * 100,
            'r2_change': s2['r2_score'] - s1['r2_score'],
            'r2_change_percent': ((s2['r2_score'] - s1['r2_score']) / abs(s1['r2_score'])) * 100,
            'coverage_change': s2['prediction_coverage_95'] - s1['prediction_coverage_95'],
            'uncertainty_change': s2['mean_uncertainty'] - s1['mean_uncertainty']
        }
        
        return comparison
