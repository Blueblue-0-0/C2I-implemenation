import pandas as pd
import torch
import numpy as np
import time
import os
from typing import Dict, List, Any

class TestEvaluator:
    """Comprehensive test evaluation and CSV recording system"""
    
    def __init__(self, test_data_path, results_dir, experiment_name="synthetic_experiment"):
        self.test_data_path = test_data_path
        self.results_dir = results_dir
        self.experiment_name = experiment_name
        
        # Load test data
        self.test_data = self._load_test_data()
        
        # Initialize results storage
        self.evaluation_results = []
        self.csv_path = os.path.join(results_dir, f"{experiment_name}_evaluation_results.csv")
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"TestEvaluator initialized:")
        print(f"  - Test data: {self.test_data['x'].shape[0]} samples")
        print(f"  - Results will be saved to: {self.csv_path}")
    
    def _load_test_data(self):
        """Load test data from CSV using noisy observed values for realistic evaluation"""
        if not os.path.exists(self.test_data_path):
            raise FileNotFoundError(f"Test data not found: {self.test_data_path}")
            
        df = pd.read_csv(self.test_data_path)
        
        # Handle different column naming conventions
        x_cols = [col for col in df.columns if col.startswith('x')]
        
        # FIXED: Use noisy observed values for evaluation (realistic scenario)
        if 'y_noisy' in df.columns:
            y_col = 'y_noisy'  # Use noisy observations (realistic)
            eval_type = 'noisy_observed'
            print("Using noisy observed values for evaluation (realistic scenario)")
        elif 'y' in df.columns:
            y_col = 'y'  # Fallback to y column (should contain noisy data)
            eval_type = 'standard'
            print("Using y column for evaluation")
        elif 'y_true' in df.columns:
            y_col = 'y_true'  # Last resort - true values (unrealistic)
            eval_type = 'true_values'
            print("WARNING: Using true values for evaluation (unrealistic scenario)")
        else:
            raise ValueError("No y column found in test data")
        
        test_x = torch.tensor(df[x_cols].values, dtype=torch.float32)
        test_y = torch.tensor(df[y_col].values, dtype=torch.float32)
        
        print(f"Loaded test data: X shape {test_x.shape}, Y shape {test_y.shape}")
        print(f"Evaluation type: {eval_type}")
        print(f"Y range: [{test_y.min():.4f}, {test_y.max():.4f}]")
        
        # OPTIONAL: Store both for comparison if available
        test_data = {'x': test_x, 'y': test_y, 'evaluation_type': eval_type}
        
        if 'y_true' in df.columns and y_col != 'y_true':
            test_y_true = torch.tensor(df['y_true'].values, dtype=torch.float32)
            test_data['y_true'] = test_y_true
            print(f"Also available - True Y range: [{test_y_true.min():.4f}, {test_y_true.max():.4f}]")
            
            # Show noise level
            noise_std = torch.std(test_y - test_y_true).item()
            print(f"Noise level (std): {noise_std:.4f}")
        
        return test_data

    def evaluate_agents(self, agents, stage_name, stage_number, additional_info=None):
        """Evaluate all agents on test data and record results"""
        print(f"\n{'='*60}")
        print(f"EVALUATION STAGE {stage_number}: {stage_name.upper()}")
        print(f"Evaluation type: {self.test_data['evaluation_type']}")
        print(f"{'='*60}")
        
        evaluation_time = time.strftime('%Y-%m-%d %H:%M:%S')
        
        stage_results = []
        
        for agent_idx, agent in enumerate(agents):
            # Evaluate single agent
            metrics = self._evaluate_single_agent(agent, agent_idx)
            
            # FIXED: Ensure agent_id is in the metrics
            metrics['agent_id'] = agent_idx  # Add this line
            
            # Create result record
            result_record = {
                'timestamp': evaluation_time,
                'experiment_name': self.experiment_name,
                'stage_number': stage_number,
                'stage_name': stage_name,
                'agent_id': agent_idx,  # Ensure this is here
                'agent_name': f'Agent_{agent_idx+1}',
                **metrics
            }
            
            # Add additional info if provided
            if additional_info:
                result_record.update(additional_info)
            
            # Store result
            self.evaluation_results.append(result_record)
            stage_results.append(metrics)  # This now includes agent_id
            
            # Print results with dual metrics if available
            print(f"Agent {agent_idx+1} - {stage_name}:")
            print(f"  MSE: {metrics['mse']:.6f} | MAE: {metrics['mae']:.6f} | R2: {metrics['r2_score']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f} | Coverage: {metrics['coverage_95']:.4f}")
            print(f"  Uncertainty: {metrics['avg_prediction_uncertainty']:.6f} | Data: {metrics['training_data_size']}")
            
            # Show comparison with true values if available
            if 'true_mse' in metrics:
                print(f"  [vs True] MSE: {metrics['true_mse']:.6f} | R2: {metrics['true_r2_score']:.6f} | Coverage: {metrics['true_coverage_95']:.4f}")
        
        # Save to CSV immediately after each evaluation
        self._save_to_csv()
        
        print(f"Stage {stage_number} evaluation completed and saved!")
        print(f"Evaluation used: {self.test_data['evaluation_type']}")
        
        return stage_results  # Return metrics for SimpleProgressTracker
    
    def _evaluate_single_agent(self, agent, agent_idx):
        """Evaluate a single agent on test data using noisy observed values"""
        agent.model.eval()
        if hasattr(agent, 'likelihood'):
            agent.likelihood.eval()
        
        test_x = self.test_data['x']
        test_y = self.test_data['y']  # Now contains noisy observed values
        
        # Make predictions in batches to avoid memory issues
        batch_size = 500
        predictions = []
        variances = []
        
        with torch.no_grad():
            for i in range(0, len(test_x), batch_size):
                batch_x = test_x[i:i+batch_size].to(agent.device)
                
                try:
                    # Get posterior distribution
                    posterior = agent.model(batch_x)
                    pred_mean = posterior.mean.cpu()
                    pred_var = posterior.variance.cpu()
                    
                    predictions.append(pred_mean)
                    variances.append(pred_var)
                    
                except Exception as e:
                    print(f"Warning: Error in batch {i//batch_size}: {e}")
                    # Use fallback prediction
                    pred_mean = torch.zeros(len(batch_x))
                    pred_var = torch.ones(len(batch_x))
                    predictions.append(pred_mean)
                    variances.append(pred_var)
        
        # Concatenate all predictions
        all_predictions = torch.cat(predictions, dim=0)
        all_variances = torch.cat(variances, dim=0)
        
        # FIXED: Calculate metrics against noisy observed values
        test_y_flat = test_y.flatten()  # Noisy observed values
        pred_flat = all_predictions.flatten()
        
        # Basic regression metrics (against noisy observations)
        mse = torch.mean((test_y_flat - pred_flat) ** 2).item()
        mae = torch.mean(torch.abs(test_y_flat - pred_flat)).item()
        rmse = np.sqrt(mse)
        
        # R2 score against noisy observations
        ss_res = torch.sum((test_y_flat - pred_flat) ** 2).item()
        ss_tot = torch.sum((test_y_flat - torch.mean(test_y_flat)) ** 2).item()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Uncertainty metrics
        pred_std = torch.sqrt(all_variances).flatten()
        mean_uncertainty = torch.mean(pred_std).item()
        
        # Maximum error (against noisy observations)
        max_error = torch.max(torch.abs(test_y_flat - pred_flat)).item()
        
        # FIXED: Coverage against noisy observed values (realistic)
        finite_mask = torch.isfinite(test_y_flat) & torch.isfinite(pred_flat) & torch.isfinite(pred_std)
        
        if finite_mask.any():
            finite_y = test_y_flat[finite_mask]  # Noisy observations
            finite_pred = pred_flat[finite_mask]
            finite_std = pred_std[finite_mask]
            
            lower_bound = finite_pred - 1.96 * finite_std
            upper_bound = finite_pred + 1.96 * finite_std
            coverage_mask = (finite_y >= lower_bound) & (finite_y <= upper_bound)
            coverage_95 = torch.mean(coverage_mask.float()).item()
        else:
            coverage_95 = 0.0
        
        # Training data size
        training_size = agent.train_x.shape[0] if hasattr(agent, 'train_x') else 0
        
        # OPTIONAL: Calculate metrics against true values for comparison
        true_metrics = {}
        if 'y_true' in self.test_data and self.test_data['evaluation_type'] != 'true_values':
            test_y_true_flat = self.test_data['y_true'].flatten()
            
            # Metrics against true values (for research comparison)
            true_mse = torch.mean((test_y_true_flat - pred_flat) ** 2).item()
            true_mae = torch.mean(torch.abs(test_y_true_flat - pred_flat)).item()
            
            true_ss_res = torch.sum((test_y_true_flat - pred_flat) ** 2).item()
            true_ss_tot = torch.sum((test_y_true_flat - torch.mean(test_y_true_flat)) ** 2).item()
            true_r2 = 1 - (true_ss_res / (true_ss_tot + 1e-8))
            
            # Coverage against true values
            true_finite_mask = torch.isfinite(test_y_true_flat) & finite_mask
            if true_finite_mask.any():
                true_finite_y = test_y_true_flat[true_finite_mask]
                true_finite_pred = pred_flat[true_finite_mask]
                true_finite_std = pred_std[true_finite_mask]
                
                true_lower = true_finite_pred - 1.96 * true_finite_std
                true_upper = true_finite_pred + 1.96 * true_finite_std
                true_coverage_mask = (true_finite_y >= true_lower) & (true_finite_y <= true_upper)
                true_coverage_95 = torch.mean(true_coverage_mask.float()).item()
            else:
                true_coverage_95 = 0.0
            
            true_metrics = {
                'true_mse': true_mse,
                'true_mae': true_mae,
                'true_r2_score': true_r2,
                'true_coverage_95': true_coverage_95
            }
        
        # Return primary metrics (against noisy observations) + optional true metrics
        metrics = {
            'mse': mse,  # Against noisy observations
            'mae': mae,  # Against noisy observations
            'rmse': rmse,
            'r2_score': r2,  # Against noisy observations
            'max_error': max_error,
            'avg_prediction_uncertainty': mean_uncertainty,  # FIXED: Match baseline naming
            'coverage_95': coverage_95,  # Against noisy observations
            'num_test_samples': len(test_y_flat),
            'num_finite_samples': finite_mask.sum().item(),
            'training_data_size': training_size,
            'predictions_mean': pred_flat.mean().item(),
            'predictions_std': pred_flat.std().item(),
            'observed_values_mean': test_y_flat.mean().item(),  # FIXED: Renamed from true_values
            'observed_values_std': test_y_flat.std().item(),    # FIXED: Renamed from true_values
            'evaluation_type': self.test_data['evaluation_type']
        }
        
        # Add true metrics if available
        metrics.update(true_metrics)
        
        return metrics
    
    def _save_to_csv(self):
        """Save all results to CSV"""
        if not self.evaluation_results:
            return
        
        df = pd.DataFrame(self.evaluation_results)
        df.to_csv(self.csv_path, index=False)
        print(f"Results saved to: {self.csv_path}")
    
    def get_summary_by_stage(self):
        """Get summary statistics by stage"""
        if not self.evaluation_results:
            return None
        
        df = pd.DataFrame(self.evaluation_results)
        summary = df.groupby(['stage_number', 'stage_name']).agg({
            'mse': ['mean', 'std', 'min', 'max'],
            'mae': ['mean', 'std'],
            'r2_score': ['mean', 'std'],
            'coverage_95': ['mean', 'std'],
            'training_data_size': 'mean'
        }).round(6)
        
        return summary
    
    def get_agent_progress(self, agent_id):
        """Get progress for a specific agent"""
        if not self.evaluation_results:
            return None
        
        df = pd.DataFrame(self.evaluation_results)
        agent_data = df[df['agent_id'] == agent_id].sort_values('stage_number')
        
        return agent_data[['stage_number', 'stage_name', 'mse', 'mae', 'r2_score', 'coverage_95', 'training_data_size']]
    
    def create_progress_summary(self):
        """Create a comprehensive progress summary"""
        if not self.evaluation_results:
            print("No evaluation results to summarize")
            return None
        
        df = pd.DataFrame(self.evaluation_results)
        
        # Summary by stage
        stage_summary = df.groupby(['stage_number', 'stage_name']).agg({
            'mse': ['mean', 'std'],
            'r2_score': ['mean', 'std'],
            'training_data_size': 'mean'
        }).round(6)
        
        # Progress trends
        stage_means = df.groupby('stage_number')[['mse', 'r2_score']].mean()
        
        # Calculate improvements
        if len(stage_means) > 1:
            initial_mse = stage_means['mse'].iloc[0]
            final_mse = stage_means['mse'].iloc[-1]
            mse_improvement = ((initial_mse - final_mse) / initial_mse) * 100
            
            initial_r2 = stage_means['r2_score'].iloc[0]
            final_r2 = stage_means['r2_score'].iloc[-1]
            r2_improvement = ((final_r2 - initial_r2) / abs(initial_r2 + 1e-8)) * 100
        else:
            mse_improvement = 0
            r2_improvement = 0
        
        summary_path = os.path.join(self.results_dir, f"{self.experiment_name}_progress_summary.txt")
        
        # Use UTF-8 encoding and avoid Unicode characters
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"EXPERIMENT PROGRESS SUMMARY: {self.experiment_name}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Stages: {df['stage_number'].max()}\n")
            f.write(f"Total Agents: {df['agent_id'].nunique()}\n")
            f.write(f"Test Samples: {df['num_test_samples'].iloc[0]}\n\n")
            f.write(f"OVERALL IMPROVEMENT:\n")
            f.write(f"  MSE Improvement: {mse_improvement:+.2f}%\n")
            f.write(f"  R2 Improvement: {r2_improvement:+.2f}%\n\n")  # Fixed: R² -> R2
            f.write(f"STAGE-BY-STAGE SUMMARY:\n")
            f.write(str(stage_summary))
        
        print(f"Progress summary saved to: {summary_path}")
        return summary_path