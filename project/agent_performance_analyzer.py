import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

class AgentPerformanceAnalyzer:
    """Comprehensive agent performance analysis for streaming experiments"""
    
    def __init__(self):
        self.agent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
        self.agent_markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
        self.agent_styles = ['-', '--', '-.', ':']  # Solid, Dashed, Dash-dot, Dotted
        self.function_types = ['multimodal', 'sinusoidal', 'polynomial', 'rbf_mixture']
        
        # Metrics configuration
        self.METRICS_CONFIG = {
            'mse': {
                'title': 'Mean Squared Error',
                'ylabel': 'MSE',
                'format': '.6f',  # FIXED
                'lower_is_better': True
            },
            'mae': {
                'title': 'Mean Absolute Error',
                'ylabel': 'MAE',
                'format': '.6f',  # FIXED
                'lower_is_better': True
            },
            'rmse': {
                'title': 'Root Mean Squared Error',
                'ylabel': 'RMSE',
                'format': '.6f',  # FIXED
                'lower_is_better': True
            },
            'r2_score': {
                'title': 'R² Score',
                'ylabel': 'R² Score',
                'format': '.4f',  # FIXED
                'lower_is_better': False
            },
            'mean_uncertainty': {
                'title': 'Mean Prediction Uncertainty',
                'ylabel': 'Mean Uncertainty',
                'format': '.4f',  # FIXED
                'lower_is_better': True
            },
            'std_uncertainty': {
                'title': 'Std Prediction Uncertainty',
                'ylabel': 'Std Uncertainty',
                'format': '.4f',  # FIXED
                'lower_is_better': True
            },
            'coverage_95': {
                'title': '95% Prediction Coverage',
                'ylabel': 'Coverage Rate',
                'format': '.3f',  # FIXED
                'lower_is_better': False
            },
            'coverage_50': {
                'title': '50% Prediction Coverage',
                'ylabel': 'Coverage Rate',
                'format': '.3f',  # FIXED
                'lower_is_better': False
            },
            'training_data_size': {
                'title': 'Training Data Size',
                'ylabel': 'Number of Samples',
                'format': '.0f',  # FIXED: Changed from 'd' to '.0f'
                'lower_is_better': False
            },
            'test_data_size': {
                'title': 'Test Data Size',
                'ylabel': 'Number of Samples',
                'format': '.0f',  # FIXED: Changed from 'd' to '.0f'
                'lower_is_better': False
            }
        }
        
    def load_experiment_data(self, csv_path):
        """Load experiment data from CSV file"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded data: {len(df)} records from {csv_path}")
        
        # Extract experiment info
        experiment_info = {
            'function_type': df['function_type'].iloc[0] if 'function_type' in df.columns else 'unknown',
            'num_agents': df['agent_id'].nunique(),
            'total_stages': df['stage_number'].max(),
            'num_records': len(df)
        }
        
        print(f"Experiment info: {experiment_info}")
        return df, experiment_info
    
    def create_agent_performance_curves(self, df, experiment_info, save_dir):
        """Create comprehensive agent performance curve analysis"""
        
        function_type = experiment_info['function_type']
        num_agents = experiment_info['num_agents']
        
        print(f"\nCreating agent performance curves for {function_type}...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Performance metrics to analyze
        metrics = {
            'mse': {
                'title': 'Mean Squared Error (MSE)',
                'ylabel': 'MSE',
                'yscale': 'log',
                'better': 'lower',
                'format': '.6f'
            },
            'r2_score': {
                'title': 'R2 Score',
                'ylabel': 'R2 Score',
                'yscale': 'linear',
                'better': 'higher',
                'format': '.4f'
            },
            'mean_uncertainty': {
                'title': 'Mean Prediction Uncertainty',
                'ylabel': 'Mean Uncertainty',
                'yscale': 'linear',
                'better': 'lower',
                'format': '.4f'
            },
            'coverage_95': {
                'title': '95% Prediction Coverage',
                'ylabel': 'Coverage',
                'yscale': 'linear',
                'better': 'target_0.95',
                'format': '.3f'
            },
            'training_data_size': {
                'title': 'Training Data Size',
                'ylabel': 'Training Data Size',
                'yscale': 'linear',
                'better': 'higher',
                'format': 'd'
            }
        }
        
        # Create individual metric plots
        for metric, config in metrics.items():
            if metric in df.columns:
                self._create_single_metric_plot(df, metric, config, function_type, num_agents, save_dir)
        
        # Create comprehensive dashboard
        self._create_comprehensive_dashboard(df, metrics, function_type, num_agents, save_dir)
        
        # Create stage type analysis
        self._create_stage_type_analysis(df, function_type, num_agents, save_dir)
        
        # Create agent comparison summary
        self._create_agent_comparison_summary(df, function_type, num_agents, save_dir)
        
        # Create performance improvement analysis
        self._create_improvement_analysis(df, function_type, num_agents, save_dir)
        
        print(f"All agent performance plots saved to: {save_dir}")
    
    def _create_single_metric_plot(self, df, metric, config, function_type, num_agents, save_dir):
        """Create plot for a single metric with COMPLETELY SAFE formatting"""
        
        plt.figure(figsize=(12, 8))
        
        # Plot lines for each agent
        for agent_id in range(num_agents):
            agent_data = df[df['agent_id'] == agent_id].sort_values('stage_number')
            if len(agent_data) > 0:
                plt.plot(agent_data['stage_number'], agent_data[metric], 
                        marker='o', linewidth=2, markersize=6,
                        label=f'Agent {agent_id+1}', alpha=0.8)
                
                # COMPLETELY SAFE: Annotation formatting
                final_value = agent_data[metric].iloc[-1]
                
                # Safe formatting function with multiple fallbacks
                def safe_format_annotation(value, format_spec):
                    try:
                        if format_spec == 'd':
                            return f'{int(float(value))}'
                        elif format_spec == '.0f':
                            return f'{float(value):.0f}'
                        elif 'f' in format_spec:
                            return f'{float(value):{format_spec}}'
                        else:
                            return f'{value}'
                    except (ValueError, TypeError):
                        try:
                            if isinstance(value, (int, float)):
                                if abs(float(value) - round(float(value))) < 1e-10:
                                    return f'{int(round(float(value)))}'
                                else:
                                    return f'{float(value):.4f}'
                            else:
                                return str(value)
                        except:
                            return str(value)
                
                annotation_text = safe_format_annotation(final_value, config.get("format", ".4f"))
                
                plt.annotate(annotation_text,
                            xy=(agent_data['stage_number'].iloc[-1], final_value),
                            xytext=(10, 5), textcoords='offset points',
                            fontsize=10, alpha=0.8)
        
        # Plot formatting
        plt.xlabel('Stage Number', fontsize=12)
        plt.ylabel(config['ylabel'], fontsize=12)
        plt.title(f'{config["title"]} - {function_type.upper()} - All Agents', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, f'{function_type}_{metric}_by_agent.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  {config['title']} plot saved: {plot_path}")
        return plot_path
    
    def _create_comprehensive_dashboard(self, df, metrics, function_type, num_agents, save_dir):
        """Create comprehensive dashboard with all metrics"""
        
        # Filter available metrics
        available_metrics = {k: v for k, v in metrics.items() if k in df.columns}
        
        # Create subplot layout
        n_metrics = len(available_metrics)
        cols = 3
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(24, 6*rows))
        fig.suptitle(f'Comprehensive Agent Performance Dashboard - {function_type.title()} Function',
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Flatten axes for easy indexing
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for idx, (metric, config) in enumerate(available_metrics.items()):
            ax = axes[idx]
            
            # Plot each agent
            for agent_id in range(num_agents):
                agent_data = df[df['agent_id'] == agent_id].copy()
                agent_data = agent_data.sort_values('stage_number')
                stage_performance = agent_data.groupby('stage_number')[metric].mean()
                
                ax.plot(stage_performance.index, stage_performance.values,
                       color=self.agent_colors[agent_id],
                       marker=self.agent_markers[agent_id],
                       linestyle=self.agent_styles[agent_id],
                       linewidth=2,
                       markersize=4,
                       label=f'Agent {agent_id + 1}',
                       alpha=0.8)
            
            # Formatting
            ax.set_xlabel('Stage Number', fontsize=11)
            ax.set_ylabel(config['ylabel'], fontsize=11)
            ax.set_title(config['title'], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if config['yscale'] == 'log':
                ax.set_yscale('log')
            
            if metric == 'coverage_95':
                ax.axhline(y=0.95, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add legend only to first subplot
            if idx == 0:
                ax.legend(fontsize=10, loc='best')
        
        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = os.path.join(save_dir, f'{function_type}_agent_performance_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Dashboard saved: {dashboard_path}")
        plt.close()
    
    def _create_stage_type_analysis(self, df, function_type, num_agents, save_dir):
        """Analyze performance by stage type"""
        
        # Categorize stages
        def categorize_stage(stage_name):
            if pd.isna(stage_name):
                return 'Unknown'
            stage_name = str(stage_name).lower()
            if 'initial' in stage_name:
                return 'Initial Training'
            elif 'retrain' in stage_name:
                return 'Buffer Retrain'
            elif 'dac' in stage_name or 'consensus' in stage_name:
                return 'DAC Consensus'
            elif 'final' in stage_name:
                return 'Final Consensus'
            else:
                return 'Other'
        
        df_copy = df.copy()
        df_copy['stage_type'] = df_copy['stage_name'].apply(categorize_stage)
        
        # Create stage type analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Performance by Stage Type - {function_type.title()} Function',
                    fontsize=16, fontweight='bold')
        
        # Plot 1: MSE by stage type
        ax1 = axes[0, 0]
        stage_types = df_copy['stage_type'].unique()
        for agent_id in range(num_agents):
            agent_data = df_copy[df_copy['agent_id'] == agent_id]
            stage_type_mse = agent_data.groupby('stage_type')['mse'].mean()
            
            ax1.plot(range(len(stage_type_mse)), stage_type_mse.values,
                    color=self.agent_colors[agent_id],
                    marker=self.agent_markers[agent_id],
                    linewidth=2,
                    markersize=8,
                    label=f'Agent {agent_id + 1}')
        
        ax1.set_xlabel('Stage Type')
        ax1.set_ylabel('Mean MSE')
        ax1.set_title('MSE by Stage Type')
        ax1.set_xticks(range(len(stage_type_mse)))
        ax1.set_xticklabels(stage_type_mse.index, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: R² by stage type
        ax2 = axes[0, 1]
        for agent_id in range(num_agents):
            agent_data = df_copy[df_copy['agent_id'] == agent_id]
            stage_type_r2 = agent_data.groupby('stage_type')['r2_score'].mean()
            
            ax2.plot(range(len(stage_type_r2)), stage_type_r2.values,
                    color=self.agent_colors[agent_id],
                    marker=self.agent_markers[agent_id],
                    linewidth=2,
                    markersize=8,
                    label=f'Agent {agent_id + 1}')
        
        ax2.set_xlabel('Stage Type')
        ax2.set_ylabel('Mean R2 Score')
        ax2.set_title('R2 Score by Stage Type')
        ax2.set_xticks(range(len(stage_type_r2)))
        ax2.set_xticklabels(stage_type_r2.index, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Stage count distribution
        ax3 = axes[1, 0]
        stage_counts = df_copy.groupby(['agent_id', 'stage_type']).size().unstack(fill_value=0)
        stage_counts.plot(kind='bar', ax=ax3, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        ax3.set_xlabel('Agent')
        ax3.set_ylabel('Number of Stages')
        ax3.set_title('Stage Type Distribution by Agent')
        ax3.legend(title='Stage Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Improvement by stage type
        ax4 = axes[1, 1]
        improvement_data = []
        for agent_id in range(num_agents):
            agent_data = df_copy[df_copy['agent_id'] == agent_id]
            for stage_type in agent_data['stage_type'].unique():
                type_data = agent_data[agent_data['stage_type'] == stage_type]
                if len(type_data) > 1:
                    initial_mse = type_data['mse'].iloc[0]
                    final_mse = type_data['mse'].iloc[-1]
                    improvement = ((initial_mse - final_mse) / initial_mse) * 100
                    improvement_data.append({
                        'agent': agent_id,
                        'stage_type': stage_type,
                        'improvement': improvement
                    })
        
        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            for agent_id in range(num_agents):
                agent_improvements = improvement_df[improvement_df['agent'] == agent_id]
                ax4.scatter(agent_improvements['stage_type'], agent_improvements['improvement'],
                           color=self.agent_colors[agent_id], s=100, alpha=0.7,
                           label=f'Agent {agent_id + 1}')
        
        ax4.set_xlabel('Stage Type')
        ax4.set_ylabel('MSE Improvement (%)')
        ax4.set_title('MSE Improvement by Stage Type')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save stage type analysis
        stage_path = os.path.join(save_dir, f'{function_type}_stage_type_analysis.png')
        plt.savefig(stage_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Stage type analysis saved: {stage_path}")
        plt.close()
    
    def _create_agent_comparison_summary(self, df, function_type, num_agents, save_dir):
        """Create agent comparison summary"""
        
        # Calculate summary statistics
        summary_data = []
        for agent_id in range(num_agents):
            agent_data = df[df['agent_id'] == agent_id].copy()
            
            # Initial and final performance
            initial_stage = agent_data[agent_data['stage_number'] == agent_data['stage_number'].min()]
            final_stage = agent_data[agent_data['stage_number'] == agent_data['stage_number'].max()]
            
            initial_mse = initial_stage['mse'].iloc[0] if len(initial_stage) > 0 else np.nan
            final_mse = final_stage['mse'].mean() if len(final_stage) > 0 else np.nan
            initial_r2 = initial_stage['r2_score'].iloc[0] if len(initial_stage) > 0 else np.nan
            final_r2 = final_stage['r2_score'].mean() if len(final_stage) > 0 else np.nan
            
            # Best performance
            best_mse = agent_data['mse'].min()
            best_r2 = agent_data['r2_score'].max()
            
            # Improvements
            mse_improvement = ((initial_mse - final_mse) / initial_mse) * 100 if not np.isnan(initial_mse) and initial_mse != 0 else 0
            r2_improvement = ((final_r2 - initial_r2) / abs(initial_r2 + 1e-8)) * 100 if not np.isnan(initial_r2) else 0
            
            summary_data.append({
                'Agent': f'Agent {agent_id + 1}',
                'Initial MSE': initial_mse,
                'Final MSE': final_mse,
                'Best MSE': best_mse,
                'MSE Improvement (%)': mse_improvement,
                'Initial R2': initial_r2,
                'Final R2': final_r2,
                'Best R2': best_r2,
                'R2 Improvement (%)': r2_improvement,
                'Final Data Size': final_stage['training_data_size'].iloc[0] if len(final_stage) > 0 else 0,
                'Total Stages': agent_data['stage_number'].max()
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Agent Performance Comparison Summary - {function_type.title()} Function',
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Initial vs Final MSE
        ax1 = axes[0, 0]
        agents = summary_df['Agent']
        x_pos = np.arange(len(agents))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, summary_df['Initial MSE'], width, 
                       label='Initial MSE', color='lightcoral', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, summary_df['Final MSE'], width,
                       label='Final MSE', color='lightgreen', alpha=0.8)
        
        ax1.set_xlabel('Agent')
        ax1.set_ylabel('MSE')
        ax1.set_title('Initial vs Final MSE')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(agents)
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Initial vs Final R²
        ax2 = axes[0, 1]
        bars3 = ax2.bar(x_pos - width/2, summary_df['Initial R2'], width,
                       label='Initial R2', color='lightcoral', alpha=0.8)
        bars4 = ax2.bar(x_pos + width/2, summary_df['Final R2'], width,
                       label='Final R2', color='lightgreen', alpha=0.8)
        
        ax2.set_xlabel('Agent')
        ax2.set_ylabel('R2 Score')
        ax2.set_title('Initial vs Final R2')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(agents)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: MSE Improvement
        ax3 = axes[0, 2]
        bars5 = ax3.bar(agents, summary_df['MSE Improvement (%)'], 
                       color=[self.agent_colors[i] for i in range(num_agents)], alpha=0.8)
        ax3.set_xlabel('Agent')
        ax3.set_ylabel('MSE Improvement (%)')
        ax3.set_title('MSE Improvement by Agent')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars5, summary_df['MSE Improvement (%)']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: R² Improvement
        ax4 = axes[1, 0]
        bars6 = ax4.bar(agents, summary_df['R2 Improvement (%)'],
                       color=[self.agent_colors[i] for i in range(num_agents)], alpha=0.8)
        ax4.set_xlabel('Agent')
        ax4.set_ylabel('R2 Improvement (%)')
        ax4.set_title('R2 Improvement by Agent')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Best Performance Comparison
        ax5 = axes[1, 1]
        ax5.scatter(summary_df['Best MSE'], summary_df['Best R2'], 
                   s=150, c=[self.agent_colors[i] for i in range(num_agents)], alpha=0.8)
        
        for i, agent in enumerate(summary_df['Agent']):
            ax5.annotate(agent, (summary_df['Best MSE'].iloc[i], summary_df['Best R2'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax5.set_xlabel('Best MSE')
        ax5.set_ylabel('Best R2')
        ax5.set_title('Best Performance Scatter')
        ax5.set_xscale('log')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Final Data Size
        ax6 = axes[1, 2]
        bars7 = ax6.bar(agents, summary_df['Final Data Size'],
                       color=[self.agent_colors[i] for i in range(num_agents)], alpha=0.8)
        ax6.set_xlabel('Agent')
        ax6.set_ylabel('Final Training Data Size')
        ax6.set_title('Training Data Utilization')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comparison summary
        comparison_path = os.path.join(save_dir, f'{function_type}_agent_comparison_summary.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Comparison summary saved: {comparison_path}")
        plt.close()
        
        # Save summary table
        table_path = os.path.join(save_dir, f'{function_type}_agent_summary_table.csv')
        summary_df.to_csv(table_path, index=False)
        print(f"  Summary table saved: {table_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"AGENT PERFORMANCE SUMMARY - {function_type.upper()}")
        print(f"{'='*80}")
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        return summary_df
    
    def _create_improvement_analysis(self, df, function_type, num_agents, save_dir):
        """Create detailed improvement analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Performance Improvement Analysis - {function_type.title()} Function',
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Stage-to-stage improvement rate
        ax1 = axes[0, 0]
        for agent_id in range(num_agents):
            agent_data = df[df['agent_id'] == agent_id].copy()
            agent_data = agent_data.sort_values('stage_number')
            stage_mse = agent_data.groupby('stage_number')['mse'].mean()
            
            # Calculate improvement rates
            improvement_rates = []
            stages = []
            for i in range(1, len(stage_mse)):
                prev_mse = stage_mse.iloc[i-1]
                curr_mse = stage_mse.iloc[i]
                improvement = ((prev_mse - curr_mse) / prev_mse) * 100
                improvement_rates.append(improvement)
                stages.append(stage_mse.index[i])
            
            ax1.plot(stages, improvement_rates,
                    color=self.agent_colors[agent_id],
                    marker=self.agent_markers[agent_id],
                    linewidth=2,
                    markersize=4,
                    label=f'Agent {agent_id + 1}',
                    alpha=0.8)
        
        ax1.set_xlabel('Stage Number')
        ax1.set_ylabel('MSE Improvement Rate (%)')
        ax1.set_title('Stage-to-Stage MSE Improvement Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Cumulative improvement
        ax2 = axes[0, 1]
        for agent_id in range(num_agents):
            agent_data = df[df['agent_id'] == agent_id].copy()
            agent_data = agent_data.sort_values('stage_number')
            stage_mse = agent_data.groupby('stage_number')['mse'].mean()
            
            initial_mse = stage_mse.iloc[0]
            cumulative_improvement = ((initial_mse - stage_mse) / initial_mse) * 100
            
            ax2.plot(stage_mse.index, cumulative_improvement,
                    color=self.agent_colors[agent_id],
                    marker=self.agent_markers[agent_id],
                    linewidth=2,
                    markersize=4,
                    label=f'Agent {agent_id + 1}',
                    alpha=0.8)
        
        ax2.set_xlabel('Stage Number')
        ax2.set_ylabel('Cumulative MSE Improvement (%)')
        ax2.set_title('Cumulative MSE Improvement from Initial')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R2 progression
        ax3 = axes[1, 0]
        for agent_id in range(num_agents):
            agent_data = df[df['agent_id'] == agent_id].copy()
            agent_data = agent_data.sort_values('stage_number')
            stage_r2 = agent_data.groupby('stage_number')['r2_score'].mean()
            
            ax3.plot(stage_r2.index, stage_r2.values,
                    color=self.agent_colors[agent_id],
                    marker=self.agent_markers[agent_id],
                    linewidth=2,
                    markersize=4,
                    label=f'Agent {agent_id + 1}',
                    alpha=0.8)
        
        ax3.set_xlabel('Stage Number')
        ax3.set_ylabel('R2 Score')
        ax3.set_title('R2 Score Progression')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning efficiency (improvement per data point)
        ax4 = axes[1, 1]
        for agent_id in range(num_agents):
            agent_data = df[df['agent_id'] == agent_id].copy()
            agent_data = agent_data.sort_values('stage_number')
            
            initial_mse = agent_data['mse'].iloc[0]
            current_mse = agent_data.groupby('stage_number')['mse'].mean()
            data_size = agent_data.groupby('stage_number')['training_data_size'].mean()
            
            # Calculate efficiency as improvement per additional data point
            initial_data = data_size.iloc[0]
            efficiency = []
            stages = []
            
            for i in range(1, len(current_mse)):
                mse_improvement = ((initial_mse - current_mse.iloc[i]) / initial_mse) * 100
                data_increase = data_size.iloc[i] - initial_data
                if data_increase > 0:
                    eff = mse_improvement / data_increase
                    efficiency.append(eff)
                    stages.append(current_mse.index[i])
            
            if efficiency:
                ax4.plot(stages, efficiency,
                        color=self.agent_colors[agent_id],
                        marker=self.agent_markers[agent_id],
                        linewidth=2,
                        markersize=4,
                        label=f'Agent {agent_id + 1}',
                        alpha=0.8)
        
        ax4.set_xlabel('Stage Number')
        ax4.set_ylabel('Learning Efficiency (% improvement per data point)')
        ax4.set_title('Learning Efficiency Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save improvement analysis
        improvement_path = os.path.join(save_dir, f'{function_type}_improvement_analysis.png')
        plt.savefig(improvement_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Improvement analysis saved: {improvement_path}")
        plt.close()

    def create_cross_function_comparison(self, all_experiment_data, save_dir):
        """Create cross-function type comparison"""
        
        print(f"\nCreating cross-function comparison analysis...")
        os.makedirs(save_dir, exist_ok=True)
        
        # Combine all data
        all_summaries = []
        for function_type, (df, experiment_info) in all_experiment_data.items():
            num_agents = experiment_info['num_agents']
            
            # Calculate summary for each agent
            for agent_id in range(num_agents):
                agent_data = df[df['agent_id'] == agent_id].copy()
                
                initial_stage = agent_data[agent_data['stage_number'] == agent_data['stage_number'].min()]
                final_stage = agent_data[agent_data['stage_number'] == agent_data['stage_number'].max()]
                
                initial_mse = initial_stage['mse'].iloc[0] if len(initial_stage) > 0 else np.nan
                final_mse = final_stage['mse'].mean() if len(final_stage) > 0 else np.nan
                initial_r2 = initial_stage['r2_score'].iloc[0] if len(initial_stage) > 0 else np.nan
                final_r2 = final_stage['r2_score'].mean() if len(final_stage) > 0 else np.nan
                
                mse_improvement = ((initial_mse - final_mse) / initial_mse) * 100 if not np.isnan(initial_mse) and initial_mse != 0 else 0
                r2_improvement = ((final_r2 - initial_r2) / abs(initial_r2 + 1e-8)) * 100 if not np.isnan(initial_r2) else 0
                
                all_summaries.append({
                    'function_type': function_type,
                    'agent_id': agent_id,
                    'agent_name': f'Agent {agent_id + 1}',
                    'initial_mse': initial_mse,
                    'final_mse': final_mse,
                    'mse_improvement': mse_improvement,
                    'initial_r2': initial_r2,
                    'final_r2': final_r2,
                    'r2_improvement': r2_improvement,
                    'total_stages': agent_data['stage_number'].max()
                })
        
        summary_df = pd.DataFrame(all_summaries)
        
        # Create cross-function comparison plots
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('Cross-Function Type Agent Performance Comparison',
                    fontsize=20, fontweight='bold')
        
        # Plot 1: Final MSE comparison by function type
        ax1 = axes[0, 0]
        function_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, func_type in enumerate(self.function_types):
            if func_type in summary_df['function_type'].values:
                func_data = summary_df[summary_df['function_type'] == func_type]
                
                for agent_id in range(4):  # Assuming 4 agents
                    agent_data = func_data[func_data['agent_id'] == agent_id]
                    if len(agent_data) > 0:
                        ax1.scatter([i + (agent_id-1.5)*0.1], agent_data['final_mse'].values,
                                  color=self.agent_colors[agent_id], s=100, alpha=0.8,
                                  label=f'Agent {agent_id+1}' if i == 0 else "")
        
        ax1.set_xlabel('Function Type')
        ax1.set_ylabel('Final MSE')
        ax1.set_title('Final MSE by Function Type and Agent')
        ax1.set_xticks(range(len(self.function_types)))
        ax1.set_xticklabels([f.title() for f in self.function_types])
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MSE Improvement comparison
        ax2 = axes[0, 1]
        for i, func_type in enumerate(self.function_types):
            if func_type in summary_df['function_type'].values:
                func_data = summary_df[summary_df['function_type'] == func_type]
                
                for agent_id in range(4):
                    agent_data = func_data[func_data['agent_id'] == agent_id]
                    if len(agent_data) > 0:
                        ax2.scatter([i + (agent_id-1.5)*0.1], agent_data['mse_improvement'].values,
                                  color=self.agent_colors[agent_id], s=100, alpha=0.8)
        
        ax2.set_xlabel('Function Type')
        ax2.set_ylabel('MSE Improvement (%)')
        ax2.set_title('MSE Improvement by Function Type and Agent')
        ax2.set_xticks(range(len(self.function_types)))
        ax2.set_xticklabels([f.title() for f in self.function_types])
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Final R2 comparison
        ax3 = axes[1, 0]
        for i, func_type in enumerate(self.function_types):
            if func_type in summary_df['function_type'].values:
                func_data = summary_df[summary_df['function_type'] == func_type]
                
                for agent_id in range(4):
                    agent_data = func_data[func_data['agent_id'] == agent_id]
                    if len(agent_data) > 0:
                        ax3.scatter([i + (agent_id-1.5)*0.1], agent_data['final_r2'].values,
                                  color=self.agent_colors[agent_id], s=100, alpha=0.8)
        
        ax3.set_xlabel('Function Type')
        ax3.set_ylabel('Final R2 Score')
        ax3.set_title('Final R2 Score by Function Type and Agent')
        ax3.set_xticks(range(len(self.function_types)))
        ax3.set_xticklabels([f.title() for f in self.function_types])
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Agent performance ranking across functions
        ax4 = axes[1, 1]
        agent_rankings = []
        
        for func_type in self.function_types:
            if func_type in summary_df['function_type'].values:
                func_data = summary_df[summary_df['function_type'] == func_type]
                func_data_sorted = func_data.sort_values('final_mse')
                
                for rank, (_, agent_row) in enumerate(func_data_sorted.iterrows()):
                    agent_rankings.append({
                        'function_type': func_type,
                        'agent_id': agent_row['agent_id'],
                        'rank': rank + 1
                    })
        
        if agent_rankings:
            ranking_df = pd.DataFrame(agent_rankings)
            ranking_pivot = ranking_df.pivot(index='function_type', columns='agent_id', values='rank')
            
            for agent_id in range(4):
                if agent_id in ranking_pivot.columns:
                    ax4.plot(range(len(ranking_pivot)), ranking_pivot[agent_id],
                            color=self.agent_colors[agent_id], marker='o', linewidth=2,
                            label=f'Agent {agent_id+1}')
        
        ax4.set_xlabel('Function Type')
        ax4.set_ylabel('Performance Rank (1=Best)')
        ax4.set_title('Agent Performance Ranking Across Functions')
        ax4.set_xticks(range(len(self.function_types)))
        ax4.set_xticklabels([f.title() for f in self.function_types])
        ax4.set_ylim(0.5, 4.5)
        ax4.invert_yaxis()
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Function difficulty analysis
        ax5 = axes[2, 0]
        function_difficulty = summary_df.groupby('function_type').agg({
            'final_mse': 'mean',
            'mse_improvement': 'mean'
        })
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(function_difficulty)]
        bars = ax5.bar(range(len(function_difficulty)), function_difficulty['final_mse'],
                      color=colors, alpha=0.8)
        
        ax5.set_xlabel('Function Type')
        ax5.set_ylabel('Average Final MSE')
        ax5.set_title('Function Difficulty (Average Final MSE)')
        ax5.set_xticks(range(len(function_difficulty)))
        ax5.set_xticklabels([f.title() for f in function_difficulty.index])
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, function_difficulty['final_mse']):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 6: Learning potential analysis
        ax6 = axes[2, 1]
        bars2 = ax6.bar(range(len(function_difficulty)), function_difficulty['mse_improvement'],
                       color=colors, alpha=0.8)
        
        ax6.set_xlabel('Function Type')
        ax6.set_ylabel('Average MSE Improvement (%)')
        ax6.set_title('Learning Potential (Average MSE Improvement)')
        ax6.set_xticks(range(len(function_difficulty)))
        ax6.set_xticklabels([f.title() for f in function_difficulty.index])
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, function_difficulty['mse_improvement']):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save cross-function comparison
        comparison_path = os.path.join(save_dir, 'cross_function_agent_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Cross-function comparison saved: {comparison_path}")
        plt.close()
        
        # Save summary data
        summary_path = os.path.join(save_dir, 'cross_function_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"  Cross-function summary saved: {summary_path}")
        
        # Print analysis summary
        print(f"\n{'='*100}")
        print("CROSS-FUNCTION AGENT PERFORMANCE ANALYSIS")
        print(f"{'='*100}")
        
        print("\nAverage Performance by Function Type:")
        function_summary = summary_df.groupby('function_type').agg({
            'final_mse': ['mean', 'std'],
            'mse_improvement': ['mean', 'std'],
            'final_r2': ['mean', 'std']
        }).round(4)
        print(function_summary)
        
        print("\nBest Performing Agent by Function:")
        for func_type in self.function_types:
            if func_type in summary_df['function_type'].values:
                func_data = summary_df[summary_df['function_type'] == func_type]
                best_agent = func_data.loc[func_data['final_mse'].idxmin()]
                print(f"  {func_type.title()}: {best_agent['agent_name']} "
                      f"(MSE: {best_agent['final_mse']:.6f}, "
                      f"Improvement: {best_agent['mse_improvement']:.1f}%)")
        
        return summary_df

def analyze_single_experiment(csv_path, save_dir=None):
    """Analyze a single experiment CSV file"""
    
    analyzer = AgentPerformanceAnalyzer()
    
    # Load data
    df, experiment_info = analyzer.load_experiment_data(csv_path)
    
    # Determine save directory
    if save_dir is None:
        csv_dir = os.path.dirname(csv_path)
        function_type = experiment_info['function_type']
        save_dir = os.path.join(csv_dir, f'{function_type}_agent_analysis')
    
    # Create analysis
    analyzer.create_agent_performance_curves(df, experiment_info, save_dir)
    
    return save_dir, df, experiment_info

def analyze_all_function_types(base_dir='project/train_record'):
    """Analyze all function types comprehensively"""
    
    analyzer = AgentPerformanceAnalyzer()
    function_types = ['multimodal', 'sinusoidal', 'polynomial', 'rbf_mixture']
    
    print(f"""
{'='*120}
COMPREHENSIVE AGENT PERFORMANCE ANALYSIS - ALL FUNCTION TYPES
Function Types: {', '.join([f.title() for f in function_types])}
{'='*120}
""")
    
    # Find CSV files for each function type
    all_experiment_data = {}
    results = {}
    
    for function_type in function_types:
        print(f"\n{'*'*80}")
        print(f"ANALYZING {function_type.upper()} FUNCTION")
        print(f"{'*'*80}")
        
        # Look for CSV file
        csv_pattern = f"synthetic_{function_type}_streaming"
        csv_file = None
        
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if csv_pattern in file and file.endswith('_evaluation_results.csv'):
                    csv_file = os.path.join(root, file)
                    break
            if csv_file:
                break
        
        if csv_file and os.path.exists(csv_file):
            try:
                print(f"Found: {csv_file}")
                save_dir, df, experiment_info = analyze_single_experiment(csv_file)
                results[function_type] = save_dir
                all_experiment_data[function_type] = (df, experiment_info)
                print(f"SUCCESS: {function_type} analysis completed!")
            except Exception as e:
                print(f"ERROR analyzing {function_type}: {e}")
                continue
        else:
            print(f"WARNING: No CSV file found for {function_type}")
            continue
    
    # Create cross-function comparison if we have multiple functions
    if len(all_experiment_data) > 1:
        print(f"\n{'*'*80}")
        print("CREATING CROSS-FUNCTION COMPARISON")
        print(f"{'*'*80}")
        
        cross_function_dir = os.path.join(base_dir, 'cross_function_analysis')
        summary_df = analyzer.create_cross_function_comparison(all_experiment_data, cross_function_dir)
        results['cross_function'] = cross_function_dir
    
    # Final summary
    print(f"\n{'='*120}")
    print("AGENT PERFORMANCE ANALYSIS COMPLETED FOR ALL FUNCTION TYPES")
    print(f"{'='*120}")
    
    print(f"Successfully analyzed: {len(all_experiment_data)}/{len(function_types)} function types")
    
    for function_type, save_dir in results.items():
        if function_type != 'cross_function':
            print(f"  {function_type.upper()}: {save_dir}")
    
    if 'cross_function' in results:
        print(f"  CROSS-FUNCTION ANALYSIS: {results['cross_function']}")
    
    print(f"\nGenerated plots for each function type:")
    print("  - Individual metric evolution curves")
    print("  - Comprehensive performance dashboard")
    print("  - Stage type analysis")
    print("  - Agent comparison summary")
    print("  - Performance improvement analysis")
    if len(all_experiment_data) > 1:
        print("  - Cross-function comparison analysis")
    
    return results, all_experiment_data

if __name__ == "__main__":
    # Analyze all function types
    results, all_data = analyze_all_function_types()
    
    print(f"\n{'='*120}")
    print("ANALYSIS COMPLETE! Check the generated directories for detailed plots and data.")
    print(f"{'='*120}")