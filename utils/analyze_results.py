import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_function_results(function_type):
    """Analyze results for a specific function type"""
    
    results_dir = f"project/results/{function_type}"
    
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        return None
    
    print(f"\n{'='*80}")
    print(f"ANALYZING {function_type.upper()} RESULTS")
    print(f"{'='*80}")
    
    # Load all CSV files with your actual naming convention
    files_found = {}
    
    # Check for baseline VSGP results
    baseline_path = os.path.join(results_dir, 'baseline_vsgp.csv')
    if os.path.exists(baseline_path):
        files_found['baseline_vsgp'] = pd.read_csv(baseline_path)
        print(f" Found baseline_vsgp.csv")
    
    # Check for standard DAC results  
    standard_path = os.path.join(results_dir, 'standard_dac.csv')
    if os.path.exists(standard_path):
        files_found['standard_dac'] = pd.read_csv(standard_path)
        print(f" Found standard_dac.csv with {len(files_found['standard_dac'])} records")
    
    # Check for weighted DAC PoE results
    weighted_path = os.path.join(results_dir, 'weighted_dac_poe.csv')
    if os.path.exists(weighted_path):
        files_found['weighted_dac_poe'] = pd.read_csv(weighted_path)
        print(f" Found weighted_dac_poe.csv with {len(files_found['weighted_dac_poe'])} records")
    
    # Check for R2 comparison results
    comparison_path = os.path.join(results_dir, 'r2_comparison.csv')
    if os.path.exists(comparison_path):
        files_found['r2_comparison'] = pd.read_csv(comparison_path)
        print(f" Found r2_comparison.csv")
    
    if not files_found:
        print(f"ERROR: No result files found in {results_dir}")
        print(f"Expected files: baseline_vsgp.csv, standard_dac.csv, weighted_dac_poe.csv, r2_comparison.csv")
        return None
    
    return analyze_combined_results(files_found, function_type, results_dir)

def analyze_combined_results(files_found, function_type, results_dir):
    """Analyze and visualize combined results with R2 focus"""
    
    # Create analysis directory
    analysis_dir = os.path.join(results_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"\nAnalysis will be saved to: {analysis_dir}")
    
    # ============================================================================
    # R2-FOCUSED STAGE-BY-STAGE PROGRESS ANALYSIS
    # ============================================================================
    
    dac_methods = ['standard_dac', 'weighted_dac_poe']
    available_dac_methods = [method for method in dac_methods if method in files_found]
    
    if available_dac_methods:
        print(f"\n--- R2-FOCUSED STAGE-BY-STAGE PROGRESS ANALYSIS ---")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{function_type.title()} - DAC Methods R2-Focused Progress', fontsize=16)
        
        # R2 Evolution (PRIMARY METRIC)
        ax1 = axes[0, 0]
        for method_name in available_dac_methods:
            df = files_found[method_name]
            # Group by stage for clean visualization
            stage_summary = df.groupby('stage_number')['r2_score'].mean()
            method_display = method_name.replace('_', ' ').title()
            ax1.plot(stage_summary.index, stage_summary.values, 
                    marker='o', label=f'{method_display}', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Stage Number')
        ax1.set_ylabel('R2 Score (PRIMARY METRIC)')
        ax1.set_title('R2 Score Evolution - PRIMARY COMPARISON')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # MSE Evolution (SECONDARY METRIC)
        ax2 = axes[0, 1]
        for method_name in available_dac_methods:
            df = files_found[method_name]
            stage_summary = df.groupby('stage_number')['mse'].mean()
            method_display = method_name.replace('_', ' ').title()
            ax2.plot(stage_summary.index, stage_summary.values,
                    marker='s', label=f'{method_display}', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Stage Number')
        ax2.set_ylabel('MSE (log scale, secondary)')
        ax2.set_title('MSE Evolution - SECONDARY METRIC')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Training Data Size Evolution
        ax3 = axes[1, 0]
        for method_name in available_dac_methods:
            df = files_found[method_name]
            stage_summary = df.groupby('stage_number')['training_data_size'].mean()
            method_display = method_name.replace('_', ' ').title()
            ax3.plot(stage_summary.index, stage_summary.values,
                    marker='^', label=f'{method_display}', linewidth=2, markersize=6)
        
        ax3.set_xlabel('Stage Number')
        ax3.set_ylabel('Training Data Size')
        ax3.set_title('Data Accumulation Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Stage Type Distribution
        ax4 = axes[1, 1]
        if available_dac_methods:
            df = files_found[available_dac_methods[0]]  # Use first available method
            stage_types = df['stage_name'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(stage_types)))
            wedges, texts, autotexts = ax4.pie(stage_types.values, labels=stage_types.index, 
                                             autopct='%1.1f%%', colors=colors, startangle=90)
            ax4.set_title('Stage Type Distribution')
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        progress_plot_path = os.path.join(analysis_dir, f'{function_type}_r2_focused_progress.png')
        plt.savefig(progress_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"R2-focused progress analysis saved to: {progress_plot_path}")
    
    # ============================================================================
    # AGENT-LEVEL R2 PERFORMANCE ANALYSIS
    # ============================================================================
    
    if available_dac_methods:
        print(f"\n--- AGENT-LEVEL R2 PERFORMANCE ANALYSIS ---")
        
        n_methods = len(available_dac_methods)
        fig, axes = plt.subplots(2, n_methods, figsize=(8*n_methods, 12))
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'{function_type.title()} - Agent R2 Performance Comparison', fontsize=16)
        
        for i, method_name in enumerate(available_dac_methods):
            df = files_found[method_name]
            method_display = method_name.replace('_', ' ').title()
            
            # R2 trajectories by agent
            ax1 = axes[0, i]
            for agent_id in sorted(df['agent_id'].unique()):
                agent_data = df[df['agent_id'] == agent_id]
                ax1.plot(agent_data['stage_number'], agent_data['r2_score'], 
                        marker='o', label=f'Agent {agent_id}', alpha=0.8, linewidth=2)
            
            ax1.set_xlabel('Stage Number')
            ax1.set_ylabel('R2 Score (PRIMARY METRIC)')
            ax1.set_title(f'{method_display}\nAgent R2 Trajectories')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Final R2 performance comparison
            ax2 = axes[1, i]
            final_stage = df['stage_number'].max()
            final_data = df[df['stage_number'] == final_stage]
            
            agents = sorted(final_data['agent_id'].unique())
            final_r2 = [final_data[final_data['agent_id'] == agent]['r2_score'].iloc[0] for agent in agents]
            
            x_pos = np.arange(len(agents))
            bars = ax2.bar(x_pos, final_r2, alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, len(agents))))
            ax2.set_xlabel('Agent ID')
            ax2.set_ylabel('Final R2 Score')
            ax2.set_title(f'{method_display}\nFinal Agent R2 Performance')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f'Agent {a}' for a in agents])
            ax2.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, val in zip(bars, final_r2):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        agent_plot_path = os.path.join(analysis_dir, f'{function_type}_agent_r2_analysis.png')
        plt.savefig(agent_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Agent R2 analysis saved to: {agent_plot_path}")
    
    # ============================================================================
    # R2-FOCUSED METHOD COMPARISON
    # ============================================================================
    
    if 'r2_comparison' in files_found:
        print(f"\n--- R2-FOCUSED METHOD COMPARISON ---")
        
        comparison_df = files_found['r2_comparison']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{function_type.title()} - R2-Focused Method Comparison', fontsize=16)
        
        methods = comparison_df['Method']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(methods)]  # Better colors
        
        # R2 Comparison (PRIMARY)
        ax1 = axes[0]
        r2_values = comparison_df['R2']
        bars1 = ax1.bar(methods, r2_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_ylabel('R2 Score (PRIMARY METRIC)')
        ax1.set_title('R2 Score Comparison\n(PRIMARY METRIC)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars1, r2_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # MSE Comparison (SECONDARY)
        ax2 = axes[1]
        mse_values = comparison_df['MSE']
        bars2 = ax2.bar(methods, mse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax2.set_ylabel('MSE (log scale, secondary)')
        ax2.set_title('MSE Comparison\n(SECONDARY METRIC)')
        ax2.set_yscale('log')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, mse_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.2,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Method Performance Ranking
        ax3 = axes[2]
        # Sort methods by R2 performance
        ranking_df = comparison_df.sort_values('R2', ascending=False).reset_index(drop=True)
        ranking_df['Rank'] = ranking_df.index + 1
        
        bars3 = ax3.barh(ranking_df['Method'], ranking_df['R2'], 
                        color=['gold', 'silver', '#CD7F32'][:len(ranking_df)], alpha=0.8)
        ax3.set_xlabel('R2 Score')
        ax3.set_title('Method Ranking\n(by R2 Performance)')
        ax3.set_xlim(0, 1)
        
        for i, (bar, val) in enumerate(zip(bars3, ranking_df['R2'])):
            ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'#{i+1}: {val:.4f}', va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        comparison_plot_path = os.path.join(analysis_dir, f'{function_type}_r2_method_comparison.png')
        plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"R2-focused method comparison saved to: {comparison_plot_path}")
    
    # ============================================================================
    # R2-FOCUSED PERFORMANCE SUMMARY REPORT
    # ============================================================================
    
    print(f"\n--- R2-FOCUSED PERFORMANCE SUMMARY ---")
    
    summary_report = []
    summary_report.append(f"{function_type.upper()} EXPERIMENT - R2-FOCUSED SUMMARY")
    summary_report.append("="*80)
    summary_report.append("PRIMARY METRIC: R2 Score")
    summary_report.append("SECONDARY METRIC: MSE")
    summary_report.append("")
    
    # Baseline performance
    if 'baseline_vsgp' in files_found:
        baseline = files_found['baseline_vsgp'].iloc[0]
        summary_report.append("BASELINE VSGP (Centralized):")
        summary_report.append(f"  R2 Score: {baseline['r2_score']:.6f} (PRIMARY)")
        summary_report.append(f"  MSE: {baseline['mse']:.6f} (secondary)")
        summary_report.append(f"  Training Data: {baseline['training_data_size']}")
        summary_report.append(f"  Coverage: {baseline.get('coverage', 'N/A')}")
        summary_report.append("")
    
    # DAC methods performance with R2 focus
    for method_name in available_dac_methods:
        df = files_found[method_name]
        method_display = method_name.replace('_', ' ').title()
        
        # Initial vs Final performance
        initial_stage = df['stage_number'].min()
        final_stage = df['stage_number'].max()
        
        initial_r2 = df[df['stage_number'] == initial_stage]['r2_score'].mean()
        final_r2 = df[df['stage_number'] == final_stage]['r2_score'].mean()
        
        initial_mse = df[df['stage_number'] == initial_stage]['mse'].mean()
        final_mse = df[df['stage_number'] == final_stage]['mse'].mean()
        
        # R2 improvement (PRIMARY FOCUS)
        r2_improvement = final_r2 - initial_r2
        r2_improvement_pct = (r2_improvement / max(abs(initial_r2), 0.001)) * 100
        
        # MSE improvement (SECONDARY)
        mse_improvement_pct = ((initial_mse - final_mse) / initial_mse) * 100
        
        final_data_size = df[df['stage_number'] == final_stage]['training_data_size'].mean()
        final_coverage = df[df['stage_number'] == final_stage]['coverage'].mean()
        
        summary_report.append(f"{method_display} (Distributed):")
        summary_report.append(f"  R2 Score: {initial_r2:.6f} → {final_r2:.6f} (PRIMARY)")
        summary_report.append(f"  R2 Improvement: {r2_improvement:+.6f} ({r2_improvement_pct:+.2f}%)")
        summary_report.append(f"  MSE: {initial_mse:.6f} → {final_mse:.6f} (secondary)")
        summary_report.append(f"  MSE Improvement: {mse_improvement_pct:.2f}%")
        summary_report.append(f"  Final Data Size: {final_data_size:.0f}")
        summary_report.append(f"  Total Stages: {final_stage}")
        summary_report.append(f"  Final Coverage: {final_coverage:.4f}")
        summary_report.append("")
    
    # Method ranking by R2
    if 'r2_comparison' in files_found:
        comparison_df = files_found['r2_comparison']
        ranking = comparison_df.sort_values('R2', ascending=False)
        
        summary_report.append("METHOD RANKING (by R2 Score - PRIMARY METRIC):")
        for i, (_, row) in enumerate(ranking.iterrows()):
            summary_report.append(f"  #{i+1}: {row['Method']} - R2 = {row['R2']:.6f}")
        summary_report.append("")
        
        best_method = ranking.iloc[0]['Method']
        best_r2 = ranking.iloc[0]['R2']
        summary_report.append(f"BEST PERFORMING METHOD: {best_method}")
        summary_report.append(f"BEST R2 SCORE: {best_r2:.6f}")
    
    # Save summary report
    summary_path = os.path.join(analysis_dir, f'{function_type}_r2_focused_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_report))
    
    print(f"R2-focused summary report saved to: {summary_path}")
    
    # Print summary to console
    for line in summary_report:
        print(line)
    
    return {
        'function_type': function_type,
        'files_found': files_found,
        'analysis_dir': analysis_dir,
        'summary_report': summary_report,
        'available_methods': available_dac_methods
    }

def analyze_overall_results():
    """Analyze R2-focused results across all function types"""
    
    print(f"\n{'='*120}")
    print("OVERALL R2-FOCUSED ANALYSIS ACROSS ALL FUNCTION TYPES")
    print("PRIMARY METRIC: R2 Score | SECONDARY METRIC: MSE")
    print(f"{'='*120}")
    
    results_base_dir = "project/results"
    
    if not os.path.exists(results_base_dir):
        print(f"ERROR: Results directory not found: {results_base_dir}")
        return None
    
    # Find all function type directories
    function_types = []
    for item in os.listdir(results_base_dir):
        item_path = os.path.join(results_base_dir, item)
        if os.path.isdir(item_path) and item not in ['overall', 'overall_analysis']:
            # Check if it has result files with your naming convention
            has_results = any(os.path.exists(os.path.join(item_path, filename)) 
                            for filename in ['r2_comparison.csv', 'baseline_vsgp.csv', 
                                           'standard_dac.csv', 'weighted_dac_poe.csv'])
            if has_results:
                function_types.append(item)
    
    if not function_types:
        print("ERROR: No function type results found")
        print("Expected files: baseline_vsgp.csv, standard_dac.csv, weighted_dac_poe.csv, r2_comparison.csv")
        return None
    
    print(f"Found results for function types: {', '.join(function_types)}")
    
    # Analyze each function type
    all_function_results = {}
    for func_type in function_types:
        print(f"\nAnalyzing {func_type}...")
        result = analyze_function_results(func_type)
        if result:
            all_function_results[func_type] = result
    
    # Create overall R2-focused comparison
    if len(all_function_results) > 1:
        create_overall_r2_comparison(all_function_results)
    
    return all_function_results

def create_overall_r2_comparison(all_results):
    """Create overall R2-focused comparison across all function types"""
    
    print(f"\n--- CREATING OVERALL R2-FOCUSED COMPARISON ---")
    
    # Create overall analysis directory
    overall_dir = "project/results/overall_r2_analysis"
    os.makedirs(overall_dir, exist_ok=True)
    
    # Collect R2-focused comparison data
    overall_comparison_data = []
    
    for func_type, result in all_results.items():
        if 'r2_comparison' in result['files_found']:
            comparison_df = result['files_found']['r2_comparison']
            for _, row in comparison_df.iterrows():
                overall_comparison_data.append({
                    'Function': func_type.title(),
                    'Method': row['Method'],
                    'R2': row['R2'],  # PRIMARY METRIC
                    'MSE': row['MSE'],  # SECONDARY METRIC
                    'Type': row.get('Type', 'Unknown')
                })
    
    if not overall_comparison_data:
        print("No comparison data found across function types")
        return
    
    overall_df = pd.DataFrame(overall_comparison_data)
    
    # Save overall comparison data
    overall_csv_path = os.path.join(overall_dir, 'overall_r2_comparison.csv')
    overall_df.to_csv(overall_csv_path, index=False, encoding='utf-8')
    print(f"Overall R2-focused comparison data saved to: {overall_csv_path}")
    
    # Create comprehensive R2-focused visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Overall R2-Focused Method Comparison Across All Function Types', fontsize=16)
    
    # R2 by function and method (PRIMARY)
    ax1 = axes[0, 0]
    r2_pivot = overall_df.pivot(index='Function', columns='Method', values='R2')
    r2_pivot.plot(kind='bar', ax=ax1, colormap='viridis')
    ax1.set_title('R2 Comparison by Function Type (PRIMARY METRIC)')
    ax1.set_ylabel('R2 Score')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # MSE by function and method (SECONDARY)
    ax2 = axes[0, 1]
    mse_pivot = overall_df.pivot(index='Function', columns='Method', values='MSE')
    mse_pivot.plot(kind='bar', ax=ax2, logy=True, colormap='plasma')
    ax2.set_title('MSE Comparison by Function Type (SECONDARY)')
    ax2.set_ylabel('MSE (log scale)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Average R2 across all functions (PRIMARY FOCUS)
    ax3 = axes[1, 0]
    avg_performance = overall_df.groupby('Method').agg({
        'R2': 'mean', 
        'MSE': 'mean'
    }).reset_index().sort_values('R2', ascending=False)
    
    x_pos = np.arange(len(avg_performance))
    colors = ['gold', 'silver', '#CD7F32'][:len(avg_performance)]
    bars = ax3.bar(x_pos, avg_performance['R2'], alpha=0.8, color=colors, 
                   edgecolor='black', linewidth=1)
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Average R2 Score (PRIMARY METRIC)')
    ax3.set_title('Average R2 Performance Ranking')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(avg_performance['Method'], rotation=45)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and ranking
    for i, (bar, val) in enumerate(zip(bars, avg_performance['R2'])):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'#{i+1}\n{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Method type distribution
    ax4 = axes[1, 1]
    type_counts = overall_df['Type'].value_counts()
    colors_pie = ['lightblue', 'lightcoral', 'lightgreen']
    wedges, texts, autotexts = ax4.pie(type_counts.values, labels=type_counts.index, 
                                      autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax4.set_title('Method Type Distribution')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    overall_plot_path = os.path.join(overall_dir, 'overall_r2_focused_comparison.png')
    plt.savefig(overall_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Overall R2-focused comparison plot saved to: {overall_plot_path}")
    
    # Create R2-focused summary report
    summary_lines = []
    summary_lines.append("OVERALL R2-FOCUSED PERFORMANCE SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append("PRIMARY METRIC: R2 Score")
    summary_lines.append("SECONDARY METRIC: MSE")
    summary_lines.append("")
    
    # Method ranking by R2 (PRIMARY METRIC)
    summary_lines.append("METHOD RANKING (by Average R2 Score):")
    for i, (_, row) in enumerate(avg_performance.iterrows()):
        summary_lines.append(f"  #{i+1}: {row['Method']}")
        summary_lines.append(f"      R2 Score: {row['R2']:.6f} (PRIMARY)")
        summary_lines.append(f"      MSE: {row['MSE']:.6f} (secondary)")
        summary_lines.append("")
    
    # Best performing method
    best_method = avg_performance.iloc[0]['Method']
    best_r2 = avg_performance.iloc[0]['R2']
    best_mse = avg_performance.iloc[0]['MSE']
    
    summary_lines.append("RECOMMENDATION:")
    summary_lines.append(f"  BEST OVERALL METHOD: {best_method}")
    summary_lines.append(f"  BEST R2 SCORE: {best_r2:.6f}")
    summary_lines.append(f"  CORRESPONDING MSE: {best_mse:.6f}")
    summary_lines.append("")
    
    # Function-specific insights
    summary_lines.append("FUNCTION-SPECIFIC INSIGHTS:")
    for func_type in overall_df['Function'].unique():
        func_data = overall_df[overall_df['Function'] == func_type]
        best_for_func = func_data.loc[func_data['R2'].idxmax()]
        summary_lines.append(f"  {func_type}: Best method is {best_for_func['Method']} (R2 = {best_for_func['R2']:.4f})")
    
    # Save summary
    summary_path = os.path.join(overall_dir, 'overall_r2_focused_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    
    print(f"Overall R2-focused summary saved to: {summary_path}")
    
    # Print to console
    for line in summary_lines:
        print(line)

def main():
    """Main R2-focused analysis function"""
    
    print(f"""
{'='*120}
R2-FOCUSED RESULTS ANALYSIS FOR SYNTHETIC EXPERIMENTS
Analyzing: baseline_vsgp.csv, standard_dac.csv, weighted_dac_poe.csv, r2_comparison.csv
PRIMARY METRIC: R2 Score | SECONDARY METRIC: MSE
{'='*120}
""")
    
    # First try to analyze overall results
    try:
        overall_results = analyze_overall_results()
        
        if overall_results:
            print(f"\n{'='*120}")
            print("R2-FOCUSED ANALYSIS COMPLETE!")
            print(f"{'='*120}")
            print(f"Analyzed {len(overall_results)} function types")
            print("Generated R2-focused analysis files:")
            print("  - Individual function R2-focused analysis plots and reports")
            print("  - Overall R2-focused comparison across all functions")
            print("  - R2-focused performance summary reports")
            print("  - Method ranking by R2 performance")
            print(f"{'='*120}")
            return overall_results
        else:
            print("No results found to analyze")
            print("Make sure you have files named:")
            print("  - baseline_vsgp.csv")
            print("  - standard_dac.csv") 
            print("  - weighted_dac_poe.csv")
            print("  - r2_comparison.csv")
            return None
            
    except Exception as e:
        print(f"ERROR during R2-focused analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    
    if result:
        print(f"\n{'='*60}")
        print("SUCCESS! Check the generated R2-focused analysis files.")
        print("Focus: R2 Score as primary comparison metric")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("ANALYSIS FAILED! Check the error messages above.")
        print("Ensure your experiment files are named correctly.")
        print(f"{'='*60}")