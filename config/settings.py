"""
Configuration and settings management for the distributed GP learning system.
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    
    # Agent configuration
    num_agents: int = 3
    topology: str = "ring"
    inducing_points: int = 10
    
    # Training configuration
    epochs_per_stage: int = 50
    consensus_steps: int = 5
    learning_rate: float = 0.01
    
    # Data configuration
    data_points: int = 200
    test_split: float = 0.2
    noise_level: float = 0.1
    
    # GPU configuration
    cuda_enabled: bool = True
    
    # Function types to evaluate
    function_types: list = None
    
    # Output configuration
    save_results: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        if self.function_types is None:
            self.function_types = ["multimodal", "sinusoidal", "polynomial", "rbf_mixture"]

@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    # Base directories
    project_root: str = "."
    data_dir: str = "project/data"
    results_dir: str = "results"
    visualization_dir: str = "visualization"
    config_dir: str = "config"
    
    # Result file naming
    baseline_file: str = "baseline_vsgp.csv"
    standard_dac_file: str = "standard_dac.csv"
    weighted_dac_file: str = "weighted_dac_poe.csv"
    comparison_file: str = "r2_comparison.csv"

class SettingsManager:
    """Manages experiment settings and configuration recording."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.experiment_config = ExperimentConfig()
        self.path_config = PathConfig()
        
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
    
    def save_settings(self, filename: str = None) -> str:
        """Save current settings to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_settings_{timestamp}.json"
        
        filepath = os.path.join(self.config_dir, filename)
        
        settings = {
            "experiment_config": asdict(self.experiment_config),
            "path_config": asdict(self.path_config),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        
        return filepath
    
    def load_settings(self, filepath: str) -> None:
        """Load settings from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # Update experiment config
        exp_config = settings.get("experiment_config", {})
        for key, value in exp_config.items():
            if hasattr(self.experiment_config, key):
                setattr(self.experiment_config, key, value)
        
        # Update path config
        path_config = settings.get("path_config", {})
        for key, value in path_config.items():
            if hasattr(self.path_config, key):
                setattr(self.path_config, key, value)
    
    def get_experiment_config(self) -> ExperimentConfig:
        """Get the current experiment configuration."""
        return self.experiment_config
    
    def get_path_config(self) -> PathConfig:
        """Get the current path configuration."""
        return self.path_config
    
    def update_config(self, **kwargs) -> None:
        """Update experiment configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.experiment_config, key):
                setattr(self.experiment_config, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
    
    def print_settings(self) -> None:
        """Print current settings in a readable format."""
        print("=== Experiment Configuration ===")
        for key, value in asdict(self.experiment_config).items():
            print(f"{key}: {value}")
        
        print("\n=== Path Configuration ===")
        for key, value in asdict(self.path_config).items():
            print(f"{key}: {value}")
    
    def get_result_filepath(self, method: str, function_type: str = None) -> str:
        """Get the filepath for saving results."""
        if function_type:
            filename = f"{function_type}_{method}.csv"
        else:
            if method == "baseline":
                filename = self.path_config.baseline_file
            elif method == "standard_dac":
                filename = self.path_config.standard_dac_file
            elif method == "weighted_dac":
                filename = self.path_config.weighted_dac_file
            elif method == "comparison":
                filename = self.path_config.comparison_file
            else:
                filename = f"{method}.csv"
        
        return os.path.join(self.path_config.data_dir, filename)

# Global settings manager instance
settings = SettingsManager()

def get_settings() -> SettingsManager:
    """Get the global settings manager instance."""
    return settings

def save_experiment_record(experiment_name: str, results: Dict[str, Any]) -> str:
    """Save experiment results with settings for reproducibility."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    record = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "settings": {
            "experiment_config": asdict(settings.experiment_config),
            "path_config": asdict(settings.path_config)
        },
        "results": results
    }
    
    filename = f"experiment_record_{experiment_name}_{timestamp}.json"
    filepath = os.path.join(settings.config_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
    
    return filepath

# Example usage and default settings
if __name__ == "__main__":
    # Example of how to use the settings manager
    sm = SettingsManager()
    
    # Print current settings
    sm.print_settings()
    
    # Update some settings
    sm.update_config(num_agents=5, epochs_per_stage=100)
    
    # Save settings
    saved_file = sm.save_settings()
    print(f"\nSettings saved to: {saved_file}")
    
    # Example of saving an experiment record
    example_results = {
        "r2_scores": [0.85, 0.82, 0.88],
        "mse_scores": [0.15, 0.18, 0.12]
    }
    record_file = save_experiment_record("test_experiment", example_results)
    print(f"Experiment record saved to: {record_file}")
