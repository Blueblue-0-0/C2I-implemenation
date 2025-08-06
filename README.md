# Distributed Gaussian Process Learning with DAC Consensus

A comprehensive evaluation system for distributed Gaussian Process learning using Distributed Averaging Consensus (DAC) methods.

## 📋 Project Overview

This project implements and evaluates three different approaches for Gaussian Process learning:
- **Baseline VSGP**: Centralized Variational Sparse GP trained on complete dataset
- **Standard DAC**: Distributed learning with standard averaging consensus
- **Weighted DAC**: Distributed learning with Product-of-Experts weighted consensus

### Key Features
- Multi-agent distributed learning system
- DAC consensus for hyperparameter coordination
- R² metric focused evaluation
- Stage-by-stage performance tracking
- GPU acceleration support
- Comprehensive result analysis and visualization

## 🏗️ Project Structure

```
├── project/                    # Core implementation
│   ├── main_synthetic.py      # Main experiment runner
│   ├── analyze_results.py     # R² focused analysis
│   ├── agent.py              # Agent implementation
│   ├── gp_model.py           # GP model definitions
│   ├── dac.py                # DAC consensus algorithms
│   └── data/                 # Experiment results (CSV)
├── dataset/                  # Dataset files and preprocessing
├── experiments/              # Experimental comparisons
└── visualization/            # Plots and analysis results

```

## 🚀 Quick Start

### Environment Setup
1. Create and activate virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### Running Experiments
```powershell
cd project
python main_synthetic.py
```

### Analyzing Results
```powershell
python analyze_results.py
```

## 📊 Experiment Configuration

### Function Types
- `multimodal`: Multi-modal synthetic function
- `sinusoidal`: Sinusoidal patterns
- `polynomial`: Polynomial functions  
- `rbf_mixture`: RBF mixture models

### Evaluation Metrics
- **Primary**: R² Score (coefficient of determination)
- **Secondary**: MSE (Mean Squared Error)
- **Tracking**: Stage-by-stage performance evolution

### Learning Stages
1. **Initial**: Base model training
2. **DAC**: First consensus round
3. **Retrain**: Model refinement
4. **DAC**: Second consensus round
5. **Retrain**: Final model refinement
6. **Final**: Final evaluation

## 📁 File Naming Convention

### Result Files
- `baseline_vsgp.csv`: Centralized baseline results
- `standard_dac.csv`: Standard DAC consensus results
- `weighted_dac_poe.csv`: Weighted DAC with PoE results
- `r2_comparison.csv`: Comparative R² analysis

### Function-Specific Results
- `{function_type}_{method}.csv`: Individual experiment results
- `{function_type}_r2_comparison.csv`: Per-function R² comparison

## 🔧 Settings & Configuration

### Agent Configuration
```python
{
    "num_agents": 3,
    "topology": "ring",
    "inducing_points": 10,
    "epochs_per_stage": 50,
    "consensus_steps": 5,
    "cuda_enabled": True
}
```

### Experiment Settings
- **Data Points**: 200 per function type
- **Test Split**: 20% of data
- **Noise Level**: 0.1
- **Learning Rate**: 0.01

## 📈 Analysis Features

### Performance Comparison
- R² score comparison across methods
- Stage-by-stage convergence analysis
- Statistical significance testing
- Visualization of learning curves

### Consensus Analysis
- Hyperparameter convergence tracking
- Inducing point optimization
- Agent coordination effectiveness

## 🛠️ Core Components

### Agent System (`agent.py`)
- Variational Sparse GP implementation
- DAC consensus integration
- Performance tracking
- GPU optimization

### DAC Consensus (`dac.py`)
- Standard averaging consensus
- Weighted Product-of-Experts consensus
- Ring topology communication
- Convergence detection

### GP Models (`gp_model.py`)
- Variational Sparse GP
- Hyperparameter optimization
- Inducing point management
- CUDA acceleration

## 📋 Requirements

See `requirements.txt` for complete dependency list:
- PyTorch >= 1.9.0
- GPyTorch >= 1.6.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0

## 🔬 Research Focus

This implementation focuses on:
1. **Distributed Learning**: Multi-agent GP learning with consensus
2. **Consensus Methods**: Comparing standard vs weighted DAC approaches
3. **Performance Evaluation**: R²-based metric comparison
4. **Practical Applications**: Real-world distributed learning scenarios

## 📝 Usage Notes

- Windows encoding handled for R² Unicode symbols
- GPU acceleration automatically detected
- Results stored in CSV format for easy analysis
- Comprehensive logging and progress tracking included


