import os
import numpy as np
import torch
import pandas as pd

# Ensemble functions

def poe_ensemble(means, covariances):
    """
    means: list of [n_points] arrays, one per agent
    covariances: list of [n_points, n_points] arrays, one per agent
    Returns: poe_mean [n_points], poe_var [n_points], poe_cov [n_points, n_points]
    """
    means = [np.asarray(m) for m in means]
    covariances = [np.asarray(cov) for cov in covariances]
    n_agents = len(means)
    n_points = means[0].shape[0]

    # Compute precision matrices and weighted means
    inv_covs = [np.linalg.inv(cov) for cov in covariances]
    precision_sum = np.sum(inv_covs, axis=0)
    poe_cov = np.linalg.inv(precision_sum)

    weighted_mean = np.zeros(n_points)
    for i in range(n_agents):
        weighted_mean += inv_covs[i] @ means[i]
    poe_mean = poe_cov @ weighted_mean

    poe_var = np.diag(poe_cov)
    return poe_mean, poe_cov

def pcm_ensemble(means, covariances=None):
    means = np.array(means)
    pcm_mean = np.mean(means, axis=0)
    # Full covariance: average agent covariances
    if covariances is not None:
        pcm_cov = np.mean(np.array(covariances), axis=0)
    else:
        # If no covariances, use identity
        pcm_cov = np.eye(means.shape[1])
    return pcm_mean, pcm_cov

def get_moe_gating_weights(means, variances):
    variances = np.array(variances)
    inv_vars = 1.0 / (variances + 1e-8)
    weights = inv_vars / np.sum(inv_vars, axis=0)
    return weights

def moe_ensemble(means, weights, covariances=None):
    means = np.array(means)  # shape: (n_agents, n_points)
    weights = np.array(weights)  # shape: (n_agents, n_points)
    n_agents, n_points = means.shape
    # Compute mean: weighted sum per point
    moe_mean = np.sum(weights * means, axis=0)  # shape: (n_points,)
    if covariances is not None:
        covariances = [np.asarray(cov) for cov in covariances]  # each: (n_points, n_points)
        moe_cov = np.zeros_like(covariances[0])
        for i in range(n_agents):
            mu_i = means[i]  # shape: (n_points,)
            Sigma_i = covariances[i]  # shape: (n_points, n_points)
            diff = (mu_i - moe_mean).reshape(-1, 1)  # shape: (n_points, 1)
            # For per-point weights, use mean of weights for covariance aggregation
            w_i = np.mean(weights[i])
            moe_cov += w_i * (Sigma_i + diff @ diff.T)
    else:
        moe_cov = np.eye(n_points)
    return moe_mean, moe_cov

# Set paths
DATA_ROOT = r"E:/TUM/RCI-S5-SS25/GP/Practice/data/synthetic/rbf_mixture"
RESULTS_DIR = r"E:/TUM/RCI-S5-SS25/GP/Practice/validation/results/rbf_mixture"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_AGENTS = 4
INIT_TRAIN_SIZE = 300

# Utility: Load agent data

def load_agent_data(agent_idx):
    path = os.path.join(DATA_ROOT, f"agent{agent_idx+1}.csv")
    df = pd.read_csv(path)
    x_cols = [c for c in df.columns if c.startswith('x')]
    x = df[x_cols].values[:INIT_TRAIN_SIZE]
    y = df['y'].values[:INIT_TRAIN_SIZE].reshape(-1, 1)
    return x, y

def load_test_data():
    path = os.path.join(DATA_ROOT, "test.csv")
    df = pd.read_csv(path)
    x_cols = [c for c in df.columns if c.startswith('x')]
    x = df[x_cols].values
    y = df['y'].values if 'y' in df.columns else df['y_true'].values
    return x, y

def load_inducing_points():
    path = os.path.join(DATA_ROOT, "inducing.csv")
    df = pd.read_csv(path)
    x_cols = [c for c in df.columns if c.startswith('x')]
    return df[x_cols].values

# Train VSGP model (baseline)
def train_vsgp(train_x, train_y, inducing_points, num_iter=200):
    import gpytorch
    from gpytorch.models import VariationalGP
    from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
    from gpytorch.kernels import ScaleKernel, RBFKernel
    from gpytorch.means import ConstantMean
    from gpytorch.likelihoods import GaussianLikelihood

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    inducing_points = torch.tensor(inducing_points, dtype=torch.float32)

    class BaselineVSGP(VariationalGP):
        def __init__(self, inducing_points):
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            variational_strategy = VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
            super().__init__(variational_strategy)
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(RBFKernel())
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    model = BaselineVSGP(inducing_points)
    likelihood = GaussianLikelihood()
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))
    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y.squeeze())
        loss.backward()
        optimizer.step()
    # Record optimised hyperparameters
    hypers = {
        'mean_constant': model.mean_module.constant.item(),
        'kernel_lengthscale': model.covar_module.base_kernel.lengthscale.item(),
        'kernel_outputscale': model.covar_module.outputscale.item(),
    }
    return model, likelihood, hypers

# Extract variational mean/covariance at inducing points
def get_gp_mean_covar(model):
    var_dist = model.variational_strategy._variational_distribution
    mean = var_dist.variational_mean.detach().cpu().numpy()
    chol = var_dist.chol_variational_covar.detach().cpu().numpy()
    covar = chol @ chol.T
    return mean, covar

# Evaluate model on test set
def evaluate_model(model, likelihood, test_x, test_y):
    model.eval()
    likelihood.eval()
    test_x = torch.tensor(test_x, dtype=torch.float32)
    with torch.no_grad():
        pred_dist = likelihood(model(test_x))
        pred_mean = pred_dist.mean.cpu().numpy()
        pred_var = pred_dist.variance.cpu().numpy()
    mse = np.mean((pred_mean - test_y) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_mean - test_y))
    ss_res = np.sum((test_y - pred_mean) ** 2)
    ss_tot = np.sum((test_y - np.mean(test_y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return dict(r2=r2, rmse=rmse, mae=mae, mse=mse)

import copy  # at top of file, ensure copy is imported
def inject_ensemble_to_vsgp_fixed(base_model, base_likelihood, ensemble_mean, ensemble_cov):
    """
    Inject the ensemble mean and covariance into a copy of the trained VSGP model's variational distribution.
    Avoid retraining; maintain the trained hyperparameters and variational distribution structure.
    """
    import torch
    # Deepcopy the trained model and likelihood to preserve original
    model = copy.deepcopy(base_model)
    likelihood = copy.deepcopy(base_likelihood)

    # Convert ensemble statistics to torch
    mean_t = torch.tensor(ensemble_mean, dtype=torch.float32)
    cov_t = torch.tensor(ensemble_cov, dtype=torch.float32)

    # Inject into variational distribution
    var_dist = model.variational_strategy._variational_distribution
    with torch.no_grad():
        var_dist.variational_mean.copy_(mean_t)
        # Compute Cholesky factor with small jitter
        jitter = 1e-6 * torch.eye(cov_t.size(0))
        chol = torch.linalg.cholesky(cov_t + jitter)
        var_dist.chol_variational_covar.copy_(chol)

    # Freeze model and likelihood
    for param in model.parameters():
        param.requires_grad = False
    for param in likelihood.parameters():
        param.requires_grad = False

    model.eval()
    likelihood.eval()
    return model, likelihood

if __name__ == "__main__":
    # 1. Load all agents' initial data
    all_x, all_y = [], []
    for i in range(N_AGENTS):
        x, y = load_agent_data(i)
        all_x.append(x)
        all_y.append(y)
    full_train_x = np.vstack(all_x)
    full_train_y = np.vstack(all_y)
    inducing_points = load_inducing_points()
    test_x, test_y = load_test_data()

    # 2. Train baseline VSGP on full data
    baseline_model, baseline_likelihood, hypers = train_vsgp(full_train_x, full_train_y, inducing_points, num_iter=200)
    baseline_mean, baseline_cov = get_gp_mean_covar(baseline_model)
    baseline_metrics = evaluate_model(baseline_model, baseline_likelihood, test_x, test_y)

    # 3. Train each agent locally with fixed inducing points and hyperparams
    agent_means, agent_covs, agent_metrics = [], [], []
    for i in range(N_AGENTS):
        x, y = load_agent_data(i)
        # Build agent model with fixed hypers
        model, likelihood, _ = train_vsgp(x, y, inducing_points, num_iter=200)
        # Inject fixed hypers
        model.mean_module.constant.data.fill_(hypers['mean_constant'])
        model.covar_module.base_kernel.lengthscale.data.fill_(hypers['kernel_lengthscale'])
        model.covar_module.outputscale.data.fill_(hypers['kernel_outputscale'])
        mean, cov = get_gp_mean_covar(model)
        agent_means.append(mean)
        agent_covs.append(cov)
        metrics = evaluate_model(model, likelihood, test_x, test_y)
        agent_metrics.append(metrics)

    # --- After collecting agent_means and agent_covs, compute ensemble means/covariances ---
    poe_mean, poe_cov = poe_ensemble(agent_means, agent_covs)
    pcm_mean, pcm_cov = pcm_ensemble(agent_means, agent_covs)
    moe_weights = get_moe_gating_weights(agent_means, np.array([np.diag(cov) for cov in agent_covs]))
    moe_mean, moe_cov = moe_ensemble(agent_means, moe_weights, agent_covs)

    # Evaluate ensemble metrics using ensemble means and full covariances
    poe_model, poe_likelihood = inject_ensemble_to_vsgp_fixed(baseline_model, baseline_likelihood, poe_mean, poe_cov)
    poe_metrics = evaluate_model(poe_model, poe_likelihood, test_x, test_y)
    poe_model_mean, poe_model_cov = get_gp_mean_covar(poe_model)
    pcm_model, pcm_likelihood = inject_ensemble_to_vsgp_fixed(baseline_model, baseline_likelihood, pcm_mean, pcm_cov)
    pcm_metrics = evaluate_model(pcm_model, pcm_likelihood, test_x, test_y)
    pcm_model_mean, pcm_model_cov = get_gp_mean_covar(pcm_model)
    moe_model, moe_likelihood = inject_ensemble_to_vsgp_fixed(baseline_model, baseline_likelihood, moe_mean, moe_cov)
    moe_metrics = evaluate_model(moe_model, moe_likelihood, test_x, test_y)
    moe_model_mean, moe_model_cov = get_gp_mean_covar(moe_model)
    # --- Save all means ---
    means_dict = {f"agent{i+1}_mean": agent_means[i] for i in range(N_AGENTS)}
    means_dict.update({
        "poe_mean": poe_mean,
        "pcm_mean": pcm_mean,
        "moe_mean": moe_mean,
        "baseline_mean": baseline_mean,
        "poe_model_mean": poe_model_mean,
        "pcm_model_mean": pcm_model_mean,
        "moe_model_mean": moe_model_mean
    })
    pd.DataFrame(means_dict).to_csv(os.path.join(RESULTS_DIR, "all_means.csv"), index=False)

    # --- Save all covariances ---
    covar_dict = {f"agent{i+1}_covar": agent_covs[i].flatten() for i in range(N_AGENTS)}
    covar_dict.update({
        "poe_covar": poe_cov.flatten(),
        "pcm_covar": pcm_cov.flatten(),
        "moe_covar": moe_cov.flatten(),
        "baseline_covar": baseline_cov.flatten()
    })
    pd.DataFrame(covar_dict).to_csv(os.path.join(RESULTS_DIR, "all_covariances.csv"), index=False)

    # --- Save all metrics ---
    metrics_list = []
    for i in range(N_AGENTS):
        m = agent_metrics[i]
        m['method'] = f"Agent{i+1}"
        metrics_list.append(m)
    metrics_list.append({**baseline_metrics, 'method': 'Baseline'})
    metrics_list.append({**poe_metrics, 'method': 'PoE'})
    metrics_list.append({**pcm_metrics, 'method': 'PCM'})
    metrics_list.append({**moe_metrics, 'method': 'MoE'})
    pd.DataFrame(metrics_list).to_csv(os.path.join(RESULTS_DIR, "all_metrics.csv"), index=False)

    # --- Save hyperparameters ---
    pd.DataFrame([hypers]).to_csv(os.path.join(RESULTS_DIR, "baselines_hyperparams.csv"), index=False)

    # Plot comparison of metrics across baseline, agents, and ensemble methods
    import matplotlib.pyplot as plt

    # Load all metrics
    all_metrics_df = pd.read_csv(os.path.join(RESULTS_DIR, "all_metrics.csv"))

    # Prepare data for ensemble comparison
    methods = ['Baseline', 'PoE', 'PCM', 'MoE']
    metrics = ['r2', 'rmse', 'mae', 'mse']
    data = []
    for m in methods:
        row = all_metrics_df[all_metrics_df['method'] == m]
        if not row.empty:
            row = row.iloc[0]
            data.append([row[metric] for metric in metrics])
        else:
            data.append([np.nan]*len(metrics))
    data = np.array(data)

    # Plot ensemble comparison
    fig, axs = plt.subplots(1, len(metrics), figsize=(16, 4))
    for i, metric in enumerate(metrics):
        axs[i].bar(methods, data[:, i], color=['C0', 'C1', 'C2', 'C3'])
        axs[i].set_title(metric.upper())
        axs[i].set_ylabel(metric)
        axs[i].set_xticklabels(methods, rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_comparison.png"))
    

    # Prepare data for agent vs ensemble comparison
    all_methods = [f"Agent{i+1}" for i in range(N_AGENTS)] + ["Baseline", "PoE", "PCM", "MoE"]
    data = []
    for m in all_methods:
        row = all_metrics_df[all_metrics_df['method'] == m]
        if not row.empty:
            row = row.iloc[0]
            data.append([row[metric] for metric in metrics])
        else:
            data.append([np.nan]*len(metrics))
    data = np.array(data)

    # Plot agent vs ensemble comparison
    fig, axs = plt.subplots(1, len(metrics), figsize=(20, 5))
    for i, metric in enumerate(metrics):
        axs[i].bar(all_methods, data[:, i], color=[f'C{j}' for j in range(len(all_methods))])
        axs[i].set_title(metric.upper())
        axs[i].set_ylabel(metric)
        axs[i].set_xticklabels(all_methods, rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "agent_vs_ensemble_metrics.png"))
    

    # Plot agent domains, full training data, and highlight inducing points
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot all training data
    ax.scatter(full_train_x[:, 0], full_train_x[:, 1], c='lightgray', s=8, alpha=0.5, label='Full Training Data')

    # Plot agent domains and data
    agent_colors = ['red', 'blue', 'green', 'orange']
    for i, color in enumerate(agent_colors):
        x, y = load_agent_data(i)
        ax.scatter(x[:, 0], x[:, 1], c=color, s=18, alpha=0.7, label=f'Agent {i+1} Data')
        # Draw agent domain boundaries (2x2 grid)
        x1_bounds = np.linspace(full_train_x[:, 0].min(), full_train_x[:, 0].max(), 3)
        x2_bounds = np.linspace(full_train_x[:, 1].min(), full_train_x[:, 1].max(), 3)
        if i == 0:
            row, col = 0, 0
        elif i == 1:
            row, col = 0, 1
        elif i == 2:
            row, col = 1, 0
        elif i == 3:
            row, col = 1, 1
        x1_min = x1_bounds[col]
        x1_max = x1_bounds[col + 1]
        x2_min = x2_bounds[row]
        x2_max = x2_bounds[row + 1]
        ax.plot([x1_min, x1_max, x1_max, x1_min, x1_min],
                [x2_min, x2_min, x2_max, x2_max, x2_min],
                color=color, linewidth=2, alpha=0.8)

    # Highlight inducing points
    ax.scatter(inducing_points[:, 0], inducing_points[:, 1], c='black', s=40, marker='x', label='Inducing Points')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Agent Domains, Full Training Data, and Inducing Points')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "agent_domains_train_inducing.png"))
    plt.close()
