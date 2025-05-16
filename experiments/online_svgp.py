import torch
import math
import gpytorch
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)

# 1. Initial data and streaming data
init_x = torch.linspace(0, 0.3, 20, device=device)
init_y = torch.sin(2 * math.pi * init_x) + 0.2 * torch.randn(init_x.size(), device=device)
stream_x = torch.linspace(0.3, 1, 180, device=device)
stream_y = torch.sin(2 * math.pi * stream_x) + 0.2 * torch.randn(stream_x.size(), device=device)

# Fixed test set for evaluation
test_x = torch.linspace(0, 1, 500, device=device)
test_y = torch.sin(2 * math.pi * test_x)

# 2. SVGP Model definition
class SVGPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0)).to(device)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean().to(device)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel().to(device)
        ).to(device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# 3. Initialize model and likelihood
num_inducing = 20
inducing_points = init_x.unsqueeze(-1).clone()
model = SVGPRegressionModel(inducing_points=inducing_points).to(device)
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

# 4. Training function
def train_svgp(model, likelihood, x, y, num_iter=50):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y.size(0)).to(device)
    for _ in range(num_iter):
        optimizer.zero_grad()
        output = model(x.unsqueeze(-1))
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()

# 5. Online update loop
batch_size = 10
all_x = [init_x]
all_y = [init_y]
inducing_traj = [model.variational_strategy.inducing_points.detach().cpu().numpy().flatten().copy()]
rmse_list = []
lengthscale_list = []
outputscale_list = []
noise_list = []

# Initial training
train_svgp(model, likelihood, init_x, init_y, num_iter=100)
inducing_traj.append(model.variational_strategy.inducing_points.detach().cpu().numpy().flatten().copy())

# Evaluate after initial training
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(model(test_x.unsqueeze(-1)))
    mean = pred.mean.cpu().numpy()
    rmse = np.sqrt(mean_squared_error(test_y.cpu().numpy(), mean))
    rmse_list.append(rmse)
    # Record kernel hyperparameters
    lengthscale_list.append(model.covar_module.base_kernel.lengthscale.item())
    outputscale_list.append(model.covar_module.outputscale.item())
    noise_list.append(likelihood.noise.item())

for i in range(0, len(stream_x), batch_size):
    batch_x = stream_x[i:i+batch_size]
    batch_y = stream_y[i:i+batch_size]
    # Add new data
    all_x.append(batch_x)
    all_y.append(batch_y)
    # Retrain on all data so far (for simplicity)
    x_so_far = torch.cat(all_x)
    y_so_far = torch.cat(all_y)
    train_svgp(model, likelihood, x_so_far, y_so_far, num_iter=20)
    # Record inducing points
    inducing_traj.append(model.variational_strategy.inducing_points.detach().cpu().numpy().flatten().copy())
    # Evaluate and record RMSE and kernel hyperparameters
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x.unsqueeze(-1)))
        mean = pred.mean.cpu().numpy()
        rmse = np.sqrt(mean_squared_error(test_y.cpu().numpy(), mean))
        rmse_list.append(rmse)
        lengthscale_list.append(model.covar_module.base_kernel.lengthscale.item())
        outputscale_list.append(model.covar_module.outputscale.item())
        noise_list.append(likelihood.noise.item())

# 6. Plot inducing point trajectories with RMSE tags
inducing_traj = np.array(inducing_traj)  # shape: (steps, num_inducing)
plt.figure(figsize=(12, 6))
for i in range(num_inducing):
    plt.plot(inducing_traj[:, i], color='magenta', alpha=0.7)
for step, rmse in enumerate(rmse_list):
    plt.text(step, inducing_traj[step, 0], f"{rmse:.3f}", fontsize=8, color='blue', ha='center', va='bottom', rotation=45)
plt.xlabel('Update Step')
plt.ylabel('Inducing Point Location')
plt.title('Inducing Point Trajectories During Online SVGP Learning\n(RMSE at each step)')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'online_svgp_inducing_trajectories.png'))
plt.close()

# 7. Plot kernel hyperparameters over time
plt.figure(figsize=(10, 6))
plt.plot(lengthscale_list, label='Lengthscale')
plt.plot(outputscale_list, label='Outputscale')
plt.plot(noise_list, label='Likelihood Noise')
plt.xlabel('Update Step')
plt.ylabel('Value')
plt.title('Kernel Hyperparameters During Online SVGP Learning')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'online_svgp_hyperparams.png'))
plt.close()

# 8. Plot RMSE over time
plt.figure(figsize=(8, 4))
plt.plot(rmse_list, marker='o')
plt.xlabel('Update Step')
plt.ylabel('RMSE on Test Set')
plt.title('Test RMSE During Online SVGP Learning')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'online_svgp_rmse_curve.png'))
plt.close()

# 9. Plot final model fit and inducing points
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(model(test_x.unsqueeze(-1)))
    mean = pred.mean.cpu().numpy()
    lower, upper = pred.confidence_region()
    lower = lower.cpu().numpy()
    upper = upper.cpu().numpy()

plt.figure(figsize=(10, 6))
plt.plot(test_x.cpu().numpy(), np.sin(2 * np.pi * test_x.cpu().numpy()), 'g--', label='True Function')
plt.plot(test_x.cpu().numpy(), mean, 'b', label='SVGP Predictive Mean')
plt.fill_between(test_x.cpu().numpy(), lower, upper, alpha=0.3, label='Confidence')
plt.scatter(torch.cat(all_x).cpu().numpy(), torch.cat(all_y).cpu().numpy(), color='k', s=10, alpha=0.3, label='Data')
final_inducing = model.variational_strategy.inducing_points.detach().cpu().numpy().flatten()
plt.scatter(final_inducing, np.sin(2 * np.pi * final_inducing), marker='x', color='magenta', s=80, label='Final Inducing Points')
plt.legend()
plt.title('Online SVGP: Final Fit and Inducing Points')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'online_svgp_final_fit.png'))
plt.close()

print(f"Done! Plots saved in '{results_dir}'.")
