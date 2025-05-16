import torch
import math
import gpytorch
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results_20')
os.makedirs(results_dir, exist_ok=True)

# Generate synthetic data: y = sin(2πx) + noise
N = 300
train_x = torch.linspace(0, 1, N, device=device)
train_y = torch.sin(2 * math.pi * train_x) + 0.2 * torch.randn(train_x.size(), device=device)
test_x = torch.linspace(0, 1, 1000, device=device)
test_y = torch.sin(2 * math.pi * test_x)

# --- Full GP Model ---
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- SVGP Model ---
class SVGPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, learn_inducing_locations=True):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0)).to(device)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_locations
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

def train_gp(model, likelihood, train_x, train_y, num_iter=300, is_svgp=False):
    model.train()
    likelihood.train()
    if is_svgp:
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0)).to(device)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device)
    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(train_x.unsqueeze(-1) if is_svgp else train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    return model, likelihood

def evaluate(model, likelihood, test_x, test_y, is_svgp=False):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x.unsqueeze(-1) if is_svgp else test_x))
        mean = pred.mean.cpu().numpy()
        rmse = np.sqrt(mean_squared_error(test_y.cpu().numpy(), mean))
    return mean, pred, rmse

# --- Train and evaluate Full GP ---
likelihood_full = gpytorch.likelihoods.GaussianLikelihood().to(device)
model_full = ExactGPModel(train_x, train_y, likelihood_full).to(device)
model_full, likelihood_full = train_gp(model_full, likelihood_full, train_x, train_y, num_iter=300, is_svgp=False)
mean_full, pred_full, rmse_full = evaluate(model_full, likelihood_full, test_x, test_y, is_svgp=False)

# --- Train and evaluate SVGP (fixed inducing locations) ---
num_inducing = 20
inducing_points = torch.linspace(0, 1, num_inducing, device=device).unsqueeze(-1)
likelihood_svgp_fixed = gpytorch.likelihoods.GaussianLikelihood().to(device)
model_svgp_fixed = SVGPRegressionModel(inducing_points, learn_inducing_locations=False).to(device)
model_svgp_fixed, likelihood_svgp_fixed = train_gp(model_svgp_fixed, likelihood_svgp_fixed, train_x, train_y, num_iter=300, is_svgp=True)
mean_svgp_fixed, pred_svgp_fixed, rmse_svgp_fixed = evaluate(model_svgp_fixed, likelihood_svgp_fixed, test_x, test_y, is_svgp=True)

# --- Train and evaluate SVGP (learned inducing locations) ---
likelihood_svgp_learn = gpytorch.likelihoods.GaussianLikelihood().to(device)
model_svgp_learn = SVGPRegressionModel(inducing_points.clone(), learn_inducing_locations=True).to(device)
model_svgp_learn, likelihood_svgp_learn = train_gp(model_svgp_learn, likelihood_svgp_learn, train_x, train_y, num_iter=300, is_svgp=True)
mean_svgp_learn, pred_svgp_learn, rmse_svgp_learn = evaluate(model_svgp_learn, likelihood_svgp_learn, test_x, test_y, is_svgp=True)

# --- Plotting: Function and predictions with learned inducing points marked ---
plt.figure(figsize=(10, 6))
plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k.', alpha=0.2, label='Train Data')
plt.plot(test_x.cpu().numpy(), test_y.cpu().numpy(), 'g--', label='True Function')
plt.plot(test_x.cpu().numpy(), mean_full, 'b', label=f'Full GP (RMSE={rmse_full:.3f})')
plt.plot(test_x.cpu().numpy(), mean_svgp_fixed, 'r', label=f'SVGP Fixed (RMSE={rmse_svgp_fixed:.3f})')
plt.plot(test_x.cpu().numpy(), mean_svgp_learn, 'm', label=f'SVGP Learned (RMSE={rmse_svgp_learn:.3f})')

# Mark learned inducing points on the function plot
learned_inducing = model_svgp_learn.variational_strategy.inducing_points.detach().cpu().numpy().flatten()
plt.scatter(learned_inducing, np.sin(2 * np.pi * learned_inducing), 
            marker='x', color='magenta', s=80, label='Learned Inducing Points')

plt.legend()
plt.title('Full GP vs SVGP (Fixed vs Learned Inducing Locations)')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'gp_comparison_with_inducing.png'))
plt.close()

# --- Show inducing points locations ---
plt.figure(figsize=(8, 2))
plt.scatter(inducing_points.cpu().numpy(), np.zeros_like(inducing_points.cpu().numpy()), label='Initial Inducing (Fixed)', color='red')
plt.scatter(learned_inducing, np.ones_like(learned_inducing), label='Learned Inducing', color='magenta', marker='x')
plt.yticks([0, 1], ['Fixed', 'Learned'])
plt.title('Inducing Points Locations')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'inducing_points.png'))
plt.close()

print(f"All done! Plots saved in '{results_dir}'.")