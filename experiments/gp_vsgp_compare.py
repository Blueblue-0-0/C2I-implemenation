import torch
import math
import gpytorch
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

# ==== Training and Model Parameters (easy to modify) ====
N = 300  # Number of training points
TEST_POINTS = 1000  # Number of test points
NOISE_LEVEL = 0.1
FUNCTION_FREQUENCY = 2 * math.pi
NUM_INDUCING = 20
TRAIN_ITER = 100
LEARNING_RATE = 0.02

# ==== Plotting Parameters ====
FIGURE_SIZE = (10, 6)
INDUCING_MARKER_SIZE = 80
DATA_POINT_SIZE = 10
CONFIDENCE_ALPHA = 0.3

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Results Directory ====
results_dir = os.path.join(os.path.dirname(__file__), '..', f'results_{NUM_INDUCING}')
os.makedirs(results_dir, exist_ok=True)

# ==== Data Generation ====
train_x = torch.linspace(0, 1, N, device=device)
train_y = torch.sin(FUNCTION_FREQUENCY * train_x) + NOISE_LEVEL * torch.randn(train_x.size(), device=device)
test_x = torch.linspace(0, 1, TEST_POINTS, device=device)
test_y = torch.sin(FUNCTION_FREQUENCY * test_x)

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

# --- VSGP Model ---
class VSGPRegressionModel(gpytorch.models.ApproximateGP):
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

# --- FITC Sparse GP Model ---
class FITCRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, learn_inducing_locations=True):
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(0)).to(device)
        variational_strategy = gpytorch.variational.FITCVariationalStrategy(
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

def train_gp(model, likelihood, train_x, train_y, num_iter=TRAIN_ITER, is_vsgp=False):
    model.train()
    likelihood.train()
    if is_vsgp:
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=LEARNING_RATE)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0)).to(device)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device)
    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(train_x.unsqueeze(-1) if is_vsgp else train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    return model, likelihood

def evaluate(model, likelihood, test_x, test_y, is_vsgp=False):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x.unsqueeze(-1) if is_vsgp else test_x))
        mean = pred.mean.cpu().numpy()
        rmse = np.sqrt(mean_squared_error(test_y.cpu().numpy(), mean))
    return mean, pred, rmse

# --- Train and evaluate Full GP ---
likelihood_full = gpytorch.likelihoods.GaussianLikelihood().to(device)
model_full = ExactGPModel(train_x, train_y, likelihood_full).to(device)
model_full, likelihood_full = train_gp(model_full, likelihood_full, train_x, train_y, num_iter=TRAIN_ITER, is_vsgp=False)
mean_full, pred_full, rmse_full = evaluate(model_full, likelihood_full, test_x, test_y, is_vsgp=False)

# --- Train and evaluate VSGP (fixed inducing locations) ---
inducing_points = torch.linspace(0, 1, NUM_INDUCING, device=device).unsqueeze(-1)
likelihood_vsgp_fixed = gpytorch.likelihoods.GaussianLikelihood().to(device)
model_vsgp_fixed = VSGPRegressionModel(inducing_points, learn_inducing_locations=False).to(device)
model_vsgp_fixed, likelihood_vsgp_fixed = train_gp(model_vsgp_fixed, likelihood_vsgp_fixed, train_x, train_y, num_iter=TRAIN_ITER, is_vsgp=True)
mean_vsgp_fixed, pred_vsgp_fixed, rmse_vsgp_fixed = evaluate(model_vsgp_fixed, likelihood_vsgp_fixed, test_x, test_y, is_vsgp=True)

# --- Train and evaluate VSGP (learned inducing locations) ---
likelihood_vsgp_learn = gpytorch.likelihoods.GaussianLikelihood().to(device)
model_vsgp_learn = VSGPRegressionModel(inducing_points.clone(), learn_inducing_locations=True).to(device)
model_vsgp_learn, likelihood_vsgp_learn = train_gp(model_vsgp_learn, likelihood_vsgp_learn, train_x, train_y, num_iter=TRAIN_ITER, is_vsgp=True)
mean_vsgp_learn, pred_vsgp_learn, rmse_vsgp_learn = evaluate(model_vsgp_learn, likelihood_vsgp_learn, test_x, test_y, is_vsgp=True)

# --- Plotting: Function and predictions with learned inducing points marked ---
plt.figure(figsize=FIGURE_SIZE)
plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k.', alpha=0.2, label='Train Data')
plt.plot(test_x.cpu().numpy(), test_y.cpu().numpy(), 'g--', label='True Function')
plt.plot(test_x.cpu().numpy(), mean_full, 'b', label=f'Full GP (RMSE={rmse_full:.3f})')
plt.plot(test_x.cpu().numpy(), mean_vsgp_fixed, 'r', label=f'VSGP Fixed (RMSE={rmse_vsgp_fixed:.3f})')
plt.plot(test_x.cpu().numpy(), mean_vsgp_learn, 'm', label=f'VSGP Learned (RMSE={rmse_vsgp_learn:.3f})')

# Mark learned inducing points on the function plot
learned_inducing = model_vsgp_learn.variational_strategy.inducing_points.detach().cpu().numpy().flatten()
plt.scatter(learned_inducing, np.sin(FUNCTION_FREQUENCY * learned_inducing), 
            marker='x', color='magenta', s=INDUCING_MARKER_SIZE, label='Learned Inducing Points')

plt.legend()
plt.title('Full GP vs VSGP (Fixed vs Learned Inducing Locations)')
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