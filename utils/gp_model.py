import gpytorch
import torch
import numpy as np
from sklearn.metrics import mean_squared_error

class VSGPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, device='cpu', likelihood=None):
        # inducing_points: torch tensor of shape (num_inducing, D)
        inducing_points = inducing_points.to(device)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=False
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # Set GaussianLikelihood as default if not provided
        self.likelihood = likelihood if likelihood is not None else gpytorch.likelihoods.GaussianLikelihood().to(device)

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)

    def evaluate_rmse(self, test_x, test_y):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self(test_x))
            mean = preds.mean.cpu().numpy()
            rmse = np.sqrt(mean_squared_error(test_y.cpu().numpy(), mean))
        return rmse

    def evaluate_nll(self, test_x, test_y):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self(test_x))
            mean = preds.mean.cpu().numpy()
            var = preds.variance.cpu().numpy()
            nll = 0.5 * np.log(2 * np.pi * var) + 0.5 * ((test_y.cpu().numpy() - mean) ** 2) / var
            nll = np.mean(nll)
        return nll
