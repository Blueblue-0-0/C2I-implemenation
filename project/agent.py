import threading
import torch
import gpytorch
import numpy as np
from gp_model import VSGPRegressionModel
from train import train_gp

class Agent:
    def __init__(self, agent_id, inducing_points, train_x, train_y, neighbors, buffer_size=100, device='cpu'):
        self.id = agent_id
        self.device = device
        # Pass inducing_points as torch tensor directly
        self.model = VSGPRegressionModel(inducing_points=inducing_points, device=device).to(device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.train_x = train_x.to(device)
        self.train_y = train_y.to(device)
        self.neighbors = neighbors
        self.lock = threading.Lock()
        self.ready_for_dac = threading.Event()
        self.consensus_mean = None
        self.consensus_var = None
        self.buffer_size = buffer_size
        self.data_buffer_x = [self.train_x]
        self.data_buffer_y = [self.train_y]
        self.loss_history = []
        self.hypers = None

    def train_local(self, num_iter=100):
        # Use the updated train_gp function
        self.model, self.likelihood, self.loss_history, self.hypers = train_gp(
            self.model, self.likelihood, self.train_x, self.train_y, num_iter, self.device, 
        )

    def predict_mean_and_var(self, test_x):
        # Always inject consensus before prediction if available
        if self.consensus_mean is not None and self.consensus_var is not None:
            self.inject_consensus_to_variational()
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            pred = self.model(test_x.to(self.device))
            mean = pred.mean.cpu().numpy()
            var = pred.variance.cpu().numpy()
        return mean, var

    def update_data(self, new_x, new_y):
        self.data_buffer_x.append(new_x.to(self.device))
        self.data_buffer_y.append(new_y.to(self.device))
        self.train_x = torch.cat(self.data_buffer_x)
        self.train_y = torch.cat(self.data_buffer_y)
        # If buffer is full, retrain local model and signal ready for DAC
        if self.train_x.shape[0] >= self.buffer_size:
            self.train_local()
            self.ready_for_dac.set()

    def inject_consensus_to_variational(self):
        M = self.model.variational_strategy.inducing_points.size(0)
        device = self.device
        # Ensure consensus_mean is shape (M,)
        mean = np.asarray(self.consensus_mean).reshape(-1)
        assert mean.shape[0] == M, f"consensus_mean shape {mean.shape} does not match #inducing {M}"
        self.model.variational_strategy._variational_distribution.variational_mean.data = \
            torch.tensor(mean, dtype=torch.float32, device=device)
        # Ensure consensus_var is (M,) or (M, M)
        var = np.asarray(self.consensus_var)
        if var.ndim == 1:
            assert var.shape[0] == M, f"consensus_var shape {var.shape} does not match #inducing {M}"
            chol = torch.diag(torch.sqrt(torch.tensor(var, dtype=torch.float32, device=device)))
        else:
            assert var.shape == (M, M), f"consensus_var shape {var.shape} does not match ({M},{M})"
            chol = torch.linalg.cholesky(torch.tensor(var, dtype=torch.float32, device=device))
        self.model.variational_strategy._variational_distribution.chol_variational_covar.data = chol

    def evaluate_rmse(self, test_x, test_y):
        return self.model.evaluate_rmse(self.likelihood, test_x, test_y)

    def evaluate_nll(self, test_x, test_y):
        return self.model.evaluate_nll(self.likelihood, test_x, test_y)
