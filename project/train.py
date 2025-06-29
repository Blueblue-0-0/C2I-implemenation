import os
import torch
import gpytorch

def train_gp(model, likelihood=None, train_x=None, train_y=None, num_iter=100, device='cpu'):
    # Set GaussianLikelihood as default if not provided
    if likelihood is None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0)).to(device)

    loss_history = []
    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(train_x.to(device))
        loss = -mll(output, train_y.to(device).squeeze(-1))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    # Collect optimized hyperparameters
    hypers = {
        "lengthscale": model.covar_module.base_kernel.lengthscale.detach().cpu().numpy(),
        "outputscale": model.covar_module.outputscale.detach().cpu().numpy(),
        "likelihood_noise": likelihood.noise.detach().cpu().numpy()
    }

    return model, likelihood, loss_history, hypers

