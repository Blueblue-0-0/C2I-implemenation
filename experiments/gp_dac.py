import torch
import gpytorch
import threading
import queue
import time
import numpy as np

# Simulated "network" between agents
message_queues = [queue.Queue(), queue.Queue()]

# Shared x_star where predictions are made
x_star = torch.linspace(0, 1, 20).unsqueeze(-1)

# Define a minimal GP model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )

# Dynamic average consensus (DAC) loop
def dynamic_average_consensus(agent_id, local_pred, queue_in, queue_out):
    avg = local_pred.clone()
    for t in range(50):
        try:
            neighbor_pred = queue_in.get(timeout=0.1)
            avg = avg + 0.5 * (neighbor_pred - avg)
            queue_out.put(avg.clone())
        except queue.Empty:
            pass
        time.sleep(0.1)
    print(f"Agent {agent_id} final DAC output:\n{avg.squeeze().numpy()}")

# Each agent runs local GP and DAC
def run_agent(agent_id, x, y):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(x, y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(30):
        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        pred = model(x_star).mean

    # Start DAC loop in parallel
    t = threading.Thread(
        target=dynamic_average_consensus,
        args=(agent_id, pred, message_queues[1 - agent_id], message_queues[agent_id])
    )
    t.start()
    return t

# Simulate two agents with different local data
x1 = torch.linspace(0, 0.5, 10).unsqueeze(-1)
y1 = torch.sin(x1 * (2 * np.pi)) + 0.1 * torch.randn_like(x1)

x2 = torch.linspace(0.5, 1.0, 10).unsqueeze(-1)
y2 = torch.sin(x2 * (2 * np.pi)) + 0.3 * torch.randn_like(x2)

# Launch both agents
t1 = run_agent(0, x1, y1)
t2 = run_agent(1, x2, y2)

t1.join()
t2.join()
