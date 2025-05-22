import threading
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import gpytorch

# === Parameters ===
N_AGENTS = 5
T_STEPS = 100
DT = 0.1
NOISE_STDS = [0.1, 0.2, 0.3, 0.4, 0.5]  # Different std for each agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulated time-varying input for each agent (with agent-specific noise)
def input_signal(t, agent_id):
    true_val = np.sin(0.2 * t + agent_id)
    noise = np.random.normal(0, NOISE_STDS[agent_id])
    return true_val + noise

# === Dynamic Average Consensus ===
class DynamicConsensus:
    def __init__(self, N):
        self.N = N
        self.x = np.zeros(N)  # state estimate
        self.v = np.zeros(N)  # integral term
        self.y = np.zeros(N)  # current inputs
        self.history = []

    def step(self, t):
        self.y = np.array([input_signal(t, i) for i in range(self.N)])
        for i in range(self.N):
            neighbors = [j for j in range(self.N) if j != i]
            diff = sum(self.x[j] - self.x[i] for j in neighbors)
            self.v[i] += diff * DT
            self.x[i] = self.v[i] + self.y[i]
        self.history.append(self.x.copy())

# === GPyTorch GP Model for each agent ===
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class LocalGP:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.X = []
        self.y = []
        self.history = []
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        # Dummy initial data to initialize the model
        init_x = torch.tensor([0.0], dtype=torch.float32, device=device)
        init_y = torch.tensor([0.0], dtype=torch.float32, device=device)
        self.model = ExactGPModel(init_x, init_y, self.likelihood).to(device)

    def update(self, t, signal):
        self.X.append([t])
        self.y.append(signal)
        # Convert to torch tensors
        train_x = torch.tensor(self.X, dtype=torch.float32, device=device)
        train_y = torch.tensor(self.y, dtype=torch.float32, device=device)
        # Re-initialize model with all data so far
        self.model.set_train_data(inputs=train_x, targets=train_y, strict=False)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        # Only a few steps for online update
        for _ in range(10):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        # Predict at current time
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(torch.tensor([[t]], dtype=torch.float32, device=device)))
            self.history.append(pred.mean.cpu().item())

# === Simulation ===
consensus = DynamicConsensus(N_AGENTS)
gp_models = [LocalGP(i) for i in range(N_AGENTS)]

for t_idx in range(T_STEPS):
    t = t_idx * DT
    consensus.step(t)
    for i in range(N_AGENTS):
        signal = input_signal(t, i)
        gp_models[i].update(t, signal)

# === Plotting ===
history = np.array(consensus.history)
gp_outputs = np.array([m.history for m in gp_models])

plt.figure(figsize=(12, 6))
for i in range(N_AGENTS):
    plt.plot(history[:, i], label=f'Consensus Agent {i}', linestyle='--')
    plt.plot(gp_outputs[i], label=f'GP Model {i}', alpha=0.6)
plt.title('Dynamic Consensus and GPyTorch GP Estimations\n(Each agent: different noise std)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
