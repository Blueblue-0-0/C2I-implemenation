import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 5
T = 50
DT = 2
NOISE_STDS = [0.1, 0.2, 0.3, 0.4, 0.5]

def local_input(t, agent_id):
    return np.sin(0.2 * t + 0.2 *agent_id) + 0.5 *np.random.normal(0, NOISE_STDS[agent_id])

# Ring Laplacian
A = np.zeros((N, N))
for i in range(N):
    A[i, (i-1)%N] = 1
    A[i, (i+1)%N] = 1
D = np.diag(A.sum(axis=1))
L = D - A

# DAC Gains
alpha = 0.1
beta = 0.1

# Static AC parameters
static_alpha = 0.1
static_steps = 5  # Number of static AC iterations per time step

# --- DAC initialization ---
u0 = np.array([local_input(0, i) for i in range(N)])
x_dac = np.ones(N) * np.mean(u0)
v = np.zeros(N)
u_prev = u0.copy()

history_dac = []
history_static = []
all_y = []

for t_idx in range(T):
    t = t_idx * DT
    u = np.array([local_input(t, i) for i in range(N)])
    all_y.append(u)
    
    # --- DAC update ---
    x_dac = x_dac - alpha * (L @ x_dac) + v + (u - u_prev)
    v = v - beta * (L @ x_dac)
    u_prev = u.copy()
    history_dac.append(x_dac.copy())
    
    # --- Static AC: re-initialize x to current measurements, run a few steps ---
    x_static = u.copy()
    for _ in range(static_steps):
        x_static = x_static - static_alpha * (L @ x_static)
    history_static.append(x_static.copy())

history_dac = np.array(history_dac)
history_static = np.array(history_static)
all_y = np.array(all_y)
true_avg = np.mean(all_y, axis=1)

# --- Plot ---
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(history_dac[:, i], color='b', linewidth=1, label='DAC' if i == 0 else "")
    plt.plot(history_static[:, i], color='m', linestyle='--', linewidth=1, label='Static AC' if i == 0 else "")
    plt.scatter(np.arange(T), all_y[:, i], color='k', marker='x', s=40, alpha=0.7, label='Measurements' if i == 0 else "")
plt.plot(true_avg, 'r--', label='True Average', linewidth=2)
plt.xlabel('Time')
plt.ylabel(r"$x^i$")
plt.title('Dynamic AC (blue), Static AC (magenta, re-initialized), and Measurements')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()