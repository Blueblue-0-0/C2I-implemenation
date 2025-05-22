import numpy as np
import matplotlib.pyplot as plt

def consensus_update(L, x, alpha=0.2):
    return x - alpha * L @ x

def simulate_consensus(L, x0, tol=1e-3, max_steps=200):
    x = x0.copy()
    history = [x.copy()]
    for step in range(max_steps):
        x = consensus_update(L, x)
        history.append(x.copy())
        if np.max(np.abs(x - np.mean(x))) < tol:
            break
    return np.array(history), step+1

def laplacian_eigenvalues(L):
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return eigvals

# --- Define different graph topologies (undirected) ---
N = 8

# Ring graph
A_ring = np.zeros((N, N))
for i in range(N):
    A_ring[i, (i-1)%N] = 1
    A_ring[i, (i+1)%N] = 1

# Complete graph
A_complete = np.ones((N, N)) - np.eye(N)

# Line graph
A_line = np.zeros((N, N))
for i in range(N-1):
    A_line[i, i+1] = 1
    A_line[i+1, i] = 1

# Star graph
A_star = np.zeros((N, N))
for i in range(1, N):
    A_star[0, i] = 1
    A_star[i, 0] = 1

graphs = {
    "Ring": A_ring,
    "Complete": A_complete,
    "Line": A_line,
    "Star": A_star,
}

# --- Run consensus and collect results ---
np.random.seed(42)
x0 = np.random.randn(N)
results = []

for name, A in graphs.items():
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals = laplacian_eigenvalues(L)
    lambda2 = eigvals[1]
    history, steps = simulate_consensus(L, x0)
    results.append((name, lambda2, steps, history))
    print(f"{name}: lambda_2={lambda2:.4f}, steps to converge={steps}")

# --- Plot convergence curves as discrete points ---
plt.figure(figsize=(10, 6))
for name, lambda2, steps, history in results:
    errors = np.abs(history - np.mean(x0))
    plt.plot(
        np.mean(errors, axis=1),
        marker='o',
        linestyle='',
        label=rf"{name} ($\lambda_2$={lambda2:.3f})"
    )
plt.yscale('log')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel(r'Consensus Error $\|x - \bar{x}\|$ (log scale)', fontsize=14)
plt.title('Consensus Convergence for Different Graph Topologies', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("consensus_convergence.png", dpi=300)
plt.show()

# --- Plot λ₂ vs. convergence steps ---
plt.figure(figsize=(6, 4))
lambdas = [r[1] for r in results]
steps = [r[2] for r in results]
plt.scatter(lambdas, steps, s=80, c='tab:blue', edgecolors='k')
for name, lambda2, step, _ in results:
    plt.text(lambda2, step, name, fontsize=12, ha='right', va='bottom')
plt.xlabel(r'Second Smallest Eigenvalue $\lambda_2$', fontsize=14)
plt.ylabel('Steps to Converge', fontsize=14)
plt.title(r'$\lambda_2$ vs. Consensus Convergence Speed', fontsize=16)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("lambda2_vs_convergence.png", dpi=300)
plt.show()