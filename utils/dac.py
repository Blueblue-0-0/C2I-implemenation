import numpy as np

class DACConsensus:
    def __init__(self, L, alpha=0.2):
        self.L = L
        self.alpha = alpha
        self.x_dac = None
        self.u_prev = None

    def reset(self, initial_values):
        self.x_dac = np.copy(initial_values)
        self.u_prev = np.copy(initial_values)

    def step(self, values):
        u = np.copy(values)
        self.x_dac = self.x_dac - self.alpha * (self.L @ self.x_dac) + (u - self.u_prev)
        self.u_prev = u
        return np.copy(self.x_dac)
