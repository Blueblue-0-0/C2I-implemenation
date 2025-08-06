import numpy as np

def poe_ensemble(means, variances):
    """Product of Experts: precision-weighted mean/variance."""
    precisions = 1.0 / (variances + 1e-8)
    mean = np.sum(means * precisions, axis=0) / np.sum(precisions, axis=0)
    var = 1.0 / np.sum(precisions, axis=0)
    return mean, var

def pcm_ensemble(means, variances):
    """Product of Committee Members: equal-weighted mean/variance."""
    mean = np.mean(means, axis=0)
    var = np.mean(variances, axis=0)
    return mean, var

def moe_ensemble(means, variances, gating_weights):
    """Mixture of Experts: input-dependent gating (weights shape: [n_agents, n_points])."""
    mean = np.sum(gating_weights * means, axis=0)
    var = np.sum(gating_weights * (variances + means**2), axis=0) - mean**2
    return mean, var

def get_moe_gating_weights(means, variances):
        # Example: inverse variance weighting
        variances = np.array(variances)
        inv_vars = 1.0 / (variances + 1e-8)
        weights = inv_vars / np.sum(inv_vars, axis=0)
        return weights

def evaluate_ensemble(mean, var, test_x, test_y):
    # mean: [n_points], var: [n_points]
    # test_x: [n_points, input_dim], test_y: [n_points]
    pred_mean = mean  # If mean is already at test_x, otherwise interpolate
    mse = np.mean((pred_mean - test_y) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_mean - test_y))
    ss_res = np.sum((test_y - pred_mean) ** 2)
    ss_tot = np.sum((test_y - np.mean(test_y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return dict(r2=r2, rmse=rmse, mae=mae, mse=mse)