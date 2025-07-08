import json
import numpy as np
from pathlib import Path

from functions import (
    rastrigin,  grad_rastrigin,
    rosenbrock, grad_rosenbrock,
    custom_fn, grad_custom
)

# ------------------------------------------------------------------
def estimate_variance_numpy(true_grad, x_bar, sigma, batch_size=32, n_batches=200):
    """
    Monte‑Carlo estimate of Var[g(x_bar, ξ)] where
      g = true_grad(x_bar) + N(0, σ² I)/sqrt(batch_size).
    Returns σ̂² (scalar).
    """
    grads = []
    for _ in range(n_batches):
        noise = np.random.normal(scale=sigma, size=x_bar.shape) / np.sqrt(batch_size)
        grads.append(true_grad(x_bar) + noise)
    G = np.stack(grads)              # shape (n_batches, d)
    mu = G.mean(axis=0)
    sigma2 = np.mean(np.sum((G - mu) ** 2, axis=1))
    return sigma2


# ------------------------------------------------------------------
# Configuration
sigma_noise   = 0.1         # std‑dev of individual sample gradient noise
batch_size    = 32
n_batches     = 300         # Monte‑Carlo samples per test point
probe_points  = {
    "rastrigin":  [np.array([-4.,  4.]),
                   np.array([ 0.,  0.]),
                   np.array([ 3., -2.])],

    "rosenbrock": [np.array([-1.2,  1.]),
                   np.array([ 0. ,  0.]),
                   np.array([ 1.5,  1.5])],

    "custom_fn":  [np.array([-3.,  3.]),
                   np.array([ 0.,  0.]),
                   np.array([ 2., -1.])]
}

functions = {
    "rastrigin":  grad_rastrigin,
    "rosenbrock": grad_rosenbrock,
    "custom_fn":  grad_custom,
}

# ------------------------------------------------------------------
results = {}
for name, grad_f in functions.items():
    sigmas = []
    for x0 in probe_points[name]:
        s2 = estimate_variance_numpy(
            grad_f, x0, sigma=sigma_noise,
            batch_size=batch_size, n_batches=n_batches
        )
        sigmas.append(s2)

    mean_s2 = float(np.mean(sigmas))
    std_s2  = float(np.std(sigmas, ddof=1))
    results[name] = {"sigma2_mean": mean_s2, "sigma2_std": std_s2}

# save
out_path = Path("variance_estimates.json")
with out_path.open("w") as fh:
    json.dump(results, fh, indent=2)

print("Saved variance estimates to", out_path.absolute())
print(json.dumps(results, indent=2))
