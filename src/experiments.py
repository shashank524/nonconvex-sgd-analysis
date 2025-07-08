# experiments.py ----------------------------------------------------------
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from functions import (
    rastrigin,  grad_rastrigin,
    rosenbrock, grad_rosenbrock,
    custom_fn, grad_custom,
)
from optimizers import (          # <<<  updated import list
    gradient_descent,
    stochastic_gradient_descent,
    sgd_momentum,
    perturbed_sgd,                # <<<  plain PSGD
)

# ------------------------------------------------------------------------
# load pre‑computed mini‑batch‑variance estimates
var_path = Path("variance_estimates.json")
with var_path.open() as fh:
    var_dict = json.load(fh)

def sigma_for(name: str) -> float:
    """Return √(σ²) for the requested problem."""
    return float(np.sqrt(var_dict[name]["sigma2_mean"]))

# global hyper‑params -----------------------------------------------------
batch_size     = 32
sgd_lr         = 1e-2     # baseline SGD step size
momentum_lr    = 1e-2
momentum_gamma = 0.9
gd_lr          = 1e-3
eps_tol        = 1e-4

# rough smoothness / Hessian‑L estimates for toy functions
L_est, rho_est = 1e3, 1e4

# problem definitions -----------------------------------------------------
problems = {
    "rastrigin": {
        "f": rastrigin,
        "grad": grad_rastrigin,
        "x0": np.array([-4.,  4.]),
    },
    "rosenbrock": {
        "f": rosenbrock,
        "grad": grad_rosenbrock,
        "x0": np.array([-1.2, 1.0]),
    },
    "custom_fn": {
        "f": custom_fn,
        "grad": grad_custom,
        "x0": np.array([-3.,  3.]),
    },
}

# run all optimizers ------------------------------------------------------
histories = {}          # histories[problem][method] -> list[f(x)]

def make_stochastic_grad(g_true, sigma, batch):
    """Unbiased mini‑batch gradient oracle with N(0, σ² I/B) noise."""
    def oracle(x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(scale=sigma, size=(batch, *x.shape))
        return g_true(x) + noise.mean(axis=0)
    return oracle

for name, cfg in problems.items():
    f, grad, x0 = cfg["f"], cfg["grad"], cfg["x0"]
    sigma = sigma_for(name)
    stoch_grad = make_stochastic_grad(grad, sigma, batch_size)

    histories[name] = {}

    # ---- full GD --------------------------------------------------------
    _, path_gd = gradient_descent(
        grad, x0, step_size=gd_lr, tol=eps_tol, max_iters=100_000
    )
    histories[name]["GD"] = [f(p) for p in path_gd]

    # ---- plain SGD ------------------------------------------------------
    _, path_sgd = stochastic_gradient_descent(
        stoch_grad, x0, step_size=sgd_lr, tol=eps_tol, max_iters=100_000
    )
    histories[name]["SGD"] = [f(p) for p in path_sgd]

    # ---- SGD + momentum -------------------------------------------------
    _, path_mom = sgd_momentum(
        stoch_grad, x0,
        step_size=momentum_lr, momentum=momentum_gamma,
        tol=eps_tol, max_iters=100_000
    )
    histories[name]["Momentum"] = [f(p) for p in path_mom]

    # ---- Perturbed SGD (plain) -----------------------------------------
    _, path_psgd = perturbed_sgd(
        stoch_grad, f, x0,
        ell=L_est, rho=rho_est,
        eps=1e-3, sigma=sigma, delta=0.1,
        batch_size=batch_size, c=0.5, ema_beta=0.9,
        max_iters=100_000,
    )
    histories[name]["PSGD"] = [f(p) for p in path_psgd]

# ------------------------------------------------------------------------
# plotting ---------------------------------------------------------------
colors = {
    "GD": "k",
    "SGD": "tab:blue",
    "Momentum": "tab:green",
    "PSGD": "tab:red",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, series) in zip(axes, histories.items()):
    for label, vals in series.items():
        ax.plot(vals, label=label, color=colors[label])
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("f(x)")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    if ax is axes[1]:
        ax.legend()

fig.suptitle("Objective convergence on three toy functions")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("convergence_curves.png", dpi=120)
plt.show()
print("Saved plot to convergence_curves.png")
