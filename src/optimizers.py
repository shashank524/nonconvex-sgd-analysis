from __future__ import annotations
import numpy as np
from typing import Callable, Tuple


# ------------------------------------------------------------------
# helper utilities
# ------------------------------------------------------------------
def sample_ball(radius: float, dim: int) -> np.ndarray:
    """Uniform sample from ℓ2 ball in R^dim."""
    v = np.random.normal(size=dim)
    v /= np.linalg.norm(v)
    u = np.random.rand() ** (1.0 / dim)
    return radius * u * v


# ------------------------------------------------------------------
# 1. Full‑batch gradient descent
# ------------------------------------------------------------------
def gradient_descent(
grad: Callable[[np.ndarray], np.ndarray],
x0: np.ndarray,
step_size: float,
tol: float = 1e-5,
max_iters: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray]:
    x = x0.copy()
    path = [x.copy()]
    for _ in range(max_iters):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            break
        x -= step_size * g
        path.append(x.copy())
    return x, np.vstack(path)


# ------------------------------------------------------------------
# 2. Plain stochastic gradient descent
# ------------------------------------------------------------------
def stochastic_gradient_descent(
stochastic_grad: Callable[[np.ndarray], np.ndarray],
x0: np.ndarray,
step_size: float,
tol: float = 1e-3,
max_iters: int = 50_000,
ema_beta: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    x = x0.copy()
    g_ema = np.zeros_like(x0)
    path = [x.copy()]
    for _ in range(max_iters):
        g = stochastic_grad(x)
        g_ema = ema_beta * g_ema + (1 - ema_beta) * g
        if np.linalg.norm(g_ema) <= tol:
            break
        x -= step_size * g
        path.append(x.copy())
    return x, np.vstack(path)


# ------------------------------------------------------------------
# 3. Perturbed Stochastic Gradient Descent (PSGD)
# ------------------------------------------------------------------

def perturbed_sgd(
stochastic_grad: Callable[[np.ndarray], np.ndarray],
f: Callable[[np.ndarray], float],
x0: np.ndarray,
*,
ell: float,
rho: float,
eps: float = 1e-3,
sigma: float = 0.1,
delta: float = 0.1,
batch_size: int = 32,
c: float = 0.5,
ema_beta: float = 0.9,
max_iters: int = 100_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal implementation of the PSGD algorithm from the mid‑semester report.
    Variance‑related constants are heuristically estimated; this is meant for
    pedagogical demos, not tight theoretical experiments.
    """
    d = x0.size
    x = x0.copy()
    g_ema = np.zeros_like(x)

    # crude lower bound on optimum: running minimum of f
    f_star = f(x)

    # derived parameters per the write‑up
    chi = 3 * max(
        np.log(d * ell * max(f(x) - f_star, 1e-8) / (c * eps**2 * delta)),
        4.0,
    )
    g_thresh = np.sqrt(c) / chi**2 * eps + sigma / np.sqrt(batch_size)
    radius = np.sqrt(c) / chi**2 * eps / ell
    t_thresh = int(chi / c**2 * ell / np.sqrt(rho * eps))
    step_size = c / ell

    t_noise = -t_thresh - 1
    path = [x.copy()]

    for t in range(max_iters):
        g = stochastic_grad(x)
        g_ema = ema_beta * g_ema + (1 - ema_beta) * g

        # perturbation step
        if np.linalg.norm(g_ema) <= g_thresh and (t - t_noise) > t_thresh:
            x += sample_ball(radius, d)
            t_noise = t

        # SGD update
        x -= step_size * g
        path.append(x.copy())
        f_star = min(f_star, f(x))

        # termination
        if np.linalg.norm(g_ema) <= eps and (t - t_noise) > t_thresh:
            break

    return x, np.vstack(path)


# ------------------------------------------------------------------
# 4. Stochastic Gradient Descent with Momentum
# ------------------------------------------------------------------
def sgd_momentum(
    stochastic_grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    step_size: float,
    momentum: float = 0.9,
    tol: float = 1e-3,
    max_iters: int = 50_000,
    ema_beta: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vanilla SGD + Polyak momentum.

    Parameters
    ----------
    stochastic_grad : callable
        Returns g(x) — unbiased gradient estimate.
    step_size : float
        Learning rate η.
    momentum : float
        Momentum parameter γ in [0,1).  γ=0 → plain SGD.
    ema_beta : float
        Separate EMA coefficient used only for termination criterion.
    """
    x = x0.copy()
    v = np.zeros_like(x0)      # momentum buffer
    g_ema = np.zeros_like(x0)  # for stopping test
    path = [x.copy()]

    for _ in range(max_iters):
        g = stochastic_grad(x)
        g_ema = ema_beta * g_ema + (1 - ema_beta) * g
        if np.linalg.norm(g_ema) <= tol:
            break

        v = momentum * v + g
        x = x - step_size * v
        path.append(x.copy())

    return x, np.vstack(path)

