import numpy as np

def rastrigin(x, A=10):
    """Rastrigin function (multimodal, non-convex)."""
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def grad_rastrigin(x, A=10):
    """Gradient of the Rastrigin function."""
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

def rosenbrock(x, a=1, b=100):
    """Rosenbrock (banana) function in 2D."""
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def grad_rosenbrock(x, a=1, b=100):
    """Gradient of the Rosenbrock function."""
    dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    dy = 2 * b * (x[1] - x[0]**2)
    return np.array([dx, dy])

def custom_fn(x):
    """Custom multimodal function."""
    return x[0]**2 + x[1]**2 + np.sin(3*x[0]) * np.sin(3*x[1])

def grad_custom(x):
    """Gradient of the custom function."""
    dx = 2*x[0] + 3*np.cos(3*x[0]) * np.sin(3*x[1])
    dy = 2*x[1] + 3*np.sin(3*x[0]) * np.cos(3*x[1])
    return np.array([dx, dy])