import numpy as np

DEFAULT_PARAMS = {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 8.0 / 3.0,
}

DEFAULT_IC = [1.0, 1.0, 1.0]
DEFAULT_H = 0.01
DEFAULT_T_SPAN = (0, 40)


def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    Lorenz system rate function.
    Returns a length-3 NumPy array [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


def make_lorenz(rho=None, sigma=None, beta=None):
    """Return a Lorenz rate function with fixed parameters."""
    params = DEFAULT_PARAMS.copy()
    if sigma is not None:
        params["sigma"] = sigma
    if rho is not None:
        params["rho"] = rho
    if beta is not None:
        params["beta"] = beta

    def rate_fn(t, state):
        return lorenz(t, state, **params)

    return rate_fn
