import numpy as np


def rk4(f, t_span, y0, h):
    """
    Solve dy/dt = f(t, y) using the classical 4th-order Runge-Kutta method.

    Parameters:
        f : callable  – rate function f(t, y) returning a NumPy array
        t_span : tuple – (t_start, t_end)
        y0 : array-like – initial state vector
        h : float – time step

    Returns:
        t : 1-D array of time values
        y : 2-D array, shape (len(t), len(y0))
    """
    t = np.arange(t_span[0], t_span[1], h)
    y = np.zeros((len(t), len(y0)))
    y[0] = np.array(y0, dtype=float)

    for i in range(len(t) - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + 0.5 * h, y[i] + 0.5 * h * k1)
        k3 = f(t[i] + 0.5 * h, y[i] + 0.5 * h * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y
