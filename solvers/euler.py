import numpy as np


def forward_euler(f, t_span, y0, h):
    """
    Solve dy/dt = f(t, y) using Forward Euler.

    Parameters:
        f : callable  - rate function f(t, y) returning a NumPy array
        t_span : tuple - (t_start, t_end)
        y0 : array-like - initial state vector
        h : float - time step

    Returns:
        t : 1-D array of time values
        y : 2-D array, shape (len(t), len(y0))
    """
    t = np.arange(t_span[0], t_span[1], h)
    y = np.zeros((len(t), len(y0)))
    y[0] = np.array(y0, dtype=float)

    for i in range(len(t) - 1):
        y[i + 1] = y[i] + h * f(t[i], y[i])

    return t, y
