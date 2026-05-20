import numpy as np

from lorenz.system import DEFAULT_IC, DEFAULT_H, DEFAULT_T_SPAN, lorenz, make_lorenz
from solvers.euler import forward_euler
from solvers.rk4 import rk4


def fit_lyapunov(t_arr, separation, t_min=0.5, sep_min=1e-4, sep_max=0.1):
    """Fit Lyapunov exponent from trajectory separation during exponential growth."""
    log_sep = np.log(separation + 1e-300)
    mask = (t_arr >= t_min) & (separation > sep_min) & (separation < sep_max)
    if np.sum(mask) > 10:
        coeffs = np.polyfit(t_arr[mask], log_sep[mask], 1)
        return coeffs[0], coeffs
    return None, None


def fit_lyapunov_butterfly(t_arr, separation):
    """Fit Lyapunov exponent using the butterfly-effect window (t >= 1, sep < 1)."""
    log_sep = np.log(separation)
    fit_mask = (t_arr >= 1.0) & (separation > 0) & (separation < 1.0)
    if np.sum(fit_mask) > 10:
        t_fit = t_arr[fit_mask]
        log_sep_fit = log_sep[fit_mask]
        coeffs = np.polyfit(t_fit, log_sep_fit, 1)
        lyapunov_exp = coeffs[0]
        fit_line = np.exp(coeffs[1]) * np.exp(lyapunov_exp * t_arr)
        return lyapunov_exp, fit_line
    return None, None


def butterfly_effect_data(delta=1e-5, t_span=DEFAULT_T_SPAN, h=DEFAULT_H):
    """Run paired simulations for butterfly-effect analysis."""
    ic = DEFAULT_IC
    perturbed = [ic[0] + delta, ic[1], ic[2]]
    t1, traj1 = forward_euler(lorenz, t_span, ic, h)
    t2, traj2 = forward_euler(lorenz, t_span, perturbed, h)
    separation = np.sqrt(np.sum((traj1 - traj2) ** 2, axis=1))
    lyapunov_exp, fit_line = fit_lyapunov_butterfly(t1, separation)
    return {
        "t1": t1,
        "t2": t2,
        "traj1": traj1,
        "traj2": traj2,
        "separation": separation,
        "lyapunov_exp": lyapunov_exp,
        "fit_line": fit_line,
        "delta": delta,
    }


def timestep_sensitivity_data(dt_values=None, t_span=(0, 40)):
    """Run Forward Euler at multiple time steps."""
    if dt_values is None:
        dt_values = [0.1, 0.02, 0.01, 0.002]

    results = []
    for dt in dt_values:
        t_dt, traj_dt = forward_euler(lorenz, t_span, DEFAULT_IC, dt)
        max_val = np.max(np.abs(traj_dt))
        stable = np.all(np.isfinite(traj_dt)) and max_val < 1e6
        results.append(
            {
                "dt": dt,
                "t": t_dt,
                "trajectory": traj_dt,
                "max_val": max_val,
                "stable": stable,
            }
        )
    return results


def solver_comparison_data(h=DEFAULT_H, t_span=DEFAULT_T_SPAN, delta=1e-5):
    """Compare Forward Euler and RK4 trajectories and Lyapunov divergence."""
    ic = DEFAULT_IC
    ic_perturbed = [ic[0] + delta, ic[1], ic[2]]

    t_euler, traj_euler = forward_euler(lorenz, t_span, ic, h)
    t_rk4, traj_rk4 = rk4(lorenz, t_span, ic, h)
    _, traj_euler_p = forward_euler(lorenz, t_span, ic_perturbed, h)
    _, traj_rk4_p = rk4(lorenz, t_span, ic_perturbed, h)

    sep_euler = np.sqrt(np.sum((traj_euler - traj_euler_p) ** 2, axis=1))
    sep_rk4 = np.sqrt(np.sum((traj_rk4 - traj_rk4_p) ** 2, axis=1))
    lambda_euler, _ = fit_lyapunov(t_euler, sep_euler)
    lambda_rk4, _ = fit_lyapunov(t_rk4, sep_rk4)
    solver_diff = np.sqrt(np.sum((traj_euler - traj_rk4) ** 2, axis=1))

    return {
        "h": h,
        "t_euler": t_euler,
        "traj_euler": traj_euler,
        "t_rk4": t_rk4,
        "traj_rk4": traj_rk4,
        "sep_euler": sep_euler,
        "sep_rk4": sep_rk4,
        "lambda_euler": lambda_euler,
        "lambda_rk4": lambda_rk4,
        "solver_diff": solver_diff,
    }


def bifurcation_data(rho_values=None, t_span=(0, 100), h=0.01, transient=50):
    """Sweep rho and collect z local maxima for bifurcation diagram."""
    if rho_values is None:
        rho_values = np.arange(0.5, 30.5, 0.5)

    rho_plot = []
    z_extrema_plot = []

    for rho_val in rho_values:
        lorenz_rho = make_lorenz(rho=rho_val)
        t_bif, traj_bif = rk4(lorenz_rho, t_span, DEFAULT_IC, h)
        mask = t_bif > transient
        z_data = traj_bif[mask, 2]

        for j in range(1, len(z_data) - 1):
            if z_data[j] > z_data[j - 1] and z_data[j] > z_data[j + 1]:
                rho_plot.append(rho_val)
                z_extrema_plot.append(z_data[j])

    return {"rho_plot": rho_plot, "z_extrema_plot": z_extrema_plot}


def non_chaotic_regimes_data():
    """Simulate stable equilibrium and periodic limit cycle regimes."""
    regimes = [
        (5.0, "Stable Equilibrium"),
        (22.0, "Periodic Limit Cycle"),
    ]
    results = []
    for rho_val, regime_name in regimes:
        lorenz_regime = make_lorenz(rho=rho_val)
        t_reg, traj_reg = rk4(lorenz_regime, (0, 80), [5.0, 5.0, 5.0], 0.01)
        results.append(
            {
                "rho": rho_val,
                "name": regime_name,
                "t": t_reg,
                "trajectory": traj_reg,
            }
        )
    return results


def fractal_dimension_data(t_span=(0, 200), h=0.01, transient=50, n_sample=2000, seed=42):
    """Estimate correlation dimension using Grassberger-Procaccia algorithm."""
    t_frac, traj_frac = rk4(lorenz, t_span, DEFAULT_IC, h)
    transient_mask = t_frac > transient
    points = traj_frac[transient_mask][::5]
    n_pts = len(points)

    rng = np.random.default_rng(seed)
    sample_size = min(n_sample, n_pts)
    idx = rng.choice(n_pts, size=sample_size, replace=False)
    sample = points[idx]

    dists = []
    for i in range(sample_size):
        diff = sample[i + 1 :] - sample[i]
        d = np.sqrt(np.sum(diff**2, axis=1))
        dists.append(d)
    dists = np.concatenate(dists)
    n_pairs = len(dists)

    eps_values = np.logspace(-1, 2, 50)
    c_eps = np.array([np.sum(dists < eps) / n_pairs for eps in eps_values])

    valid = (c_eps > 0.005) & (c_eps < 0.5)
    log_eps = np.log(eps_values[valid])
    log_c = np.log(c_eps[valid])

    d2 = None
    fit_line_frac = None
    if len(log_eps) > 5:
        coeffs = np.polyfit(log_eps, log_c, 1)
        d2 = coeffs[0]
        fit_line_frac = np.polyval(coeffs, log_eps)

    return {
        "n_pts": n_pts,
        "eps_values": eps_values,
        "c_eps": c_eps,
        "log_eps": log_eps,
        "log_c": log_c,
        "d2": d2,
        "fit_line_frac": fit_line_frac,
    }
