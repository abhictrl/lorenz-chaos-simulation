from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lorenz.system import DEFAULT_H, DEFAULT_IC, DEFAULT_T_SPAN, lorenz
from solvers.euler import forward_euler
from solvers.rk4 import rk4


def _save_figure(fig, output_dir, filename, display=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if display:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_time_series(t, trajectory, output_dir, display=False):
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, x, linewidth=0.5, color="tab:blue")
    axes[0].set_ylabel("x(t)")
    axes[0].set_title("Lorenz Attractor - Time Series")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, y, linewidth=0.5, color="tab:orange")
    axes[1].set_ylabel("y(t)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, z, linewidth=0.5, color="tab:green")
    axes[2].set_ylabel("z(t)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return _save_figure(fig, output_dir, "lorenz_time_series.png", display)


def plot_3d_trajectory(t, trajectory, output_dir, display=False):
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=0.4, color="tab:purple")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Lorenz Attractor - 3D Trajectory (The Butterfly)")

    plt.tight_layout()
    return _save_figure(fig, output_dir, "lorenz_3d.png", display)


def plot_2d_projections(t, trajectory, output_dir, display=False):
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(x, y, linewidth=0.3, color="tab:blue")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_title("X-Y Projection")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, z, linewidth=0.3, color="tab:orange")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")
    axes[1].set_title("X-Z Projection")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(y, z, linewidth=0.3, color="tab:green")
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")
    axes[2].set_title("Y-Z Projection")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Lorenz Attractor - 2D Projections", fontsize=14)
    plt.tight_layout()
    return _save_figure(fig, output_dir, "lorenz_2d_projections.png", display)


def plot_butterfly_effect(data, output_dir, display=False):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(data["t1"], data["traj1"][:, 0], linewidth=0.6, color="tab:blue", label="Original")
    axes[0].plot(
        data["t2"],
        data["traj2"][:, 0],
        linewidth=0.6,
        color="tab:red",
        alpha=0.7,
        label=r"Perturbed ($\delta = 10^{-5}$)",
    )
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("x(t)")
    axes[0].set_title("Butterfly Effect: Two Trajectories with Tiny Perturbation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(data["t1"], data["separation"], linewidth=0.8, color="tab:purple", label=r"$|\Delta(t)|$")
    lyapunov_exp = data["lyapunov_exp"]
    if lyapunov_exp is not None:
        axes[1].semilogy(
            data["t1"],
            data["fit_line"],
            "--",
            linewidth=1.5,
            color="tab:orange",
            label=rf"Fit: $\lambda \approx {lyapunov_exp:.2f}$ (published $\approx 0.9$)",
        )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$|\Delta(t)|$ (log scale)")
    axes[1].set_title("Divergence of Nearby Trajectories (Lyapunov Exponent)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=1e-6)

    plt.tight_layout()
    return _save_figure(fig, output_dir, "lorenz_butterfly_effect.png", display)


def plot_timestep_sensitivity(results, output_dir, display=False):
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 10), sharex=True)

    for i, result in enumerate(results):
        dt = result["dt"]
        x_dt = result["trajectory"][:, 0]
        stable = result["stable"]
        color = "tab:blue" if stable else "tab:red"
        label = rf"$\Delta t = {dt}$"
        if not stable:
            label += " (UNSTABLE)"

        axes[i].plot(result["t"], x_dt, linewidth=0.5, color=color)
        axes[i].set_ylabel("x(t)")
        axes[i].set_title(label, fontsize=11)
        axes[i].grid(True, alpha=0.3)
        if not stable:
            axes[i].set_ylim(-50, 50)

    axes[-1].set_xlabel("Time")
    plt.suptitle("Time-Step Sensitivity: Forward Euler on the Lorenz System", fontsize=13)
    plt.tight_layout()
    return _save_figure(fig, output_dir, "lorenz_timestep_sensitivity.png", display)


def plot_solver_comparison(data, output_dir, display=False):
    h = data["h"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(data["t_euler"], data["traj_euler"][:, 0], linewidth=0.5, color="tab:blue", label="Forward Euler")
    axes[0].plot(data["t_rk4"], data["traj_rk4"][:, 0], linewidth=0.5, color="tab:red", alpha=0.7, label="RK4")
    axes[0].set_ylabel("x(t)")
    axes[0].set_title(rf"Solver Comparison: x(t) with $\Delta t = {h}$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(data["t_euler"], data["solver_diff"], linewidth=0.8, color="tab:purple")
    axes[1].set_ylabel(r"$||Euler - RK4||$")
    axes[1].set_title("Accumulated Difference Between Solvers")
    axes[1].grid(True, alpha=0.3)

    lambda_euler = data["lambda_euler"]
    lambda_rk4 = data["lambda_rk4"]
    axes[2].semilogy(
        data["t_euler"],
        data["sep_euler"],
        linewidth=0.8,
        color="tab:blue",
        label=rf"Euler ($\lambda \approx {lambda_euler:.2f}$)" if lambda_euler else "Euler",
    )
    axes[2].semilogy(
        data["t_rk4"],
        data["sep_rk4"],
        linewidth=0.8,
        color="tab:red",
        label=rf"RK4 ($\lambda \approx {lambda_rk4:.2f}$)" if lambda_rk4 else "RK4",
    )
    axes[2].axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Saturation ~ O(1)")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel(r"$|\Delta(t)|$ (log scale)")
    axes[2].set_title("Lyapunov Divergence: Euler vs RK4")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(bottom=1e-7)

    plt.tight_layout()
    return _save_figure(fig, output_dir, "lorenz_solver_comparison.png", display)


def plot_bifurcation(data, output_dir, display=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["rho_plot"], data["z_extrema_plot"], ",", color="black", markersize=0.5)
    ax.set_xlabel(r"$\rho$ (Rayleigh number)")
    ax.set_ylabel(r"$z$ local maxima")
    ax.set_title("Bifurcation Diagram: Lorenz System")
    ax.grid(True, alpha=0.3)
    ax.axvline(x=24.74, color="tab:red", linestyle="--", alpha=0.7, label=r"Chaos onset $\rho \approx 24.74$")
    ax.legend()

    plt.tight_layout()
    return _save_figure(fig, output_dir, "lorenz_bifurcation.png", display)


def plot_non_chaotic_regimes(regimes, output_dir, display=False):
    fig = plt.figure(figsize=(14, 10))

    for idx, regime in enumerate(regimes):
        traj_reg = regime["trajectory"]
        x_reg, y_reg, z_reg = traj_reg[:, 0], traj_reg[:, 1], traj_reg[:, 2]
        rho_val = regime["rho"]
        regime_name = regime["name"]

        ax3d = fig.add_subplot(2, 2, 2 * idx + 1, projection="3d")
        ax3d.plot(x_reg, y_reg, z_reg, linewidth=0.5, color="tab:purple")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.set_title(rf"$\rho = {rho_val}$: {regime_name}" + "\n(Phase Portrait)")

        ax_ts = fig.add_subplot(2, 2, 2 * idx + 2)
        ax_ts.plot(regime["t"], x_reg, linewidth=0.5, color="tab:blue")
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("x(t)")
        ax_ts.set_title(rf"$\rho = {rho_val}$: {regime_name}" + "\n(Time Series)")
        ax_ts.grid(True, alpha=0.3)

    plt.suptitle("Non-Chaotic Regimes of the Lorenz System", fontsize=14)
    plt.tight_layout()
    return _save_figure(fig, output_dir, "lorenz_non_chaotic.png", display)


def plot_fractal_dimension(data, output_dir, display=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        np.log(data["eps_values"]),
        np.log(data["c_eps"] + 1e-300),
        "o-",
        markersize=3,
        color="tab:blue",
        label=r"$\log C(\varepsilon)$",
    )
    d2 = data["d2"]
    if d2 is not None:
        ax.plot(
            data["log_eps"],
            data["fit_line_frac"],
            "--",
            linewidth=2,
            color="tab:red",
            label=rf"Linear fit: $D_2 \approx {d2:.2f}$ (published $\approx 2.06$)",
        )
    ax.set_xlabel(r"$\log(\varepsilon)$")
    ax.set_ylabel(r"$\log(C(\varepsilon))$")
    ax.set_title("Correlation Dimension of the Lorenz Attractor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return _save_figure(fig, output_dir, "lorenz_fractal_dimension.png", display)


def generate_demo_gif(output_path, t_span=DEFAULT_T_SPAN, h=DEFAULT_H, frames=72):
    """Generate a rotating 3D animation of the Lorenz attractor."""
    _, trajectory = rk4(lorenz, t_span, DEFAULT_IC, h)
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=0.3, color="tab:purple", alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Lorenz Attractor")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def update(frame):
        ax.view_init(elev=25, azim=frame * (360 / frames))
        return ax,

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    anim.save(output_path, writer="pillow", fps=15, dpi=80)
    plt.close(fig)
    return output_path


def run_quick_demo(output_dir, display=False):
    """Generate 3D trajectory and butterfly effect plots."""
    t, trajectory = forward_euler(lorenz, DEFAULT_T_SPAN, DEFAULT_IC, DEFAULT_H)
    paths = [
        plot_3d_trajectory(t, trajectory, output_dir, display),
    ]

    from lorenz.analysis import butterfly_effect_data

    butterfly = butterfly_effect_data()
    paths.append(plot_butterfly_effect(butterfly, output_dir, display))
    return paths, butterfly


def run_full_analysis(output_dir, display=False):
    """Generate all nine figure groups."""
    t, trajectory = forward_euler(lorenz, DEFAULT_T_SPAN, DEFAULT_IC, DEFAULT_H)
    paths = [
        plot_time_series(t, trajectory, output_dir, display),
        plot_3d_trajectory(t, trajectory, output_dir, display),
        plot_2d_projections(t, trajectory, output_dir, display),
    ]

    from lorenz.analysis import (
        bifurcation_data,
        butterfly_effect_data,
        fractal_dimension_data,
        non_chaotic_regimes_data,
        solver_comparison_data,
        timestep_sensitivity_data,
    )

    butterfly = butterfly_effect_data()
    paths.append(plot_butterfly_effect(butterfly, output_dir, display))

    ts_results = timestep_sensitivity_data()
    paths.append(plot_timestep_sensitivity(ts_results, output_dir, display))

    solver = solver_comparison_data()
    paths.append(plot_solver_comparison(solver, output_dir, display))

    bifurcation = bifurcation_data()
    paths.append(plot_bifurcation(bifurcation, output_dir, display))

    regimes = non_chaotic_regimes_data()
    paths.append(plot_non_chaotic_regimes(regimes, output_dir, display))

    fractal = fractal_dimension_data()
    paths.append(plot_fractal_dimension(fractal, output_dir, display))

    return paths, {
        "butterfly": butterfly,
        "solver": solver,
        "fractal": fractal,
        "timestep": ts_results,
    }
