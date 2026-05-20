#!/usr/bin/env python3
"""CLI entry point for Lorenz chaos simulation."""

import argparse
from pathlib import Path

from lorenz.plots import generate_demo_gif, run_full_analysis, run_quick_demo


def print_metrics(results):
    butterfly = results.get("butterfly")
    if butterfly and butterfly["lyapunov_exp"] is not None:
        print(f"Estimated Lyapunov exponent: lambda = {butterfly['lyapunov_exp']:.3f}")
        print("Published value: lambda ~ 0.9")

    solver = results.get("solver")
    if solver:
        print(f"\nSolver Comparison (dt = {solver['h']}):")
        if solver["lambda_euler"] is not None:
            print(f"  Forward Euler  Lyapunov exponent: {solver['lambda_euler']:.3f}")
        if solver["lambda_rk4"] is not None:
            print(f"  RK4            Lyapunov exponent: {solver['lambda_rk4']:.3f}")
        print("  Published value:                  ~0.905")

    fractal = results.get("fractal")
    if fractal and fractal["d2"] is not None:
        print(f"\nFractal dimension: using {fractal['n_pts']} points on the attractor")
        print(f"  Estimated correlation dimension: D2 = {fractal['d2']:.3f}")
        print("  Published value:                 D  ~ 2.06")

    ts_results = results.get("timestep")
    if ts_results:
        print("\nTime-step sensitivity:")
        for result in ts_results:
            status = "STABLE" if result["stable"] else "UNSTABLE"
            print(f"  dt = {result['dt']:6.3f} | max|state| = {result['max_val']:.2e} | {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate the Lorenz attractor with Euler and RK4 solvers."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Generate all 9 figure groups (~5-10 min)",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Generate rotating 3D demo GIF only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory for generated figures (default: output/)",
    )
    parser.add_argument(
        "--gif-path",
        type=Path,
        default=Path("assets/demo.gif"),
        help="Path for demo GIF (default: assets/demo.gif)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show plots interactively (default: headless save only)",
    )
    args = parser.parse_args()

    if args.gif:
        path = generate_demo_gif(args.gif_path)
        print(f"Demo GIF saved to {path}")
        return

    if args.full:
        print("Running full analysis (this may take several minutes)...")
        paths, results = run_full_analysis(args.output, display=args.display)
        print_metrics(results)
    else:
        print("Running quick demo...")
        paths, butterfly = run_quick_demo(args.output, display=args.display)
        results = {"butterfly": butterfly}
        print_metrics(results)

    print(f"\nSaved {len(paths)} figure(s) to {args.output.resolve()}/")
    for path in paths:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
