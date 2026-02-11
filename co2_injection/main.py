import argparse
import os

import jax
import numpy as np

from src.config import SimConfig
from src.solver import run_simulation
from src.optimization import run_optimization
from src.visualization import save_saturation_map, plot_performance_curves
from src.reporting import generate_config_report, generate_result_report, save_report

jax.config.update("jax_enable_x64", True)


def ensure_dirs():
    os.makedirs("data/outputs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("reports", exist_ok=True)


def load_baseline_sat():
    path = "data/outputs/baseline_history.npy"
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True).item()
        return data["saturation"][-1]
    return 0.0


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "optimize"], required=True)
    parser.add_argument("--strategy", default="constant")
    parser.add_argument("--freq", type=float, default=10.0)
    parser.add_argument("--amp", type=float, default=0.1)
    args = parser.parse_args()

    cfg = SimConfig()

    config_report = generate_config_report(cfg)
    print(config_report)

    if args.mode == "run":
        print(f"--- Running Simulation: {args.strategy.upper()} ---")
        params = {"amp": args.amp, "freq": args.freq}

        final_state, history = run_simulation(cfg, args.strategy, params)
        sat_hist, p_hist = history
        final_sat = sat_hist[-1]

        if args.strategy == "constant":
            np.save("data/outputs/baseline_history.npy", {"saturation": sat_hist, "pressure": p_hist})
            print(">> Baseline data saved for future comparisons.")

        save_saturation_map(final_state, f"plots/map_{args.strategy}.png")
        plot_performance_curves((sat_hist, p_hist), args.strategy, f"plots/perf_{args.strategy}.png")

        baseline_sat = 0.0
        if os.path.exists("data/outputs/baseline_history.npy"):
            base_data = np.load("data/outputs/baseline_history.npy", allow_pickle=True).item()
            baseline_sat = base_data["saturation"][-1]

        result_report = generate_result_report(args.strategy, final_sat, baseline_sat, params)
        print(result_report)

        full_log = config_report + "\n" + result_report
        save_report(full_log, f"reports/log_{args.strategy}.txt")

    elif args.mode == "optimize":
        print(f"--- Starting Optimization for {args.strategy.upper()} ---")
        best_params, best_loss = run_optimization(cfg, args.strategy)

        clean_params = {}
        for key, value in best_params.items():
            if hasattr(value, "ndim") and value.ndim == 0:
                clean_params[key] = float(value)
            else:
                clean_params[key] = np.array(value)

        print("\n>> Optimization Complete. Winner found.")

        final_params = clean_params.copy()
        if "amp" not in final_params and args.strategy != "freeform":
            final_params["amp"] = args.amp

        print(f"Running Verification with Full Params: {final_params}")
        final_state, history = run_simulation(cfg, args.strategy, final_params)
        sat_hist, p_hist = history
        final_sat = sat_hist[-1]

        save_saturation_map(final_state, f"plots/map_{args.strategy}_opt.png")
        plot_performance_curves(
            (sat_hist, p_hist), args.strategy, f"plots/perf_{args.strategy}_opt.png"
        )

        baseline_sat = load_baseline_sat()
        result_report = generate_result_report(args.strategy, final_sat, baseline_sat, final_params)
        print(result_report)

        full_log = config_report + "\n" + result_report
        save_report(full_log, f"reports/log_{args.strategy}_opt.txt")


if __name__ == "__main__":
    main()
