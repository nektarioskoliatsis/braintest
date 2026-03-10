#!/usr/bin/env python3
"""
Run the cortical microcircuit simulation on GPU (PyTorch).

Usage:
    python run_simulation.py
    python run_simulation.py --n_neurons 25000 --duration 2.0
    python run_simulation.py --no-plot
"""

import argparse
import time
import os
import sys
import yaml

from gpu_partitioned import PartitionedSimulator
from analysis import generate_figures


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated cortical microcircuit simulation"
    )
    parser.add_argument(
        "--n_neurons", type=int, default=10000, help="Total number of neurons"
    )
    parser.add_argument(
        "--duration", type=float, default=2.0, help="Simulation duration in seconds"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip generating plots"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results"),
        help="Output directory for results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["network"]["n_neurons"] = args.n_neurons
    config["simulation"]["duration"] = args.duration
    config["simulation"]["seed"] = args.seed

    print("=" * 60)
    print("Cortical Microcircuit Simulation (PyTorch GPU)")
    print("=" * 60)
    N_exc = int(args.n_neurons * 0.8)
    N_inh = args.n_neurons - N_exc
    print(f"  Neurons:    {args.n_neurons:,} ({N_exc:,} E / {N_inh:,} I)")
    print(f"  Duration:   {args.duration} s")
    print(f"  Seed:       {args.seed}")
    print("=" * 60)

    print("\nBuilding network...")
    t_build_start = time.perf_counter()
    sim = PartitionedSimulator(config, seed=args.seed)
    t_build = time.perf_counter() - t_build_start
    print(f"  Network built in {t_build:.2f}s")

    t_sim_start = time.perf_counter()
    results = sim.run(args.duration)
    t_sim = time.perf_counter() - t_sim_start

    if not args.no_plot:
        print("\nGenerating analysis figures...")
        generate_figures(results, args.duration, output_dir=args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    timing_path = os.path.join(args.output_dir, "timing.txt")
    with open(timing_path, "a") as f:
        f.write(
            f"gpu_torch,{args.n_neurons},{args.duration},"
            f"{t_build:.4f},{t_sim:.4f}\n"
        )
    print(f"\nTiming appended to {timing_path}")
    print("Done.")


if __name__ == "__main__":
    main()
