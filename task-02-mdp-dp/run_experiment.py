#!/usr/bin/env python3
"""求解 GridWorld 并保存价值、策略、配置和图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.dynamic_programming import policy_iteration, value_iteration
from src.gridworld import GridWorld
from src.visualization import policy_labels, plot_value_and_policy


TASK_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TASK_DIR / "configs/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=TASK_DIR / "outputs/default")
    parser.add_argument("--algorithm", choices=("value_iteration", "policy_iteration"))
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--theta", type=float)
    parser.add_argument("--quick", action="store_true", help="将收敛阈值放宽到最多 1e-6")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    return config


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    environment_config = config.get("environment", {})
    solver_config = config.get("solver", {})
    env = GridWorld(
        rows=int(environment_config.get("rows", 4)),
        cols=int(environment_config.get("cols", 4)),
    )
    algorithm = str(args.algorithm or solver_config.get("algorithm", "value_iteration"))
    gamma = float(args.gamma if args.gamma is not None else solver_config.get("gamma", 1.0))
    theta = float(args.theta if args.theta is not None else solver_config.get("theta", 1e-10))
    max_iterations = int(solver_config.get("max_iterations", 10_000))
    if args.quick:
        theta = max(theta, 1e-6)
        max_iterations = min(max_iterations, 1_000)
    solver = value_iteration if algorithm == "value_iteration" else policy_iteration
    result = solver(env, gamma=gamma, theta=theta, max_iterations=max_iterations)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "environment": {"rows": env.rows, "cols": env.cols},
        "solver": {
            "algorithm": algorithm,
            "gamma": gamma,
            "theta": theta,
            "max_iterations": max_iterations,
        },
    }
    (args.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    np.save(args.output_dir / "values.npy", result.values)
    np.save(args.output_dir / "policy.npy", result.policy)
    summary = {
        "algorithm": algorithm,
        "gamma": gamma,
        "theta": theta,
        "iterations": result.iterations,
        "converged": result.converged,
        "final_delta": result.delta,
        "value_grid": result.values.reshape(env.rows, env.cols).tolist(),
        "policy_labels": np.asarray(policy_labels(env, result.policy))
        .reshape(env.rows, env.cols)
        .tolist(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.no_plot:
        plot_value_and_policy(
            env, result.values, result.policy, args.output_dir / "value_and_policy.png"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果已保存到 {args.output_dir.resolve()}")
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
