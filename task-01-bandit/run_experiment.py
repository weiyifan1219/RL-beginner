#!/usr/bin/env python3
"""运行 Task 01 三策略主实验并写出可复现结果。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from src.agents import EpsilonGreedyAgent, ThompsonSamplingAgent, UCBAgent
from src.bandits import BernoulliBandit, GaussianBandit
from src.experiment import ExperimentResult, plot_results, run_experiment


TASK_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TASK_DIR / "configs/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=TASK_DIR / "outputs/default")
    parser.add_argument("--seed", type=int, help="覆盖配置中的 seed")
    parser.add_argument("--runs", type=int, help="覆盖配置中的 n_runs")
    parser.add_argument("--steps", type=int, help="覆盖配置中的 n_steps")
    parser.add_argument("--quick", action="store_true", help="使用 20 runs × 200 steps 快速验证")
    parser.add_argument("--no-plot", action="store_true", help="不生成 PNG，适合无图形 smoke test")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    return config


def build_environment(config: dict[str, Any]) -> tuple[Callable[[int], Any], int]:
    kind = config.get("kind")
    if kind == "bernoulli":
        probabilities = np.asarray(config.get("probabilities"), dtype=np.float64)
        # 先实例化一次，统一复用环境类中的参数校验。
        validated = BernoulliBandit(probabilities)
        return lambda seed: BernoulliBandit(probabilities, seed=seed), validated.n_arms
    if kind == "gaussian":
        means = np.asarray(config.get("means"), dtype=np.float64)
        std = float(config.get("std", 1.0))
        validated = GaussianBandit(means, std=std)
        return lambda seed: GaussianBandit(means, std=std, seed=seed), validated.n_arms
    raise ValueError("environment.kind must be 'bernoulli' or 'gaussian'")


def build_agent_factory(config: dict[str, Any], n_arms: int, env_kind: str):
    name = config.get("name")
    if name == "epsilon-greedy":
        epsilon = float(config.get("epsilon", 0.1))
        step_size = config.get("step_size")
        return lambda seed: EpsilonGreedyAgent(
            n_arms, epsilon=epsilon, step_size=step_size, seed=seed
        )
    if name == "ucb":
        c = float(config.get("c", 2.0))
        step_size = config.get("step_size")
        return lambda seed: UCBAgent(n_arms, c=c, step_size=step_size, seed=seed)
    if name == "thompson-sampling":
        if env_kind != "bernoulli":
            raise ValueError("Beta-Bernoulli Thompson Sampling only supports a Bernoulli environment")
        alpha = float(config.get("alpha_prior", 1.0))
        beta = float(config.get("beta_prior", 1.0))
        return lambda seed: ThompsonSamplingAgent(
            n_arms, alpha_prior=alpha, beta_prior=beta, seed=seed
        )
    raise ValueError(f"unknown agent name: {name!r}")


def safe_key(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def save_curves(results: dict[str, ExperimentResult], path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for name, result in results.items():
        prefix = safe_key(name)
        arrays[f"{prefix}_mean_reward"] = result.mean_reward
        arrays[f"{prefix}_reward_standard_error"] = result.reward_standard_error
        arrays[f"{prefix}_optimal_action_rate"] = result.optimal_action_rate
        arrays[f"{prefix}_mean_instantaneous_regret"] = result.mean_instantaneous_regret
        arrays[f"{prefix}_mean_cumulative_regret"] = result.mean_cumulative_regret
    np.savez_compressed(path, **arrays)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    experiment_config = config.setdefault("experiment", {})
    n_runs = int(args.runs if args.runs is not None else experiment_config.get("n_runs", 200))
    n_steps = int(args.steps if args.steps is not None else experiment_config.get("n_steps", 1000))
    seed = int(args.seed if args.seed is not None else experiment_config.get("seed", 42))
    if args.quick:
        n_runs = min(n_runs, 20)
        n_steps = min(n_steps, 200)

    environment_config = config.get("environment", {})
    env_factory, n_arms = build_environment(environment_config)
    agent_configs = config.get("agents")
    if not isinstance(agent_configs, list) or not agent_configs:
        raise ValueError("agents must be a non-empty list")

    results: dict[str, ExperimentResult] = {}
    for agent_config in agent_configs:
        if not isinstance(agent_config, dict):
            raise ValueError("each agent config must be a mapping")
        name = str(agent_config.get("name"))
        results[name] = run_experiment(
            env_factory,
            build_agent_factory(agent_config, n_arms, str(environment_config.get("kind"))),
            n_runs=n_runs,
            n_steps=n_steps,
            seed=seed,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        **config,
        "experiment": {**experiment_config, "n_runs": n_runs, "n_steps": n_steps, "seed": seed},
    }
    (args.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "environment": environment_config,
        "seed": seed,
        "n_runs": n_runs,
        "n_steps": n_steps,
        "agents": {name: result.summary() for name, result in results.items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_curves(results, args.output_dir / "curves.npz")
    if not args.no_plot:
        plot_results(results, args.output_dir / "learning_curves.png")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果已保存到 {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
