"""Task 02 一键自检：测试、GridWorld golden values 与 CLI。"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


TASK_DIR = Path(__file__).resolve().parents[1]
ROOT = TASK_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK_DIR))

from _eval_harness import check, run_checks  # noqa: E402
from src.dynamic_programming import iterative_policy_evaluation, value_iteration  # noqa: E402
from src.gridworld import GridWorld  # noqa: E402


def unit_tests() -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(TASK_DIR / "tests")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    lines = (completed.stdout + completed.stderr).strip().splitlines()
    return check(completed.returncode == 0, "unit_and_contract_tests", lines[-1] if lines else "")


def golden_values() -> dict[str, object]:
    env = GridWorld()
    random_result = iterative_policy_evaluation(
        env, env.uniform_random_policy(), gamma=1.0, theta=1e-10, max_iterations=20_000
    )
    optimal_result = value_iteration(env, gamma=1.0, theta=1e-10, max_iterations=10_000)
    passed = (
        random_result.converged
        and optimal_result.converged
        and np.isclose(random_result.values[5], -18.0, atol=1e-6)
        and np.isclose(optimal_result.values[5], -2.0, atol=1e-8)
    )
    return check(
        passed,
        "gridworld_golden_values",
        f"V_random(5)={random_result.values[5]:.6f}, V_optimal(5)={optimal_result.values[5]:.6f}",
    )


def quick_cli() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="rl-beginner-task02-") as directory:
        output = Path(directory)
        completed = subprocess.run(
            [
                sys.executable,
                str(TASK_DIR / "run_experiment.py"),
                "--quick",
                "--no-plot",
                "--output-dir",
                str(output),
            ],
            cwd=TASK_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        expected = ["resolved_config.yaml", "summary.json", "values.npy", "policy.npy"]
        missing = [name for name in expected if not (output / name).is_file()]
        passed = completed.returncode == 0 and not missing
        detail = "all files created" if passed else f"returncode={completed.returncode}, missing={missing}"
        return check(passed, "quick_experiment_cli", detail)


def main() -> int:
    ok = run_checks(
        [unit_tests, golden_values, quick_cli], output=TASK_DIR / "eval/result.json"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
