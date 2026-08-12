"""Task 01 一键自检：单元契约、学习行为与 CLI 输出。"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]
ROOT = TASK_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK_DIR))

from _eval_harness import check, run_checks  # noqa: E402
from src.agents import ThompsonSamplingAgent  # noqa: E402
from src.bandits import BernoulliBandit  # noqa: E402
from src.experiment import run_experiment  # noqa: E402


def unit_tests() -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(TASK_DIR / "tests")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    detail = (completed.stdout + completed.stderr).strip().splitlines()
    return check(completed.returncode == 0, "unit_and_contract_tests", detail[-1] if detail else "")


def learning_behavior() -> dict[str, object]:
    result = run_experiment(
        lambda seed: BernoulliBandit([0.1, 0.3, 0.9], seed=seed),
        lambda seed: ThompsonSamplingAgent(3, seed=seed),
        n_runs=40,
        n_steps=300,
        seed=123,
    )
    final_rate = float(result.optimal_action_rate[-50:].mean())
    final_regret = float(result.mean_cumulative_regret[-1])
    passed = final_rate > 0.9 and final_regret < 30.0
    return check(
        passed,
        "thompson_learning_behavior",
        f"last-50 optimal rate={final_rate:.3f}, final pseudo-regret={final_regret:.3f}",
    )


def quick_cli() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="rl-beginner-task01-") as directory:
        output = Path(directory)
        completed = subprocess.run(
            [
                sys.executable,
                str(TASK_DIR / "run_experiment.py"),
                "--quick",
                "--output-dir",
                str(output),
                "--no-plot",
            ],
            cwd=TASK_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        expected = ["resolved_config.yaml", "summary.json", "curves.npz"]
        missing = [name for name in expected if not (output / name).is_file()]
        passed = completed.returncode == 0 and not missing
        detail = "all files created" if passed else f"returncode={completed.returncode}, missing={missing}"
        return check(passed, "quick_experiment_cli", detail)


def main() -> int:
    ok = run_checks(
        [unit_tests, learning_behavior, quick_cli],
        output=TASK_DIR / "eval/result.json",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
