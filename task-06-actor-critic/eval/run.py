"""task-06-actor-critic 的占位自检入口；实现任务时逐步替换为数值契约。"""

from __future__ import annotations

import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
ROOT = TASK_DIR.parent
sys.path.insert(0, str(ROOT))

from _eval_harness import check, run_checks  # noqa: E402


def main() -> int:
    required = ["README.md", "src", "data", "figures", "notes"]
    ok = run_checks([
        lambda: check((TASK_DIR / item).exists(), f"scaffold:{item}")
        for item in required
    ])
    print("当前为课程骨架；完成该任务设计时，请在这里加入接口和数值自检。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
