"""RL-Beginner 的共享评测与环境检查工具。"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable


def check(condition: bool, name: str, detail: str = "") -> dict[str, Any]:
    """返回可序列化的单项检查结果，供后续各 task 的 eval/run.py 复用。"""
    result = {"name": name, "status": "passed" if condition else "failed"}
    if detail:
        result["detail"] = detail
    return result


def optional_import(module: str) -> tuple[bool, str]:
    try:
        imported = importlib.import_module(module)
        return True, str(getattr(imported, "__version__", "installed"))
    except Exception as exc:  # diagnostic path
        return False, f"{type(exc).__name__}: {exc}"


def run_checks(checks: list[Callable[[], dict[str, Any]]], output: Path | None = None) -> bool:
    """执行检查、打印结果，并可把结构化结果写入 result.json。"""
    results = [fn() for fn in checks]
    for item in results:
        icon = "通过" if item["status"] == "passed" else "失败"
        print(f"[{icon}] {item['name']}: {item.get('detail', '')}")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return all(item["status"] == "passed" for item in results)


def environment_checks() -> list[dict[str, Any]]:
    packages = {
        "numpy": "numpy", "torch": "torch", "gymnasium": "gymnasium",
        "matplotlib": "matplotlib", "tensorboard": "tensorboard", "yaml": "PyYAML",
    }
    results = [check(sys.version_info >= (3, 10), "python>=3.10", sys.version.split()[0])]
    for module, package in packages.items():
        ok, detail = optional_import(module)
        results.append(check(ok, package, detail))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="RL-Beginner shared evaluation harness")
    parser.add_argument("--check-env", action="store_true", help="检查基础 Python 包")
    args = parser.parse_args()
    if not args.check_env:
        parser.print_help()
        return 0
    results = environment_checks()
    for item in results:
        icon = "通过" if item["status"] == "passed" else "缺失"
        print(f"[{icon}] {item['name']}: {item.get('detail', '')}")
    return 0 if all(item["status"] == "passed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

