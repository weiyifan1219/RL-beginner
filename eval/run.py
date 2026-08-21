"""总入口：依次运行 task 1--7 各自的 eval/run.py。"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASKS={1:"task-01-bandit",2:"task-02-mdp-dp",3:"task-03-tabular-rl",4:"task-04-dqn",5:"task-05-policy-gradient",6:"task-06-actor-critic",7:"task-07-trpo_ppo"}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--task",type=int,choices=sorted(TASKS),action="append"); args=parser.parse_args()
    numbers=args.task or list(TASKS)
    failed=False
    for number in numbers:
        task=ROOT/TASKS[number]; script=task/"eval/run.py"
        print(f"\n=== Task {number}: {task.name} ===")
        result=subprocess.run([sys.executable,str(script)],cwd=task)
        failed |= result.returncode != 0
    return 1 if failed else 0

if __name__=="__main__": raise SystemExit(main())
