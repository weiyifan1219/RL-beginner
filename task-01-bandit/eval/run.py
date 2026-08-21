"""Task 01 自检：环境、增量更新与四种探索策略。"""
from __future__ import annotations
from pathlib import Path
import sys

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))
import json
from pathlib import Path
import numpy as np
from src import TreasureHuntBandit, GreedyAgent, EpsilonGreedyAgent, UCBAgent, ThompsonSamplingAgent

ROOT = Path(__file__).resolve().parents[1]

def check(name, fn):
    try:
        fn(); print(f"[通过] {name}"); return {"name": name, "status": "passed"}
    except Exception as exc:
        print(f"[失败] {name}: {type(exc).__name__}: {exc}"); return {"name": name, "status": "failed", "error": str(exc)}

def main():
    results = []
    results.append(check("bernoulli_environment", lambda: _env()))
    results.append(check("sample_average_update", lambda: _update()))
    results.append(check("four_exploration_agents", lambda: _agents()))
    results.append(check("thompson_reward_contract", lambda: _thompson_contract()))
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval/result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item["status"] == "passed" for item in results) else 1

def _env():
    env = TreasureHuntBandit([0.2, 0.8], seed=0)
    assert env.n_actions == 2 and env.optimal_action == 1 and env.optimal_value == 0.8
    assert env.step(0) in (0.0, 1.0)

def _update():
    agent = GreedyAgent(1, seed=0)
    agent.update(0, 1.0); agent.update(0, 0.0)
    assert np.isclose(agent.Q[0], 0.5) and agent.N[0] == 2

def _agents():
    env = TreasureHuntBandit([0.2, 0.8], seed=1)
    factories = [lambda: GreedyAgent(2, seed=1), lambda: EpsilonGreedyAgent(2, epsilon=0.1, seed=1), lambda: UCBAgent(2, seed=1), lambda: ThompsonSamplingAgent(2, seed=1)]
    for factory in factories:
        agent = factory()
        for _ in range(20):
            action = agent.select_action(); agent.update(action, env.step(action))
        assert np.all(np.isfinite(agent.Q))

def _thompson_contract():
    agent = ThompsonSamplingAgent(2, seed=0)
    try: agent.update(0, 0.5)
    except ValueError: return
    raise AssertionError("Beta-Bernoulli Thompson Sampling accepted a non-binary reward")

if __name__ == "__main__": raise SystemExit(main())
