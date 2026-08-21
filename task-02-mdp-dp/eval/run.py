"""Task 02 自检：GridWorld、Bellman backup、PI/VI 一致性。"""
from __future__ import annotations
from pathlib import Path
import sys

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))
import json
from pathlib import Path
import numpy as np
from src.gridworld import GridWorld
from src.policy_evaluation import create_uniform_random_policy, policy_evaluation
from src.policy_iteration import policy_iteration
from src.value_iteration import value_iteration

ROOT = Path(__file__).resolve().parents[1]
def check(name, fn):
    try: fn(); print(f"[通过] {name}"); return {"name": name, "status": "passed"}
    except Exception as exc: print(f"[失败] {name}: {type(exc).__name__}: {exc}"); return {"name": name, "status": "failed", "error": str(exc)}
def main():
    results=[check("gridworld_transition_contract", _env), check("policy_evaluation", _evaluation), check("policy_iteration_value_iteration", _control)]
    (ROOT/"eval").mkdir(exist_ok=True); (ROOT/"eval/result.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if all(x["status"]=="passed" for x in results) else 1
def _env():
    env=GridWorld(); assert env.reset()==env.start_state; next_state,reward,done=env.step(GridWorld.RIGHT); assert next_state != env.start_state and isinstance(reward,(int,float)) and not done
    assert env.get_next_state((0,0), GridWorld.UP)==(0,0)
def _evaluation():
    env=GridWorld(); policy=create_uniform_random_policy(env); values=policy_evaluation(env,policy,gamma=.9,theta=1e-6); assert set(values)==set(env.get_states()); assert np.all(np.isfinite(list(values.values())))
def _control():
    env=GridWorld(); policy_pi, values_pi=policy_iteration(env,gamma=.9,theta=1e-6); values_vi, policy_vi=value_iteration(env,gamma=.9,theta=1e-6); assert set(values_pi)==set(values_vi); assert max(abs(values_pi[s]-values_vi[s]) for s in values_pi) < 1e-4
if __name__=="__main__": raise SystemExit(main())
