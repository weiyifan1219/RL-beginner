"""Task 03 自检：MC/TD prediction 与 SARSA/Q-Learning control。"""
from __future__ import annotations
from pathlib import Path
import sys

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))
import json, random
from pathlib import Path
import numpy as np
from src.gridworld import GridWorld
from src.monte_carlo import first_visit_mc_prediction
from src.td_learning import td0_prediction
from src.sarsa import sarsa
from src.q_learning import q_learning
ROOT=Path(__file__).resolve().parents[1]
def check(name,fn):
    try: fn(); print(f"[通过] {name}"); return {"name":name,"status":"passed"}
    except Exception as e: print(f"[失败] {name}: {type(e).__name__}: {e}"); return {"name":name,"status":"failed","error":str(e)}
def main():
    results=[check("mc_prediction",_mc),check("td0_prediction",_td),check("sarsa_control",_sarsa),check("q_learning_control",_q)]
    (ROOT/"eval").mkdir(exist_ok=True); (ROOT/"eval/result.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if all(x["status"]=="passed" for x in results) else 1
def policy(state): return random.choice([0,1,2,3])
def _mc():
    v=first_visit_mc_prediction(GridWorld(),policy,num_episodes=5,gamma=.9); assert v and np.all(np.isfinite(list(v.values())))
def _td():
    v=td0_prediction(GridWorld(),policy,num_episodes=5,alpha=.1,gamma=.9); assert v and np.all(np.isfinite(list(v.values())))
def _sarsa():
    q, returns=sarsa(GridWorld(),num_episodes=5,alpha=.1,gamma=.9,epsilon=.1); assert q and len(returns)==5 and np.all(np.isfinite([value for row in q.values() for value in row.values()]))
def _q():
    q, returns=q_learning(GridWorld(),num_episodes=5,alpha=.1,gamma=.9,epsilon=.1); assert q and len(returns)==5 and np.all(np.isfinite([value for row in q.values() for value in row.values()]))
if __name__=="__main__": raise SystemExit(main())
