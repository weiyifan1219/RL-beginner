"""Task 06 自检：Actor/Critic 输出、one-step 与 n-step target。"""
from __future__ import annotations
from pathlib import Path
import sys

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))
import json
from pathlib import Path
from src.actor_critic import ActorCriticAgent
from src.n_step_actor_critic import NStepActorCriticAgent
ROOT=Path(__file__).resolve().parents[1]
def check(name,fn):
    try: fn(); print(f"[通过] {name}"); return {"name":name,"status":"passed"}
    except Exception as e: print(f"[失败] {name}: {type(e).__name__}: {e}"); return {"name":name,"status":"failed","error":str(e)}
def main():
    results=[check("one_step_actor_critic",_one),check("n_step_actor_critic",_nstep),check("terminal_target",_terminal)]
    (ROOT/"eval").mkdir(exist_ok=True); (ROOT/"eval/result.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if all(x["status"]=="passed" for x in results) else 1
def _one():
    a=ActorCriticAgent(4,2); action=a.select_action([0,0,0,0]); result=a.update([0,0,0,0],action,1,[0,0,0,0],False); assert "td_error" in result
def _nstep():
    a=NStepActorCriticAgent(4,2); action=a.select_action([0,0,0,0]); result=a.update([0,0,0,0],action,[1.,1.],[0,0,0,0],False); assert "advantage" in result
def _terminal():
    a=ActorCriticAgent(4,2); action=a.select_action([0,0,0,0]); result=a.update([0,0,0,0],action,1,[0,0,0,0],True); assert abs(result["td_target"]-1.)<1e-6
if __name__=="__main__": raise SystemExit(main())
