"""Task 07 自检：GAE、rollout buffer、PPO clip 与 TRPO 接口。"""
from __future__ import annotations
from pathlib import Path
import sys

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))
import json
from pathlib import Path
import torch
from src.advantage import compute_gae
from src.buffer import RolloutBuffer
from src.network import Actor,Critic
from src.ppo import PPOAgent
from src.trpo import TRPOAgent
ROOT=Path(__file__).resolve().parents[1]
def check(name,fn):
    try: fn(); print(f"[通过] {name}"); return {"name":name,"status":"passed"}
    except Exception as e: print(f"[失败] {name}: {type(e).__name__}: {e}"); return {"name":name,"status":"failed","error":str(e)}
def main():
    results=[check("gae_terminal_mask",_gae),check("rollout_buffer",_buffer),check("ppo_update_diagnostics",_ppo),check("trpo_constructs",_trpo)]
    (ROOT/"eval").mkdir(exist_ok=True); (ROOT/"eval/result.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if all(x["status"]=="passed" for x in results) else 1
def _gae():
    adv,ret=compute_gae(torch.ones(4),torch.zeros(4),torch.tensor([0.,0.,0.,1.]),0.0); assert adv.shape==(4,) and ret.shape==(4,)
def _buffer():
    b=RolloutBuffer(); b.add([0,0,0,0],0,1.,False,0.,-0.5); assert len(b)==1; b.clear(); assert len(b)==0
def _ppo():
    agent=PPOAgent(Actor(4,2),Critic(4),update_epochs=1,minibatch_size=2); adv,ret=compute_gae(torch.ones(4),torch.zeros(4),torch.tensor([0.,0.,0.,1.]),0.0); result=agent.update(torch.zeros(4,4),torch.zeros(4,dtype=torch.long),torch.zeros(4),adv,ret); assert "approx_kl" in result and "clip_fraction" in result
def _trpo():
    agent=TRPOAgent(Actor(4,2),Critic(4)); assert agent is not None
if __name__=="__main__": raise SystemExit(main())
