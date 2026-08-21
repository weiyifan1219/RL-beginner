"""Task 04 自检：Q network、replay buffer 与 DQN TD update。"""
from __future__ import annotations
from pathlib import Path
import sys

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))
import json
from pathlib import Path
import numpy as np
import torch
from src.q_network import QNetwork
from src.replay_buffer import ReplayBuffer
from src.dqn_agent import DQNAgent
ROOT=Path(__file__).resolve().parents[1]
def check(name,fn):
    try: fn(); print(f"[通过] {name}"); return {"name":name,"status":"passed"}
    except Exception as e: print(f"[失败] {name}: {type(e).__name__}: {e}"); return {"name":name,"status":"failed","error":str(e)}
def main():
    results=[check("q_network_shape",_network),check("replay_buffer_sampling",_buffer),check("dqn_td_update",_agent)]
    (ROOT/"eval").mkdir(exist_ok=True); (ROOT/"eval/result.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if all(x["status"]=="passed" for x in results) else 1
def _network(): assert QNetwork(4,2)(torch.zeros(3,4)).shape==(3,2)
def _buffer():
    b=ReplayBuffer(8); b.push(np.zeros(4),0,1.,np.ones(4),False); assert len(b)==1; sample=b.sample(1); assert len(sample)==5
def _agent():
    a=DQNAgent(4,2,batch_size=4,buffer_capacity=8,target_update_freq=1)
    for _ in range(4): a.store_transition(np.zeros(4),0,1.,np.ones(4),False)
    loss=a.update(); assert loss is not None and np.isfinite(loss)
if __name__=="__main__": raise SystemExit(main())
