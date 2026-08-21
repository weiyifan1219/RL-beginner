"""Task 05 自检：returns、policy sampling、REINFORCE 与 baseline。"""
from __future__ import annotations
from pathlib import Path
import sys

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))
import json
from pathlib import Path
import numpy as np
import torch
from src.policy_network import PolicyNetwork
from src.reinforce import compute_returns, update_policy
from src.reinforce_baseline import ValueNetwork, compute_returns as baseline_returns
ROOT=Path(__file__).resolve().parents[1]
def check(name,fn):
    try: fn(); print(f"[通过] {name}"); return {"name":name,"status":"passed"}
    except Exception as e: print(f"[失败] {name}: {type(e).__name__}: {e}"); return {"name":name,"status":"failed","error":str(e)}
def main():
    results=[check("discounted_returns",_returns),check("policy_distribution",_policy),check("reinforce_update",_update),check("baseline_value_shape",_baseline)]
    (ROOT/"eval").mkdir(exist_ok=True); (ROOT/"eval/result.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if all(x["status"]=="passed" for x in results) else 1
def _returns(): assert torch.allclose(compute_returns([1.,1.],.9),torch.tensor([1.9,1.]))
def _policy():
    p=PolicyNetwork(4,2); probs=p(torch.zeros(4)); assert probs.shape==(2,) and torch.isclose(probs.sum(),torch.tensor(1.))
def _update():
    p=PolicyNetwork(4,2); opt=torch.optim.Adam(p.parameters(),lr=1e-3); logs=[p.sample_action(torch.zeros(4))[1] for _ in range(3)]; loss,ret=update_policy(opt,logs,[1.,0.,1.]); assert np.isfinite(loss) and len(ret)==3
def _baseline():
    v=ValueNetwork(4); assert v(torch.zeros(2,4)).shape==(2,); assert len(baseline_returns([1.,1.],.9))==2
if __name__=="__main__": raise SystemExit(main())
