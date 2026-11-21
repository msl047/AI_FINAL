"""tools/export_team.py — turn an SB3 PPO model into a self-contained teams/<name>/ folder (numpy only, no torch).

    python tools/export_team.py runs/ippo_d/model.zip teams/ippo_d --tier discrete

The exported agent recomputes the MlpPolicy forward pass (Tanh MLP) in numpy and applies the same flat features
(grid2v2.features.flatten plus a one-hot robot index) that the policy was trained on.
"""
import argparse, os, sys
import numpy as np
from stable_baselines3 import PPO

AGENT_SRC = '''"""Exported PPO policy: numpy forward pass of the SB3 MlpPolicy (shared weights, one-hot robot index)."""
import os
import numpy as np
from grid2v2.agents import Agent
from grid2v2.features import flatten

HERE = os.path.dirname(os.path.abspath(__file__))


class PolicyAgent(Agent):
    @staticmethod
    def load_shared():
        return dict(np.load(os.path.join(HERE, "weights.npz")))

    def __init__(self, adapter, rink, index=0, shared=None, **kw):
        super().__init__(adapter, rink, index, **kw)
        self.w = shared; self.n_layers = int(self.w["n_layers"]); self.discrete = bool(self.w["discrete"])

    def features(self, obs):
        return np.concatenate([flatten(obs, self.rink.width, self.rink.height), np.eye(2, dtype=np.float32)[self.index]])

    def act(self, obs):
        h = self.features(obs)
        for k in range(self.n_layers):
            h = np.tanh(self.w[f"W{k}"] @ h + self.w[f"b{k}"])
        out = self.w["Wout"] @ h + self.w["bout"]
        if self.discrete:
            return int(np.argmax(out))
        return np.clip(out, -1.0, 1.0).astype(np.float32)
'''

INIT_SRC = '''"""{name} — exported PPO team ({tier} tier).  See agent.py."""
from .agent import PolicyAgent

TIER = "{tier}"


def make_team(team, agent_ids, env):
    shared = PolicyAgent.load_shared()
    return {{a: PolicyAgent(env.adapters[a], env.rink, index=i, shared=shared) for i, a in enumerate(agent_ids)}}
'''


def export(model_path, out_dir, tier):
    model = PPO.load(model_path, device="cpu")
    sd = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}
    layers = sorted({int(k.split(".")[2]) for k in sd if k.startswith("mlp_extractor.policy_net.") and k.endswith(".weight")})
    w = {}
    for i, idx in enumerate(layers):
        w[f"W{i}"] = sd[f"mlp_extractor.policy_net.{idx}.weight"]; w[f"b{i}"] = sd[f"mlp_extractor.policy_net.{idx}.bias"]
    w["Wout"] = sd["action_net.weight"]; w["bout"] = sd["action_net.bias"]
    w["n_layers"] = np.array(len(layers)); w["discrete"] = np.array(tier != "box")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "weights.npz"), **w)
    name = os.path.basename(os.path.normpath(out_dir))
    open(os.path.join(out_dir, "agent.py"), "w").write(AGENT_SRC)
    open(os.path.join(out_dir, "__init__.py"), "w").write(INIT_SRC.format(name=name, tier=tier))
    # check: numpy forward == torch forward on random inputs
    import torch
    x = np.random.default_rng(0).uniform(-1, 1, size=(5, model.observation_space.shape[0])).astype(np.float32)
    with torch.no_grad():
        feats = model.policy.mlp_extractor.forward_actor(torch.as_tensor(x))
        ref = model.policy.action_net(feats).numpy()
    h = x
    for k in range(len(layers)): h = np.tanh(h @ w[f"W{k}"].T + w[f"b{k}"])
    out = h @ w["Wout"].T + w["bout"]
    assert np.allclose(out, ref, atol=1e-4), "numpy forward pass does not match the torch policy"
    print(f"exported {model_path} -> {out_dir} ({len(layers)} hidden layers, {'discrete' if tier != 'box' else 'box'}); forward pass verified")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("model"); p.add_argument("out"); p.add_argument("--tier", default="discrete")
    a = p.parse_args(); export(a.model, a.out, a.tier)
