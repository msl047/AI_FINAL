"""Exported PPO policy: numpy forward pass of the SB3 MlpPolicy (shared weights, one-hot robot index)."""
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
