"""teams/template/agent.py — your agent.  Replace the body of `act`; keep the interface.

Budget: 50 ms per act() on the reference machine (a physics.simulate call costs about 8 us, a 23-128-128-9 MLP forward
pass about 25 us, so there is room for a few hundred look-aheads or a small search).  Over 50 ms, or an exception, is
STAY and a strike; eleven strikes forfeit the match.  Load weights in `load_shared`, not in `act`.
"""
from __future__ import annotations
import os
import numpy as np
from grid2v2.agents import Agent, ScriptedAgent
from grid2v2.features import flatten
from grid2v2.physics import simulate

HERE = os.path.dirname(os.path.abspath(__file__))


class MyAgent(Agent):
    """Starts life as the Tier 1 scripted baseline so the template plays out of the box.  Make it yours."""

    @staticmethod
    def load_shared():
        """Anything both robots share: a model, a Q-table, a lookup.  Runs once per match, inside the setup budget.
        Example:  return torch.load(os.path.join(HERE, "weights.pt"), map_location="cpu")"""
        path = os.path.join(HERE, "weights.npz")
        return dict(np.load(path)) if os.path.exists(path) else None

    def __init__(self, adapter, rink, index=0, shared=None, **kw):
        super().__init__(adapter, rink, index, **kw)
        self.shared = shared
        self.fallback = ScriptedAgent(adapter, rink, index)

    def reset(self):
        """Called at the start of every match.  Clear any per-match state here."""
        pass

    def features(self, obs) -> np.ndarray:
        """Your state representation.  `flatten` is the full 23-float baseline; add, drop or derive."""
        return flatten(obs, self.rink.width, self.rink.height)

    def act(self, obs):
        """obs -> an action in self.adapter.space.  In your own frame you always attack towards +x.
        Helpers: self.adapter.encode_move(move), self.adapter.encode_kick(angle, speed), simulate(...)."""
        # TODO: replace with your policy, e.g.
        #   q = self.shared["W2"] @ np.maximum(self.shared["W1"] @ self.features(obs), 0); return int(np.argmax(q))
        return self.fallback.act(obs)
