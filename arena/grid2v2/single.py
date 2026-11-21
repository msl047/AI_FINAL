"""grid2v2/single.py — single-agent views of the rink, for HW4.

ShootingDrill  (Phase 1, tabular)   One attacker with the ball against a goalkeeper that slides along its goal line
                                    towards the ball's row with probability `keeper_skill` each step.
                                    Score: +1.  Keeper takes the ball: -1.  Ball comes to rest loose, or 60 steps: 0.
                                    `tabular_state(obs)` indexes (attacker x, attacker y, keeper row): 693 states x 9 actions.
SoloEnv        (Phase 2, networks)  1v1 on the full rink against a team module (default baselines.scripted), you are red_0.

Both follow the gymnasium API (reset -> obs, info; step -> obs, reward, terminated, truncated, info) and use the same
observation dict as the project, so a feature function written here carries over to the 2v2 game.
"""
from __future__ import annotations
import math
import numpy as np
import gymnasium as gym
from .env import Grid2v2
from .physics import Ball, predict
from .actions import STAY, N, S, E, W
from .match import load_team

DRILL_STEPS = 60


class ShootingDrill(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, tier="discrete", p_move=0.8, p_dribble=0.5, crease=2, max_steps=DRILL_STEPS, keeper_skill=0.4):
        self.env = Grid2v2(players_per_team=1, tiers=tier, p_move=p_move, p_dribble=p_dribble, crease=crease, max_steps=10**9)
        self.max_steps = max_steps
        self.keeper_skill = keeper_skill       # the keeper reacts (moves towards the ball's row) with this probability per step
        self.observation_space = self.env.observation_space("red_0")
        self.action_space = self.env.action_space("red_0")
        self.rink = self.env.rink
        self.rows = list(self.rink.goal_rows)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset(seed=None if seed is None else int(seed))
        rng = self.env.rng
        W = self.rink.width
        x = int(rng.integers(W // 2, W - 1 - self.env.crease + 1)); y = int(rng.integers(0, self.rink.height))
        self.env.pos["red_0"] = (x, y)
        self.env.pos["blue_0"] = (W - 1, int(rng.choice(self.rows)))
        self.env.holder = "red_0"; self.env.ball = Ball.at_cell(x, y)
        self.t = 0
        return self.env.observe("red_0"), {}

    def _keeper_action(self):
        """Blue's action in blue's frame: slide along the line towards where the ball is (or will be)."""
        e = self.env; b = e.ball; kx, ky = e.pos["blue_0"]
        y = b.y
        if b.moving and b.dx > 0:
            _, _, _, pts = predict(self.rink, b.x, b.y, b.dx * b.speed, b.dy * b.speed, {})
            y = max(pts, key=lambda p: p[0])[1]
        target = int(min(max(round(y), self.rows[0]), self.rows[-1]))
        if target == ky or self.env.rng.random() > self.keeper_skill: return STAY
        return N if target > ky else S                      # y is not mirrored, so N/S mean the same for blue

    def step(self, action):
        e = self.env
        _, rewards, _, _, info = e.step({"red_0": action, "blue_0": self._keeper_action()})
        self.t += 1
        events = [ev[0] for ev in e.events]
        r = float(rewards["red_0"])
        terminated = "goal" in events or e.holder == "blue_0"
        if e.holder == "blue_0" and "goal" not in events:
            r = -1.0
        if e.holder is None and not e.ball.moving:          # a kick that ended nowhere
            terminated = True
        truncated = (not terminated) and self.t >= self.max_steps
        return e.observe("red_0"), r, terminated, truncated, {"events": list(e.events)}

    def render(self): return self.env.render()

    def tabular_state(self, obs) -> int:
        """(attacker x, attacker y, keeper row) -> one integer; the attacker always holds the ball while the episode runs."""
        x, y = int(obs["self"][0]), int(obs["self"][1]); ky = int(obs["opponents"][0][1])
        return (x * self.rink.height + y) * len(self.rows) + (ky - self.rows[0])

    @property
    def n_states(self) -> int: return self.rink.width * self.rink.height * len(self.rows)


class SoloEnv(gym.Env):
    """You are red_0 in a 1v1 against `opponent` (a team module or its name); the opponent plays with its own tier."""
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, opponent="baselines.scripted", tier="discrete", **env_kwargs):
        self.opp_mod = load_team(opponent)
        self.env = Grid2v2(players_per_team=1, tiers={"red": tier, "blue": getattr(self.opp_mod, "TIER", "discrete")}, **env_kwargs)
        self.observation_space = self.env.observation_space("red_0")
        self.action_space = self.env.action_space("red_0")
        self.rink = self.env.rink
        self.opp = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _ = self.env.reset(seed=None if seed is None else int(seed))
        self.opp = self.opp_mod.make_team("blue", ["blue_0"], self.env)["blue_0"]; self.opp.reset()
        self.last_obs = obs
        return obs["red_0"], {}

    def step(self, action):
        obs, rewards, terms, truncs, infos = self.env.step({"red_0": action, "blue_0": self.opp.act(self.last_obs["blue_0"])})
        self.last_obs = obs
        return obs["red_0"], float(rewards["red_0"]), bool(terms["red_0"]), bool(truncs["red_0"]), {"events": list(self.env.events), "score": dict(self.env.score)}

    def render(self): return self.env.render()
