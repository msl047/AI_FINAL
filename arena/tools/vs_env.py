"""tools/vs_env.py — train one team against a fixed opponent, as a PettingZoo ParallelEnv of the red robots only.

TeamVsEnv exposes red_0 and red_1 (flat Box observations with a one-hot robot index appended, so a shared policy can
specialise roles) and drives the blue robots with a team module (baselines.scripted, ...) or a frozen policy
(`opponent=("policy", predict_fn)` for self-play).  With SuperSuit this becomes an SB3 VecEnv:

    env = ss.pettingzoo_env_to_vec_env_v1(TeamVsEnv(...)); env = ss.concat_vec_envs_v1(env, 8, base_class="stable_baselines3")
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from grid2v2 import Grid2v2, flatten, N_FEATURES
from grid2v2.match import load_team, tier_of

N_OBS = N_FEATURES + 2


def flat_obs(obs, rink, index):
    v = flatten(obs, rink.width, rink.height)
    return np.concatenate([v, np.eye(2, dtype=np.float32)[index]])


class TeamVsEnv(ParallelEnv):
    metadata = {"name": "grid2v2_vs", "is_parallelizable": True}

    def __init__(self, opponent="baselines.scripted", tier="discrete", shaping=0.0, env_kwargs=None, seed=0,
                 shaping_mode="progress", possession_bonus=0.0, gamma=0.99):
        self.opp_spec = opponent
        self.pool = []                                           # (module or None, policy or None, blue tier)
        specs = opponent if isinstance(opponent, list) else ([opponent] if isinstance(opponent, tuple) else opponent.split(","))
        for sp in specs:
            if isinstance(sp, tuple) and sp[0] == "policy":
                self.pool.append((None, sp[1], sp[2] if len(sp) > 2 else tier))
            else:
                m = load_team(sp); self.pool.append((m, None, tier_of(m)))
        self.envs = {}
        for _, _, bt in self.pool:
            key = bt if isinstance(bt, str) else bt.name
            if key not in self.envs: self.envs[key] = Grid2v2(tiers={"red": tier, "blue": bt}, **(env_kwargs or {}))
        self.opp_mod, self.opp_policy, blue_tier = self.pool[0]
        self.env = self.envs[blue_tier if isinstance(blue_tier, str) else blue_tier.name]
        self.rink = self.env.rink
        self.possible_agents = ["red_0", "red_1"]; self.agents = []
        self.shaping, self.shaping_mode, self.possession_bonus, self.gamma = shaping, shaping_mode, possession_bonus, gamma
        self._obs_space = spaces.Box(-1.0, 1.0, shape=(N_OBS,), dtype=np.float32)
        self.seed0 = seed; self.episodes = 0
        self.render_mode = None

    def observation_space(self, agent): return self._obs_space
    def action_space(self, agent): return self.env.action_space(agent)

    def _flat(self, obs):
        return {a: flat_obs(obs[a], self.rink, int(a[-1])) for a in self.possible_agents}

    def reset(self, seed=None, options=None):
        s = seed if seed is not None else self.seed0 + self.episodes
        self.episodes += 1
        self.opp_mod, self.opp_policy, bt = self.pool[self.episodes % len(self.pool)]
        self.env = self.envs[bt if isinstance(bt, str) else bt.name]
        obs, infos = self.env.reset(seed=s)
        self.agents = list(self.possible_agents)
        if self.opp_mod is not None:
            self.opp = self.opp_mod.make_team("blue", self.env.members["blue"], self.env)
            for a in self.opp.values(): a.reset()
        self.last = obs
        return self._flat(obs), {a: {} for a in self.agents}

    def _blue_actions(self):
        if self.opp_mod is not None:
            return {a: self.opp[a].act(self.last[a]) for a in self.env.members["blue"]}
        return {a: self.opp_policy(flat_obs(self.last[a], self.rink, int(a[-1]))) for a in self.env.members["blue"]}

    def _phi(self, o):
        """Potential for shaping (r += gamma * phi(s') - phi(s)).  'progress': the ball's x towards the attacked goal,
        whoever has it.  'holder': ball x when we hold it, minus the distance left when they do, zero when loose."""
        bx = o["ball"][0] / (self.rink.width - 1)
        mine = o["self"][2] > 0.5 or o["mate"][2] > 0.5
        theirs = any(r[2] > 0.5 for r in o["opponents"])
        if self.shaping_mode == "progress":
            return bx
        if self.shaping_mode == "chase":          # ball progress, plus "get to the ball" when we do not have it
            d = (abs(o["self"][0] - o["ball"][0]) + abs(o["self"][1] - o["ball"][1])) / (self.rink.width + self.rink.height)
            return bx - (0.0 if mine else 0.5 * d)
        return bx if mine else (-(1 - bx) if theirs else 0.0)

    def step(self, actions):
        acts = dict(actions); acts.update(self._blue_actions())
        obs, rewards, terms, truncs, infos = self.env.step(acts)
        r = {a: float(rewards[a]) for a in self.possible_agents}
        for a in self.possible_agents:
            if self.shaping:
                r[a] += self.shaping * (self.gamma * self._phi(obs[a]) - self._phi(self.last[a]))
            if self.possession_bonus and (obs[a]["self"][2] > 0.5 or obs[a]["mate"][2] > 0.5):
                r[a] += self.possession_bonus
        self.last = obs
        done = not self.env.agents
        t = {a: bool(terms[a]) for a in self.possible_agents}; tr = {a: bool(truncs[a]) for a in self.possible_agents}
        if done: self.agents = []
        return self._flat(obs), r, t, tr, {a: {"score": dict(self.env.score)} for a in self.possible_agents}

    def render(self): return self.env.render()
    def close(self): pass
