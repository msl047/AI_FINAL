"""baselines/random.py — uniformly random actions in the Discrete(9) space."""
import numpy as np
from grid2v2.agents import RandomAgent

TIER = "discrete"


def make_team(team, agent_ids, env):
    return {a: RandomAgent(env.adapters[a], env.rink, index=i, rng=np.random.default_rng(i)) for i, a in enumerate(agent_ids)}
