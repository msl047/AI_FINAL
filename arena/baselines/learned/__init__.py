"""baselines/learned — a Tier 1 team trained by PPO against the three scripted baselines (6 M steps under the v0.4
rules; see ../internal).  Clears and counters; hard to score on.  Beat it and you have beaten something that learned."""
from .agent import PolicyAgent

TIER = "discrete"


def make_team(team, agent_ids, env):
    shared = PolicyAgent.load_shared()
    return {a: PolicyAgent(env.adapters[a], env.rink, index=i, shared=shared) for i, a in enumerate(agent_ids)}
