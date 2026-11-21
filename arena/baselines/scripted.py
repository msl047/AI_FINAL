"""baselines/scripted.py — the Tier 1 baseline: ScriptedAgent on Discrete(9).  Beat this."""
from grid2v2.agents import ScriptedAgent

TIER = "discrete"


def make_team(team, agent_ids, env):
    return {a: ScriptedAgent(env.adapters[a], env.rink, index=i) for i, a in enumerate(agent_ids)}
