"""baselines/scripted_box.py — the same scripted policy driving the Tier 2 Box(4) space: aimed shots and passes."""
from grid2v2.agents import ScriptedAgent

TIER = "box"


def make_team(team, agent_ids, env):
    return {a: ScriptedAgent(env.adapters[a], env.rink, index=i) for i, a in enumerate(agent_ids)}
