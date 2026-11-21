"""teams/template — copy this folder to teams/<your_team_name>/ and edit.  This folder IS your submission.

What the tournament runner reads from this folder (nothing else):
    TIER                            "discrete" | "box" | "fine" | an Adapter instance from grid2v2.actions
    make_team(team, agent_ids, env) -> {agent_id: Agent}   called once per match, inside the 15 s setup budget

Everything else (weights, helper modules, your feature function) lives in this folder and is imported relatively, so
the folder works wherever it is copied.  Allowed imports: the standard library, numpy, torch, gymnasium, pettingzoo,
stable_baselines3, grid2v2.  Nothing outside this folder is yours to change: not grid2v2/, not baselines/.
"""

from grid2v2.agents import ScriptedAgent

from .agent import FieldAgent

TIER = "box"


def make_team(team, agent_ids, env):
    agents = {}
    for index, agent_id in enumerate(agent_ids):
        agent_type = FieldAgent if index == 0 else ScriptedAgent
        agents[agent_id] = agent_type(env.adapters[agent_id], env.rink, index=index)
    return agents
