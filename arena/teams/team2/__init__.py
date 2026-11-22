"""Team 2 policy based on field position"""

from pathlib import Path

import numpy as np

from grid2v2.actions import MOVE_DELTA
from grid2v2.agents import ScriptedAgent
from grid2v2.features import N_FEATURES, flatten
from grid2v2.physics import simulate

from .agent import FieldAgent

TIER = "box"
TEAM_DIR = Path(__file__).resolve().parent
MIDFIELD_X = 7
FINISHING_X = 15
PHASES = ("defense", "midfield", "finishing")
N_INPUTS = N_FEATURES - 3
N_CHOICES = 6
USE_SCRIPT = 0
PASS = 5


def phase_at(x):
    if x < MIDFIELD_X:
        return "defense"
    if x < FINISHING_X:
        return "midfield"
    return "finishing"


def features(obs, rink):
    values = flatten(obs, rink.width, rink.height)[:N_INPUTS]
    return np.clip(values, -1.0, 1.0)


class Team2Field(FieldAgent):
    def __init__(self, adapter, rink, index=0, policies=None):
        super().__init__(adapter, rink, index=index)
        self.policies = {} if policies is None else policies
        self.reset()

    def reset(self):
        super().reset()
        self.policy_rng = np.random.default_rng(0)

    def act(self, obs):
        x, _, has_ball, stunned = obs["self"]
        weights = self.policies.get(phase_at(x))
        if weights is None or has_ball < 0.5 or stunned > 0.5:
            return super().act(obs)

        hidden = np.tanh(weights["w1"] @ features(obs, self.rink) + weights["b1"])
        logits = weights["w2"] @ hidden + weights["b2"]
        probabilities = np.exp(logits - logits.max())
        choice = int(
            self.policy_rng.choice(N_CHOICES, p=probabilities / probabilities.sum())
        )
        return self.choose_action(obs, choice)

    def choose_action(self, obs, choice):
        fallback = super().act(obs)
        if choice == USE_SCRIPT or obs["self"][2] < 0.5 or obs["self"][3] > 0.5:
            return fallback

        me, _, mate, _, _, occupied, _ = self._parse(obs)
        kick_occupied = self._kick_occ(me, occupied)

        # Take a shot that can score before the defenders move.
        kick = self.adapter.decode(fallback, True).kick
        if kick is not None:
            event, *_ = simulate(self.rink, *me, *kick, kick_occupied, max_steps=1)
            if event == ("goal", "right"):
                return fallback

        if choice == PASS:
            _, passes = self._kick_candidates(me, mate)
            for angle, speed in passes:
                action = self.adapter.encode_kick(angle, speed)
                kick = self.adapter.decode(action, True).kick
                event, *_ = simulate(self.rink, *me, *kick, kick_occupied)
                if event == ("taken", "mate"):
                    return action
            return fallback

        dx, dy = MOVE_DELTA[choice]
        target = (me[0] + dx, me[1] + dy)
        if (
            self.rink.inside(*target)
            and not self.forbidden(target)
            and target not in occupied
        ):
            return self.adapter.encode_move(choice)
        return fallback


def make_team(team, agent_ids, env):
    policies = {}
    for phase in PHASES:
        path = TEAM_DIR / f"{phase}.npz"
        if path.is_file():
            with np.load(path) as weights:
                policies[phase] = dict(weights)

    agents = {}
    for index, agent_id in enumerate(agent_ids):
        adapter = env.adapters[agent_id]
        if index == 0:
            agents[agent_id] = Team2Field(adapter, env.rink, index, policies=policies)
        else:
            agents[agent_id] = ScriptedAgent(adapter, env.rink, index)
    return agents