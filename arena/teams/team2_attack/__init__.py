"""Team 2 Chip with learned, three-step plans for opening a shot."""

import json
from pathlib import Path

import numpy as np

from grid2v2 import E, N, S
from grid2v2.actions import MOVE_DELTA
from grid2v2.features import N_FEATURES, flatten
from teams import team2_chip

TIER = team2_chip.TIER
MODEL = Path(__file__).with_name("team2_attack_policy.json")
PLANS = {"carry": E, "shift_up": N, "shift_down": S}
N_INPUTS = N_FEATURES + 1
MIN_GAIN = 0.025


def feature_vector(obs, rink):
    return np.concatenate(([1.0], flatten(obs, rink.width, rink.height)))


def choose_plan(weights, features, options):
    scores = [
        (float(features @ weights[name]), name)
        for name in options
        if name in weights
    ]
    gain, plan = max(scores, default=(0.0, None))
    return plan if gain > MIN_GAIN else None


class AttackField(team2_chip.ChipField):
    def __init__(self, original, weights=None, samples=None, counts=None):
        super().__init__(
            original.adapter,
            original.rink,
            index=original.index,
            policies=original.policies,
            chip_policy=original.chip_policy,
        )
        self.weights = {} if weights is None else weights
        self.samples = samples
        self.counts = counts

    def reset(self):
        super().reset()
        self.plan = None
        self.remaining = 0
        self.used_plan = False

    def start_plan(self, plan):
        self.plan = plan
        self.remaining = 3
        self.used_plan = True
        if self.counts is not None:
            self.counts[plan] += 1

    def legal_move(self, move, me, occupied):
        dx, dy = MOVE_DELTA[move]
        target = (me[0] + dx, me[1] + dy)
        return (
            self.rink.inside(*target)
            and target not in occupied
            and not self.forbidden(target)
        )

    def act(self, obs):
        rng_before = (
            self.policy_rng.bit_generator.state
            if self.samples is not None else None
        )
        scripted = super().act(obs)
        x, _, has_ball, stunned = obs["self"]

        if has_ball < 0.5:
            self.used_plan = False
            self.remaining = 0
            return scripted

        if stunned > 0.5 or self.adapter.decode(scripted, True).kick is not None:
            self.remaining = 0
            return scripted

        can_start = (
            not self.used_plan
            and x >= self.rink.width - 7
            and (self.weights or self.samples is not None)
        )
        if not self.remaining and not can_start:
            return scripted

        me, _, _, _, _, occupied, _ = self._parse(obs)

        if not self.remaining:
            options = [
                name for name, move in PLANS.items()
                if self.legal_move(move, me, occupied)
            ]
            features = feature_vector(obs, self.rink)

            if self.samples is not None and options:
                tick = int(obs["t"][0])
                self.samples[tick] = (features, options, rng_before)

            plan = choose_plan(self.weights, features, options)
            if plan is None:
                return scripted
            self.start_plan(plan)

        move = PLANS[self.plan]
        if not self.legal_move(move, me, occupied):
            self.remaining = 0
            return scripted

        self.remaining -= 1
        action = self.adapter.encode_move(move)
        if self.counts is not None and not np.array_equal(action, scripted):
            self.counts["changed_moves"] += 1
        return action


def build_team(side, ids, env, weights=None, samples=None, counts=None):
    bots = team2_chip.make_team(side, ids, env)
    bots[ids[0]] = AttackField(bots[ids[0]], weights, samples, counts)
    return bots


def make_team(side, ids, env):
    if not MODEL.exists():
        return team2_chip.make_team(side, ids, env)

    model = json.loads(MODEL.read_text(encoding="utf-8"))
    weights = {name: np.array(row) for name, row in model["weights"].items()}
    return build_team(side, ids, env, weights)