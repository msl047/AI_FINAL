"""Team 2 Chip with configurable kick, chase, and restart tactics."""

import json
import math
from pathlib import Path

import numpy as np

from grid2v2 import E, N, S, W
from grid2v2.actions import MOVE_DELTA
from grid2v2.agents import manhattan
from grid2v2.physics import predict, simulate
from .. import chip as team2_chip

TIER = team2_chip.TIER
MODEL = Path(__file__).with_name("policy.json")
SLOTS = ("deep_pressure", "midfield_pressure", "early_release", "restart")
STYLES = ("current", "carry", "short", "long", "wide_north", "wide_south", "race")
ROUTES = ("script", "x_first", "y_first")
RESTARTS = ("current", "chip_north", "chip_south", "shift_north", "shift_south")
GENES = tuple(
    f"{style}/{route}"
    for style in STYLES
    for route in (("script",) if style == "carry" else ROUTES)
)
GENE_OPTIONS = (GENES, GENES, GENES, RESTARTS)
DEFAULT = ("current/script", "current/script", "current/script", "current")
ANGLES = tuple(math.radians(a) for a in range(-90, 91, 15))
POWERS = (1.5, 2.0, 2.5, 3.0)


class ExploreField(team2_chip.ChipField):
    def __init__(self, original, genome, trace):
        super().__init__(
            original.adapter,
            original.rink,
            index=original.index,
            policies=original.policies,
            chip_policy=original.chip_policy,
        )
        self.genome = genome
        self.trace = trace

    def reset(self):
        super().reset()
        self.route = "script"
        self.opponent_score = 0
        self.restart_pending = None
        self.proposal = None

    def find_chip(self, me, opps, occupied, fallback):
        original = super().find_chip(me, opps, occupied, fallback)
        if me[0] >= 13 or self.adapter.decode(fallback, True).kick is not None:
            return original

        nearest = min(manhattan(me, pos) for pos, _ in opps)
        if nearest > 4:
            return original

        slot = 2 if nearest > 2 else int(me[0] >= 7)
        style, route = self.genome[slot].split("/")
        chosen = original

        if style == "carry":
            chosen = None
        elif style != "current":
            alternate = self.search_kick(style, me, opps, occupied)
            if alternate is not None:
                chosen = (alternate, "explore")

        before = (
            original[0]
            if original and self.chip_policy.get(original[1], True)
            else fallback
        )
        after = (
            chosen[0]
            if chosen and self.chip_policy.get(chosen[1], True)
            else fallback
        )
        is_chip = self.adapter.decode(after, True).kick is not None
        self.proposal = (before, after, is_chip, route)
        return chosen

    def search_kick(self, style, me, opps, occupied):
        best = None
        best_rank = None

        for angle in ANGLES:
            for power in POWERS:
                action = self.adapter.encode_kick(angle, power)
                kick = self.adapter.decode(action, True).kick
                event, ball, flight, _ = simulate(self.rink, *me, *kick, occupied)

                cell = (ball.cx, ball.cy)
                gain = cell[0] - me[0]
                lateral = cell[1] - me[1]

                if event is not None or self.forbidden(cell) or gain < 0:
                    continue
                if style == "short" and not 1 <= gain <= 3:
                    continue
                if style == "long" and gain < 4:
                    continue
                if style == "wide_north" and lateral < 2:
                    continue
                if style == "wide_south" and lateral > -2:
                    continue
                if style == "race" and gain < 1:
                    continue

                ours = max(flight, 1 + manhattan(me, cell))
                theirs = min(manhattan(pos, cell) for pos, _ in opps)
                margin = theirs - ours
                if margin < -1:
                    continue

                if style in ("wide_north", "wide_south"):
                    rank = (abs(lateral), margin, gain, -ours)
                elif style == "long":
                    rank = (gain, margin, -ours)
                else:
                    rank = (margin, gain, -ours)

                if best_rank is None or rank > best_rank:
                    best, best_rank = action, rank

        return best

    def legal(self, cell, occupied):
        return (
            self.rink.inside(*cell)
            and cell not in occupied
            and not self.forbidden(cell)
        )

    def restart_action(self, obs, after_goal):
        pending = self.restart_pending
        self.restart_pending = None

        me, have, _, _, _, occupied, _ = self._parse(obs)
        if not have or self.i_am_stunned:
            return None

        opening = self.genome[3]
        if pending is None:
            kickoff = (self.rink.width // 2 - 2, self.rink.height // 2)
            if not after_goal or me != kickoff or opening == "current":
                return None

            pending = "north" if opening.endswith("north") else "south"
            if opening.startswith("shift"):
                move = N if pending == "north" else S
                dx, dy = MOVE_DELTA[move]
                if self.legal((me[0] + dx, me[1] + dy), occupied):
                    self.restart_pending = pending
                    return self.adapter.encode_move(move)

        aim = math.radians(30 if pending == "north" else -30)
        action = self.adapter.encode_kick(aim, 3.0)
        kick = self.adapter.decode(action, True).kick
        event, ball, _, _ = simulate(self.rink, *me, *kick, occupied)

        if (
            event is None
            and ball.cx >= me[0] + 2
            and self.legal((ball.cx, ball.cy), occupied)
        ):
            return action
        return None

    def record_chip(self, obs, action, route):
        self.route = route
        if self.trace is not None:
            x, y = map(int, obs["self"][:2])
            kick = self.adapter.decode(action, True).kick
            _, rest, _, _ = simulate(self.rink, x, y, *kick, {})
            self.trace["chips"] += 1
            self.trace["wide"] += abs(rest.cy - y) >= 2

    def act(self, obs):
        score = int(obs["score"][1])
        after_goal = score > self.opponent_score
        self.opponent_score = score

        if (
            obs["self"][2] > 0.5
            or obs["mate"][2] > 0.5
            or np.any(obs["opponents"][:, 2] > 0.5)
        ):
            self.route = "script"

        self.proposal = None
        action = super().act(obs)

        if after_goal or self.restart_pending is not None:
            opening = self.restart_action(obs, after_goal)
            if opening is not None:
                if self.trace is not None and not np.array_equal(action, opening):
                    self.trace["restart_changes"] += 1
                if self.adapter.decode(opening, True).kick is not None:
                    self.record_chip(obs, opening, "y_first")
                return opening

        if self.proposal is not None:
            before, after, is_chip, route = self.proposal
            if np.array_equal(action, after):
                if self.trace is not None and not np.array_equal(before, after):
                    self.trace["kick_changes"] += 1
                if is_chip:
                    self.record_chip(obs, action, route)

        return action

    def _loose(self, me, mate, opps, occupied, ball):
        fallback = super()._loose(me, mate, opps, occupied, ball)
        if self.route == "script" or self.i_am_stunned:
            return fallback

        bx, by, vx, vy = ball
        if math.hypot(vx, vy) > 1e-6:
            _, rest, *_ = predict(self.rink, bx, by, vx, vy, {})
            x, y = rest.cx, rest.cy
        else:
            x, y = round(bx), round(by)

        towards_x = (E if x > me[0] else W) if x != me[0] else None
        towards_y = (N if y > me[1] else S) if y != me[1] else None
        moves = (
            (towards_x, towards_y)
            if self.route == "x_first"
            else (towards_y, towards_x)
        )

        for move in moves:
            if move is None:
                continue

            dx, dy = MOVE_DELTA[move]
            if self.legal((me[0] + dx, me[1] + dy), occupied):
                action = self.adapter.encode_move(move)
                if self.trace is not None and not np.array_equal(action, fallback):
                    self.trace["chase_changes"] += 1
                return action

        return fallback


def build_team(side, ids, env, genome, trace=None):
    bots = team2_chip.make_team(side, ids, env)
    bots[ids[0]] = ExploreField(bots[ids[0]], genome, trace)
    return bots


def make_team(side, ids, env):
    if not MODEL.exists():
        return team2_chip.make_team(side, ids, env)

    genome = tuple(json.loads(MODEL.read_text(encoding="utf-8"))["genome"])
    return build_team(side, ids, env, genome)