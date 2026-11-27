"""Explore two attacking robots, coordinated pressure, and recovery."""

import json
import math
from pathlib import Path

import numpy as np

from grid2v2 import E, N, S, STAY, W
from grid2v2.actions import MOVE_DELTA
from grid2v2.agents import ScriptedAgent, manhattan
from grid2v2.physics import predict, simulate
from .. import teamplay as previous

TIER = "box"
WINNER = ("current", 3, 3, 2, "nearest", 5, 4, "balanced", -1, 2, "carry", "north")
MODEL = Path(__file__).with_name("policy.json")
NAMES = (
    "attack", "attack_x", "support_distance", "safety_gap",
    "press", "press_x", "press_reach", "wall_distance",
    "recover", "danger_x", "cover_x", "handoff_bias",
)
OPTIONS = (
    ("current", "both_field", "forward", "wide", "give_go"),
    (4, 6, 8, 10, 12), (2, 3, 4, 5), (1, 2, 3),
    ("current", "both", "trap", "cut_outlet"),
    (7, 8, 9, 10, 11, 12), (3, 4, 5, 6, 8), (1, 2, 3, 4),
    ("current", "screen", "shot_lanes", "mark_receiver"),
    (5, 6, 7, 8, 9, 10), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3),
)
FAMILIES = {"attack": slice(0, 4), "press": slice(4, 8), "recover": slice(8, 12)}
DEFAULT = ("current", 8, 3, 2, "current", 9, 5, 2, "current", 7, 3, 1)
MOVES = (STAY, E, N, S, W)


def normalize(genome):
    genes = list(genome)
    for slots in FAMILIES.values():
        if genes[slots.start] == "current":
            genes[slots] = DEFAULT[slots]
    if genes[0] == "both_field":
        genes[2:4] = DEFAULT[2:4]
    if genes[4] != "trap":
        genes[7] = DEFAULT[7]
    return tuple(genes)


class PressPlayer:
    def __init__(self, winner, genome, trace):
        self.winner = winner
        self.index = winner.index
        self.adapter = winner.adapter
        self.rink = winner.rink
        self.reference = ScriptedAgent(self.adapter, self.rink, index=0)
        self.genome = genome
        self.trace = trace
        _, _, spacing, gap = genome[:4]
        genes = ("lead", spacing, 5, gap, *previous.DEFAULT[4:])
        self.passer = previous.TeamPlayer(self.reference, genes, None)

    def reset(self):
        self.winner.reset()
        self.reference.reset()
        self.last_owner = None
        self.their_score = 0

    def legal(self, cell, occupied):
        return self.rink.inside(*cell) and cell not in occupied and not self.reference.forbidden(cell)

    def move(self, me, target, occupied, allow=()):
        move = self.reference._toward(me, target, occupied, allow=allow)
        return self.adapter.encode_move(move)

    def attack(self, obs, fallback, me, have, mate, enemies, occupied):
        mode, start, spacing, gap = self.genome[:4]
        passer, receiver = (me, mate) if have else (mate, me)
        if passer[0] < start:
            return None

        if have:
            shot = self.adapter.decode(fallback, True).kick
            if shot is not None and simulate(
                self.rink, *me, *shot, self.reference._kick_occ(me, occupied), max_steps=2
            )[0] == ("goal", "right"):
                return fallback

        if mode == "both_field":
            return self.reference.act(obs)

        if mode == "give_go":
            if have or obs["mate"][3] < 0.5:
                plan = self.passer.pass_plan(
                    passer, receiver, enemies, have and obs["mate"][3] > 0.5,
                )
                if plan is not None:
                    kick, move = plan
                    return kick if have else self.adapter.encode_move(move)

        if have:
            if self.index == 1:
                return self.reference.act(obs)
            return None

        max_x = self.rink.width - 1
        if mode == "give_go":
            dx, lanes = min(2, spacing), (-1, 1, -2, 2)
        elif mode == "wide":
            dx, lanes = max(1, spacing - 2), (-3, 3, -2, 2)
        else:
            dx, lanes = spacing, (-2, 2, -1, 1)
        ideals = [
            (min(max_x, mate[0] + dx), min(self.rink.height - 1, max(0, mate[1] + dy)))
            for dy in lanes
        ]
        valid = [
            cell for cell in ideals if self.legal(cell, occupied)
            and min(manhattan(cell, p) for p in enemies) >= gap
        ]
        if not valid:
            return None
        target = max(valid, key=lambda c: (
            min(manhattan(c, p) for p in enemies) - 0.5 * manhattan(me, c),
            -manhattan(mate, c),
        ))
        return self.move(me, target, occupied)

    def roles(self, me, mate, threat, mate_stunned, handoff=0):
        players = ((self.index, me, False), (1 - self.index, mate, mate_stunned))
        return min(players, key=lambda row: (
            manhattan(row[1], threat) + 12 * row[2]
            + handoff * int(row[1][0] < threat[0] and row[1][0] < min(me[0], mate[0]) + 1),
            row[0],
        ))[0]

    def pressure(self, me, mate, carrier, receiver, target, occupied, mate_stunned):
        mode, start, reach, wall = self.genome[4:8]
        if target[0] < start or max(manhattan(me, target), manhattan(mate, target)) > reach:
            return None
        on_wall = min(target[1], self.rink.height - 1 - target[1]) <= wall
        if mode == "trap" and not on_wall:
            return None

        lead = self.roles(me, mate, target, mate_stunned)
        if self.index == lead or mode == "both":
            return self.move(me, target, occupied, allow=(carrier,) if carrier else ())

        lane = target[1] if receiver is None else receiver[1]
        x = max(0, target[0] - 1)
        if mode == "cut_outlet" and receiver is not None:
            x = max(0, min(target[0] - 1, round((target[0] + receiver[0]) / 2)))
        sign = 1 if lane > target[1] else -1
        y = min(self.rink.height - 1, max(0, target[1] + sign))
        for cell in (
            (x, y), (x, target[1]),
            (x, max(0, min(self.rink.height - 1, target[1] - sign))),
        ):
            if self.legal(cell, occupied):
                return self.move(me, cell, occupied)
        return None

    def recovery(self, me, mate, carrier, receiver, target, occupied, mate_stunned):
        mode, _, depth, handoff = self.genome[8:]
        if self.roles(me, mate, target, mate_stunned, handoff) == self.index:
            return self.move(me, target, occupied, allow=(carrier,) if carrier else ())

        x = min(depth, max(0, target[0] - 2))
        middle = self.rink.height // 2
        y = round(middle + (target[1] - middle) * (x + 0.5) / (target[0] + 0.5))
        if mode == "mark_receiver" and receiver is not None and receiver[0] < target[0]:
            y = round((y + receiver[1]) / 2)
        desired = (x, y)

        if mode == "shot_lanes" and carrier is not None:
            shots = []
            for fraction in (0.12, 0.32, 0.50, 0.68, 0.88):
                goal_y = self.rink.goal_lo + fraction * (self.rink.goal_hi - self.rink.goal_lo)
                angle = math.atan2(goal_y - carrier[1], -0.5 - carrier[0])
                if simulate(self.rink, *carrier, angle, 3.0, {})[0] == ("goal", "left"):
                    shots.append(angle)
            if shots:
                def blocks(cell):
                    return sum(simulate(
                        self.rink, *carrier, angle, 3.0, {cell: "cover"}
                    )[0] == ("taken", "cover") for angle in shots)

                choices = []
                for move in MOVES:
                    dx, dy = MOVE_DELTA[move]
                    cell = (me[0] + dx, me[1] + dy)
                    if self.legal(cell, occupied):
                        choices.append(((blocks(cell), -manhattan(cell, desired)), move))
                if choices:
                    value, move = max(choices, key=lambda item: item[0])
                    if value[0]:
                        return self.adapter.encode_move(move)

        for row in (y, y - 1, y + 1):
            if self.legal((x, row), occupied):
                return self.move(me, (x, row), occupied)
        return None

    def act(self, obs):
        fallback = self.winner.act(obs)
        me, have, mate, mate_has, opponents, occupied, ball = self.reference._parse(obs)
        enemies = [pos for pos, _ in opponents]
        carrier = next((pos for pos, holding in opponents if holding), None)
        receiver = next((pos for pos in enemies if pos != carrier), None)
        conceded = obs["score"][1] > self.their_score
        self.their_score = int(obs["score"][1])
        mate_stunned = obs["mate"][3] > 0.5
        if have or mate_has:
            self.last_owner = "ours"
        elif carrier is not None:
            self.last_owner = "theirs"
        if self.reference.i_am_stunned or mate is None or not enemies or conceded:
            return fallback

        action, kind = None, None
        attack = self.genome[0]
        if have or mate_has:
            if attack != "current":
                action = self.attack(obs, fallback, me, have, mate, enemies, occupied)
                kind = "attack"
        elif self.last_owner == "ours" and attack == "both_field" and ball[0] >= self.genome[1]:
            action, kind = self.reference.act(obs), "attack"
        elif self.last_owner != "ours":
            if carrier is not None:
                target = carrier
            elif math.hypot(*ball[2:]) > 1e-6:
                _, rest, *_ = predict(self.rink, *ball, {})
                target = (rest.cx, rest.cy)
            else:
                target = (round(ball[0]), round(ball[1]))

            if self.genome[8] != "current" and target[0] <= self.genome[9]:
                action = self.recovery(me, mate, carrier, receiver, target, occupied, mate_stunned)
                kind = "recovery"
            elif self.genome[4] != "current":
                action = self.pressure(me, mate, carrier, receiver, target, occupied, mate_stunned)
                kind = "pressure"

        if action is None:
            return fallback
        if self.trace is not None and not np.array_equal(action, fallback):
            self.trace[kind] += 1
        return action


def build_team(side, ids, env, genome, trace=None):
    players = previous.build_team(side, ids, env, WINNER)
    if genome == DEFAULT:
        return players
    return {
        agent: PressPlayer(players[agent], genome, None if trace is None else trace[i])
        for i, agent in enumerate(ids)
    }


def make_team(side, ids, env):
    genome = DEFAULT
    if MODEL.is_file():
        genome = tuple(json.loads(MODEL.read_text(encoding="utf-8"))["genome"])
        if len(genome) != len(OPTIONS) or any(
            value not in options for value, options in zip(genome, OPTIONS)
        ):
            raise ValueError(f"Invalid policy in {MODEL}")
    return build_team(side, ids, env, genome)