"""Four experimental tactics on top of the p05414 keeper and field player."""

import json
import math
from pathlib import Path

import numpy as np

from grid2v2 import E, N, S, STAY, W
from grid2v2.actions import MOVE_DELTA
from grid2v2.agents import manhattan
from grid2v2.physics import predict, simulate
from .. import explore as team2_explore

TIER = "box"
ATTACK = ("wide_south/y_first", "race/y_first", "carry/script", "chip_north")
MODEL = Path(__file__).with_name("policy.json")
NAMES = (
    "passing", "spacing", "pass_pressure", "min_pass_margin",
    "defense", "cover_depth", "press_bonus",
    "clearance", "clear_margin", "clear_gain",
    "restart", "first_move",
)
OPTIONS = (
    ("current", "pass_only", "run_only", "direct", "lead", "give_go", "wide"),
    (2, 3, 4, 5), (2, 3, 4, 5, 6), (1, 2, 3),
    ("current", "nearest", "goal_side", "cut_lane"), (1, 2, 3, 4, 5), (1, 2, 3, 4),
    ("current", "territory", "balanced", "outlet"), (-1, 0, 1, 2), (2, 3, 4, 5),
    ("current", "chip_south", "shift_north", "shift_south", "carry", "script"),
    ("hold", "forward", "north", "south"),
)
FAMILIES = {
    "passing": slice(0, 4), "defense": slice(4, 7),
    "clearance": slice(7, 10), "restart": slice(10, 12),
}
DEFAULT = ("current", 3, 3, 2, "current", 2, 2, "current", -1, 3, "current", "hold")
TERRITORY = (*DEFAULT[:7], "territory", -1, 3, *DEFAULT[10:])
POWERS = (1.5, 2.0, 3.0)
ANGLES = tuple(math.radians(a) for a in range(-90, 91, 15))
MOVES = (E, N, S, W)


def normalize(genome):
    genes = list(genome)
    for section in FAMILIES.values():
        if genes[section.start] == "current":
            genes[section] = DEFAULT[section]
    if genes[10] != "carry":
        genes[11] = "hold"
    return tuple(genes)


class TeamPlayer:
    def __init__(self, original, genome, trace):
        self.base = original
        self.rink = original.rink
        self.adapter = original.adapter
        self.index = original.index
        self.genome = genome
        self.trace = trace

    def reset(self):
        self.base.reset()
        self.last_owner = None
        self.their_score = 0

    def legal(self, cell, occupied):
        return self.rink.inside(*cell) and cell not in occupied and not self.base.forbidden(cell)

    def movement(self, me, target, occupied, allow=()):
        return self.adapter.encode_move(self.base._toward(me, target, occupied, allow=allow))

    def pass_plan(self, carrier, receiver, enemies, receiver_stunned):
        mode, _, pressure, margin = self.genome[:4]
        pressured = min(manhattan(carrier, p) for p in enemies) <= pressure
        targets = [(receiver, STAY)]
        if mode in ("lead", "give_go", "wide") and not receiver_stunned:
            targets += [((receiver[0] + MOVE_DELTA[m][0], receiver[1] + MOVE_DELTA[m][1]), m)
                        for m in MOVES]
        best = None
        for target, move in targets:
            if not self.legal(target, set(enemies) | {carrier}):
                continue
            gain = target[0] - carrier[0]
            gap = min(manhattan(target, p) for p in enemies)
            if gap < margin or (gain < 1 and not (pressured and mode == "wide" and gain >= -1)):
                continue
            dx, dy = target[0] - carrier[0], target[1] - carrier[1]
            angle = math.atan2(dy, dx)
            snapped = round(angle / (math.pi / 8)) * (math.pi / 8)
            occupied = {p: "opponent" for p in enemies}
            if manhattan(carrier, receiver) >= self.base.min_pass:
                occupied[receiver] = "receiver"
            for aim in (angle, snapped, angle - math.pi / 18, angle + math.pi / 18):
                for power in POWERS:
                    action = self.adapter.encode_kick(aim, power)
                    kick = self.adapter.decode(action, True).kick
                    event, rest, *_ = simulate(self.rink, *carrier, *kick, occupied, max_steps=1)
                    direct = move == STAY and event == ("taken", "receiver")
                    lead = move != STAY and event is None and (rest.cx, rest.cy) == target
                    if not (direct or lead):
                        continue
                    score = (3 * gain + gap - 2 * int(lead), int(direct), -power)
                    if best is None or score > best[0]:
                        best = (score, action, move)
        return None if best is None else best[1:]

    def support(self, me, carrier, enemies, occupied):
        mode, spacing, _, _ = self.genome[:4]
        if self.index == 1 and carrier[0] < 4:
            return None
        dx = spacing if mode in ("give_go", "run_only") else min(spacing, 3)
        x = min(self.rink.width - 1 - self.base.crease, carrier[0] + dx)
        if mode == "wide":
            x = max(1, x - 2)
        if self.index == 1 and carrier[0] < 9 and mode != "give_go":
            x = min(x, carrier[0])
        lane = 3 if mode == "wide" else 2
        targets = [
            (x, min(self.rink.height - 1, max(0, carrier[1] + dy)))
            for dy in (-lane, -1, 1, lane)
        ]
        targets = [c for c in targets if self.legal(c, occupied)]
        if not targets:
            return None
        target = max(targets, key=lambda c: (
            min(manhattan(c, p) for p in enemies) - manhattan(me, c) / 2,
            -manhattan(carrier, c),
        ))
        return self.movement(me, target, occupied)

    def clear(self, me, mate, enemies, occupied):
        mode, minimum_margin, min_gain = self.genome[7:10]
        occupied = self.base._kick_occ(me, occupied)
        angles = (*ANGLES, math.atan2(mate[1] - me[1], mate[0] - me[0]))
        best = None
        for angle in angles:
            for power in POWERS:
                event, rest, turns, _ = simulate(self.rink, *me, angle, power, occupied)
                received = event == ("taken", "mate")
                if event is not None and not received:
                    continue
                cell = (rest.cx, rest.cy)
                gain = cell[0] - me[0]
                if gain < 1 or self.base.forbidden(cell):
                    continue
                ours = turns if received else max(turns, manhattan(mate, cell))
                theirs = min(manhattan(p, cell) for p in enemies)
                gap = theirs - ours
                if not received and (gain < min_gain or gap < minimum_margin):
                    continue
                if mode == "territory":
                    rank = (gain, received, gap)
                elif mode == "outlet":
                    rank = (received, gap, gain)
                else:
                    rank = (2 * gain + max(-2, min(gap, 5)) + 2 * int(received), received)
                if best is None or rank > best[0]:
                    best = (rank, self.adapter.encode_kick(angle, power))
        return None if best is None else best[1]

    def defend(self, me, mate, enemies, occupied, ball, carrier):
        mode, depth, bias = self.genome[4:7]
        if carrier is None:
            if math.hypot(*ball[2:]) > 1e-6:
                _, rest, *_ = predict(self.rink, *ball, {})
                threat = (rest.cx, rest.cy)
            else:
                threat = (round(ball[0]), round(ball[1]))
        else:
            threat = carrier
        if threat[0] > self.rink.width // 2 + bias:
            return None

        robots = ((self.index, me), (1 - self.index, mate))
        stunned_index = self.mate_stunned

        def priority(entry):
            index, place = entry
            cost = manhattan(place, threat)
            if mode == "goal_side" and place[0] >= threat[0]:
                cost += bias
            if index != self.index and stunned_index:
                cost += 10
            return cost, index

        presser = min(robots, key=priority)[0]
        if presser == self.index:
            return self.movement(me, threat, occupied, allow=(carrier,) if carrier else ())

        x = min(depth, max(0, threat[0] - 2))
        middle = self.rink.height // 2
        y = round(middle + (threat[1] - middle) * (x + 0.5) / (threat[0] + 0.5))
        if mode == "cut_lane":
            receiver = next((pos for pos in enemies if pos != carrier and pos[0] < threat[0]), None)
            if receiver is not None:
                y = round((y + receiver[1]) / 2)
        for row in (y, y - 1, y + 1):
            if self.legal((x, row), occupied):
                return self.movement(me, (x, row), occupied)
        return None

    def act(self, obs):
        score = int(obs["score"][1])
        restart = score > self.their_score
        self.their_score = score
        fallback = self.base.act(obs)
        me, have, mate, mate_has, opponents, occupied, ball = self.base._parse(obs)
        enemies = [pos for pos, _ in opponents]
        carrier = next((pos for pos, holding in opponents if holding), None)
        self.mate_stunned = obs["mate"][3] > 0.5
        if have or mate_has:
            self.last_owner = "us"
        elif carrier is not None:
            self.last_owner = "them"
        if self.base.i_am_stunned or mate is None or not enemies:
            return fallback

        action, kind = None, None
        if restart and self.index == 0 and have:
            if self.genome[10] == "carry":
                move = {"hold": STAY, "forward": E, "north": N, "south": S}[self.genome[11]]
                dx, dy = MOVE_DELTA[move]
                if self.legal((me[0] + dx, me[1] + dy), occupied):
                    action, kind = self.adapter.encode_move(move), "restart"
            if action is None:
                return fallback
        elif self.index == 1 and have and self.genome[7] != "current":
            action, kind = self.clear(me, mate, enemies, occupied), "clearance"
        elif (have or mate_has) and self.genome[0] != "current":
            passer, receiver = (me, mate) if have else (mate, me)
            if have:
                kick = self.adapter.decode(fallback, True).kick
                if kick is not None and simulate(
                    self.rink, *me, *kick, self.base._kick_occ(me, occupied), max_steps=2,
                )[0] == ("goal", "right"):
                    return fallback
            passer_stunned = self.mate_stunned if mate_has else False
            plan = None
            keeper_clearing = mate_has and self.index == 0 and self.genome[7] != "current"
            if keeper_clearing:
                return fallback
            if self.genome[0] != "run_only" and not passer_stunned:
                plan = self.pass_plan(passer, receiver, enemies, self.mate_stunned if have else False)
            if plan is not None:
                kick, move = plan
                if have:
                    action, kind = kick, "passing"
                elif self.genome[0] != "pass_only":
                    action, kind = self.adapter.encode_move(move), "support"
            elif mate_has and self.genome[0] != "pass_only":
                action, kind = self.support(me, mate, enemies, occupied), "support"
        elif self.genome[4] != "current" and self.last_owner != "us":
            action, kind = self.defend(me, mate, enemies, occupied, ball, carrier), "defense"

        if action is None:
            return fallback
        if self.trace is not None and not np.array_equal(action, fallback):
            self.trace[kind] += 1
        return action


def build_team(side, ids, env, genome, trace=None):
    opening = genome[10]
    attack = ATTACK if opening == "current" else (*ATTACK[:3], (
        "current" if opening in ("carry", "script") else opening
    ))
    opening_trace = trace[0] if trace is not None and opening != "current" else None
    agents = team2_explore.build_team(side, ids, env, attack, opening_trace)
    if genome != DEFAULT:
        for i, agent in enumerate(ids):
            agents[agent] = TeamPlayer(agents[agent], genome, None if trace is None else trace[i])
    return agents


def make_team(side, ids, env):
    genome = DEFAULT
    if MODEL.is_file():
        genome = tuple(json.loads(MODEL.read_text(encoding="utf-8"))["genome"])
        if len(genome) != len(OPTIONS) or any(
            value not in values for value, values in zip(genome, OPTIONS)
        ):
            raise ValueError(f"Invalid team policy: {MODEL}")
    return build_team(side, ids, env, genome)