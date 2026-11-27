"""Use the current teams as bases and add situational tactics."""

import json
import math
from pathlib import Path

import numpy as np

from grid2v2 import E, N, S, STAY, W
from grid2v2.actions import MOVE_DELTA
from grid2v2.agents import ScriptedAgent, manhattan
from grid2v2.physics import predict, simulate
from teams import team2_press as press
from teams import team2_teamplay as teamplay

MODEL = Path(__file__).with_name("policy.json")
SAVED = json.loads(MODEL.read_text(encoding="utf-8")) if MODEL.is_file() else None
TIER = "fine" if SAVED and SAVED["genome"][0] == "fine" else "box"

NAMES = (
    "base", "loose", "race_limit", "cover_depth", "defense",
    "danger_x", "handoff", "counter", "counter_window", "counter_margin",
)
OPTIONS = (
    ("box", "fine", "teamplay", "press"),
    ("base", "race", "intercept"), (3, 4, 5, 6, 8, 10), (1, 2, 3, 4, 5, 6),
    ("base", "screen", "next_dribble", "next_receiver"), (6, 7, 8, 9, 10),
    (0, 1, 2, 3, 4), ("base", "give_go", "space"), (2, 3, 4, 6, 8), (0, 1, 2, 3),
)
DEFAULT = ("box", "base", 6, 4, "base", 8, 2, "base", 4, 1)
MOVES = (STAY, E, N, S, W)
FAMILIES = {"loose": 1, "defense": 4, "counter": 7}


def normalize(genome):
    g = list(genome)
    if g[1] == "base":
        g[2] = DEFAULT[2]
    if g[1] == g[4] == "base":
        g[3] = DEFAULT[3]
    if g[4] == "base":
        g[5:7] = DEFAULT[5:7]
    if g[7] == "base":
        g[8:] = DEFAULT[8:]
    return tuple(g)


class LeaguePlayer:
    def __init__(self, original, index, genome, trace):
        self.original = original
        self.index, self.adapter = index, original.adapter
        self.rink, self.genome, self.trace = original.rink, genome, trace
        self.script = ScriptedAgent(self.adapter, self.rink, index=self.index)
        pass_genes = ("lead", 2, 5, max(1, genome[9]), *teamplay.DEFAULT[4:])
        self.passer = teamplay.TeamPlayer(self.script, pass_genes, None)

    def reset(self):
        self.original.reset()
        self.last_owner, self.regain, self.their_score = None, -100, 0

    def legal(self, cell, occupied):
        return self.rink.inside(*cell) and cell not in occupied and not self.script.forbidden(cell)

    def neighbors(self, me, occupied):
        for move in MOVES:
            dx, dy = MOVE_DELTA[move]
            cell = (me[0] + dx, me[1] + dy)
            if self.legal(cell, occupied):
                yield move, cell

    def go(self, me, target, occupied, allow=()):
        return self.adapter.encode_move(self.script._toward(me, target, occupied, allow=allow))

    def screen(self, threat):
        x = min(self.genome[3], max(0, threat[0] - 2))
        middle = self.rink.height // 2
        y = round(middle + (threat[1] - middle) * (x + 0.5) / (threat[0] + 0.5))
        return x, y

    def recover_ball(self, me, mate, enemies, occupied, ball, stunned):
        flying = math.hypot(*ball[2:]) > 1e-6
        if flying:
            event, first, *_ = predict(self.rink, *ball, {}, max_steps=1)
            _, end, *_ = predict(self.rink, *ball, {})
            landing = (end.cx, end.cy)
        else:
            landing = (round(ball[0]), round(ball[1]))
        if min(manhattan(me, landing), manhattan(mate, landing)) > self.genome[2]:
            return None

        def approach(pos, other, is_stunned):
            best = (4, manhattan(pos, landing) + 20 * is_stunned, None)
            if not flying or self.genome[1] != "intercept" or is_stunned:
                return best
            for move, cell in self.neighbors(pos, set(enemies) | {other}):
                hit, _, turn, _ = predict(self.rink, *ball, {cell: True}, max_steps=3)
                can_enter = event is None and (first.cx, first.cy) == cell
                can_intercept = hit == ("taken", True) and (move == STAY or turn > 1)
                if can_enter or can_intercept:
                    rank = (1 if can_enter else turn, int(move != STAY), move)
                    if rank[:2] < best[:2]:
                        best = rank
            return best

        mine = approach(me, mate, False)
        theirs = approach(mate, me, stunned)
        if (*mine[:2], self.index) <= (*theirs[:2], 1 - self.index):
            return self.go(me, landing, occupied) if mine[2] is None else self.adapter.encode_move(mine[2])
        if self.last_owner == "ours":
            return None
        target = self.screen(landing)
        return self.go(me, target, occupied) if self.legal(target, occupied) else None

    def defend(self, me, mate, carrier, receiver, occupied, stunned):
        mode, danger, handoff = self.genome[4:7]
        if carrier[0] > danger:
            return None
        deep = min(me[0], mate[0])

        def pressure_cost(pos, is_stunned):
            opens_goal = pos[0] == deep and pos[0] < carrier[0]
            return manhattan(pos, carrier) + 20 * is_stunned + handoff * opens_goal

        if (pressure_cost(me, False), self.index) <= (pressure_cost(mate, stunned), 1 - self.index):
            return self.go(me, carrier, occupied, allow=(carrier,))

        desired = self.screen(carrier)
        if mode == "screen":
            return self.go(me, desired, occupied) if self.legal(desired, occupied) else None

        origins = [(carrier, 4)]
        if mode == "next_dribble":
            origins += [
                (cell, 1) for move, cell in self.neighbors(carrier, set(occupied) | {me})
                if move != STAY and not (
                    cell[0] < self.script.crease and cell[1] in self.rink.goal_rows
                )
            ]
        if mode == "next_receiver" and receiver is not None:
            distance = manhattan(carrier, receiver)
            if 2 <= distance <= 4:
                angle = math.atan2(receiver[1] - carrier[1], receiver[0] - carrier[0])
                blockers = occupied | {me: "defender", receiver: "receiver"}
                event, *_ = simulate(self.rink, *carrier, angle, 3.0, blockers, max_steps=1)
                if event == ("taken", "receiver"):
                    origins.append((receiver, 3))

        shots = []
        for origin, weight in origins:
            if origin[0] > 5:
                continue
            for f in (0.15, 0.50, 0.85):
                y = self.rink.goal_lo + f * (self.rink.goal_hi - self.rink.goal_lo)
                angle = math.atan2(y - origin[1], -0.5 - origin[0])
                if simulate(self.rink, *origin, angle, 3.0, {})[0] == ("goal", "left"):
                    shots.append((origin, angle, weight))

        if shots:
            def score(cell):
                blocked = sum(
                    weight for origin, angle, weight in shots
                    if origin != cell and simulate(
                        self.rink, *origin, angle, 3.0, {cell: True}
                    )[0] == ("taken", True)
                )
                return blocked, -manhattan(cell, desired)

            choices = list(self.neighbors(me, occupied))
            if choices:
                move, cell = max(choices, key=lambda item: score(item[1]))
                if score(cell)[0]:
                    return self.adapter.encode_move(move)

        return self.go(me, desired, occupied) if self.legal(desired, occupied) else None

    def counter(self, obs, fallback, me, mate, have, enemies, occupied):
        mode, _, margin = self.genome[7:]
        carrier, receiver = (me, mate) if have else (mate, me)
        if have:
            kick = self.adapter.decode(fallback, True).kick
            if kick and simulate(
                self.rink, *me, *kick, self.script._kick_occ(me, occupied), max_steps=2
            )[0] == ("goal", "right"):
                return fallback

        if mode == "give_go":
            if have or obs["mate"][3] < 0.5:
                plan = self.passer.pass_plan(
                    carrier, receiver, enemies, have and obs["mate"][3] > 0.5,
                )
                if plan:
                    kick, move = plan
                    return kick if have else self.adapter.encode_move(move)
            if not have:
                candidates = [(mate[0] + 2, mate[1] + dy) for dy in (-1, 1)]
                valid = [c for c in candidates if self.legal(c, occupied) and
                         min(manhattan(c, enemy) for enemy in enemies) >= max(1, margin)]
                if valid:
                    return self.go(me, min(valid, key=lambda c: manhattan(me, c)), occupied)
            return None
        if not have:
            return None

        kick_occupied = self.script._kick_occ(me, occupied)
        best = None
        for angle_deg in (-60, -40, -20, 0, 20, 40, 60):
            angle = math.radians(angle_deg)
            for power in (2.0, 3.0):
                action = self.adapter.encode_kick(angle, power)
                kick = self.adapter.decode(action, True).kick
                event, end, turns, _ = simulate(self.rink, *me, *kick, kick_occupied)
                if event is not None:
                    continue
                target = (end.cx, end.cy)
                gain = target[0] - me[0]
                eta = min(max(turns, manhattan(mate, target)),
                          max(turns, 1 + manhattan(me, target)))
                edge = min(manhattan(enemy, target) for enemy in enemies) - eta
                if gain >= 2 and edge >= margin:
                    rank = (2 * gain + edge, edge)
                    if best is None or rank > best[0]:
                        best = (rank, action)
        return None if best is None else best[1]

    def act(self, obs):
        fallback = self.original.act(obs)
        me, have, mate, mate_has, opponents, occupied, ball = self.script._parse(obs)
        enemies = [p for p, _ in opponents]
        carrier = next((p for p, holds in opponents if holds), None)
        other = next((p for p in enemies if p != carrier), None)
        conceded = int(obs["score"][1]) > self.their_score
        self.their_score = int(obs["score"][1])
        turn = int(obs["t"][0])
        if have or mate_has:
            if self.last_owner == "theirs":
                self.regain = turn
            self.last_owner = "ours"
        elif carrier is not None:
            self.last_owner = "theirs"
        if conceded:
            self.regain = -100
        if self.script.i_am_stunned or mate is None or not enemies or conceded:
            return fallback

        stunned = obs["mate"][3] > 0.5
        action, kind = None, None
        if have or mate_has:
            if self.genome[7] != "base" and turn - self.regain <= self.genome[8]:
                action, kind = self.counter(obs, fallback, me, mate, have, enemies, occupied), "counter"
        elif carrier is not None:
            if self.genome[4] != "base":
                action = self.defend(me, mate, carrier, other, occupied, stunned)
                kind = "defense"
        elif self.genome[1] != "base":
            action = self.recover_ball(me, mate, enemies, occupied, ball, stunned)
            kind = "loose"

        if action is None:
            return fallback
        if self.trace is not None and not np.array_equal(action, fallback):
            self.trace[kind] += 1
        return action


def build_team(side, ids, env, genome, press_genome=None, trace=None):
    base = genome[0]
    if base in ("box", "fine"):
        players = {a: ScriptedAgent(env.adapters[a], env.rink, index=0) for a in ids}
    else:
        foundation = tuple(press_genome) if base == "press" and press_genome is not None else press.DEFAULT
        players = press.build_team(side, ids, env, foundation)
    if all(genome[i] == "base" for i in FAMILIES.values()):
        return players

    result = {}
    for index, a in enumerate(ids):
        stats = None if trace is None else trace[index]
        result[a] = LeaguePlayer(players[a], index, genome, stats)
    return result


def make_team(side, ids, env):
    if SAVED is None:
        return build_team(side, ids, env, DEFAULT)
    genome = tuple(SAVED["genome"])
    if len(genome) != len(OPTIONS) or any(g not in options for g, options in zip(genome, OPTIONS)):
        raise ValueError(f"Invalid genome in {MODEL}")
    return build_team(side, ids, env, genome, SAVED.get("press_genome"))