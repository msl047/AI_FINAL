"""grid2v2/agents.py — the Agent interface and two baselines.

An Agent sees only its observation (in its own frame: attack towards +x) and returns an action in its adapter's space.
`ScriptedAgent` is the baseline every team must beat.  It uses `physics.simulate` to look ahead at its own kicks,
which is legal for you too: the physics is public and deterministic (the slip is in the movement, not the ball).
"""
from __future__ import annotations
import math
import numpy as np
from .physics import Rink, simulate, predict
from .actions import Adapter, STAY, N, E, S, W, MOVE_DELTA


class Agent:
    def __init__(self, adapter: Adapter, rink: Rink, index: int = 0, rng=None, crease: int = 2, min_pass: int = 2):
        self.adapter, self.rink, self.index, self.crease, self.min_pass = adapter, rink, index, crease, min_pass
        self.rng = rng or np.random.default_rng()

    def forbidden(self, cell) -> bool:
        """The crease of the goal we attack (our frame: the right-hand goal)."""
        return cell[1] in self.rink.goal_rows and cell[0] >= self.rink.width - self.crease

    def reset(self) -> None: pass
    def act(self, obs: dict): raise NotImplementedError


class RandomAgent(Agent):
    def reset(self): self.adapter.space.seed(int(self.rng.integers(2**31)))
    def act(self, obs): return self.adapter.space.sample()


def manhattan(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])


class ScriptedAgent(Agent):
    """Two fixed roles.  Field player (robot 0): shoot when a shot scores; pass when pressed or the teammate is well
    ahead; otherwise dribble at the goal; off the ball, chase.  Keeper (robot 1 in 2v2): hold the goal line on the row
    the ball will arrive at, tackle a carrier next door, pick up a loose ball nearby, and clear or pass when holding it."""

    PRESSURE = 2          # an opponent within this many cells counts as pressure
    AHEAD = 3             # pass forward when the teammate is this far ahead
    SHOT_SPEEDS = (3.0, 2.0)
    SHOT_STEPS = 2        # only shots that cross the line within this many steps: the ball moves before the keeper does
    PASS_SPEEDS = (2.0, 3.0)

    # ---------------------------------------------------------------------------------------------- bookkeeping
    def _parse(self, obs):
        me = (int(obs["self"][0]), int(obs["self"][1])); i_have = obs["self"][2] > 0.5
        mate = None if obs["mate"][0] < 0 else (int(obs["mate"][0]), int(obs["mate"][1])); mate_has = obs["mate"][2] > 0.5
        opps = [((int(r[0]), int(r[1])), r[2] > 0.5) for r in obs["opponents"] if r[0] >= 0]
        self.i_am_stunned = obs["self"][3] > 0.5
        occupied = {c: f"opp{k}" for k, (c, _) in enumerate(opps)}
        if mate is not None: occupied[mate] = "mate"
        ball = tuple(float(v) for v in obs["ball"])
        return me, i_have, mate, mate_has, opps, occupied, ball

    def _free(self, cell, occupied, me):
        return self.rink.inside(*cell) and cell not in occupied

    def _toward(self, me, target, occupied, allow=()):
        """First step of a shortest path (BFS) from `me` to `target` around occupied cells and our crease; a cell in
        `allow` may be entered (used to walk into a ball carrier for a tackle).  STAY when already there or walled in."""
        if me == target: return STAY
        from collections import deque
        blocked = {c for c in occupied if c not in allow}
        prev = {me: None}; q = deque([me]); found = None
        while q:
            c = q.popleft()
            if c == target: found = c; break
            for m in (E, W, N, S):
                dx, dy = MOVE_DELTA[m]; n = (c[0] + dx, c[1] + dy)
                if n in prev or not self.rink.inside(*n) or self.forbidden(n) or (n in blocked and n != target): continue
                prev[n] = (c, m); q.append(n)
        if found is None:                                        # unreachable: fall back to the greedy step
            best, best_d = STAY, manhattan(me, target)
            for m in (E, W, N, S):
                dx, dy = MOVE_DELTA[m]; c = (me[0] + dx, me[1] + dy)
                if not self.rink.inside(*c) or self.forbidden(c) or c in blocked: continue
                if manhattan(c, target) < best_d: best, best_d = m, manhattan(c, target)
            return best
        c = found
        while prev[c][0] != me: c = prev[c][0]
        return prev[c][1]

    def _goal_targets(self):
        gx = self.rink.width - 0.5; gy = (self.rink.height - 1) / 2
        return [(gx, gy), (gx, gy + 0.9), (gx, gy - 0.9)]

    def _kick_candidates(self, me, mate):
        """(angle, speed) pairs this tier can produce, aimed at the goal and at the teammate."""
        if self.adapter.name == "discrete":
            s = self.adapter.kick_speed
            return [(math.pi / 2, s), (0.0, s), (-math.pi / 2, s), (math.pi, s)], [(math.pi / 2, s), (0.0, s), (-math.pi / 2, s), (math.pi, s)]
        shots = [(math.atan2(ty - me[1], tx - me[0]), s) for tx, ty in self._goal_targets() for s in self.SHOT_SPEEDS]
        passes = [] if mate is None else [(math.atan2(mate[1] - me[1], mate[0] - me[0]), s) for s in self.PASS_SPEEDS]
        if self.adapter.name == "fine":          # snap to the adapter's lattice
            shots = [self.adapter.decode(self.adapter.encode_kick(a, s), True).kick for a, s in shots]
            passes = [self.adapter.decode(self.adapter.encode_kick(a, s), True).kick for a, s in passes]
        return shots, passes

    # ------------------------------------------------------------------------------------------------- policies
    def act(self, obs):
        me, i_have, mate, mate_has, opps, occupied, ball = self._parse(obs)
        keeper = mate is not None and self.index == 1            # in 2v2, robot 1 keeps goal; in 1v1 robot 0 does everything
        if i_have:
            return self._clear(me, mate, opps, occupied) if keeper else self._with_ball(me, mate, opps, occupied)
        opp_holder = next((c for c, h in opps if h), None)
        if keeper:
            return self.adapter.encode_move(self._keeper(me, occupied, ball, opp_holder))
        if mate_has:
            return self._support(me, mate, opps, occupied)
        if opp_holder is not None:
            return self.adapter.encode_move(self._toward(me, opp_holder, occupied, allow=(opp_holder,)))
        return self._loose(me, mate, opps, occupied, ball)

    def _kick_occ(self, me, occupied):
        """Occupancy as the ball sees it on the step of a kick: a teammate within min_pass - 1 cells cannot receive."""
        return {c: a for c, a in occupied.items() if not (a == "mate" and manhattan(c, me) < self.min_pass)}

    def _with_ball(self, me, mate, opps, occupied):
        shots, passes = self._kick_candidates(me, mate)
        occupied = self._kick_occ(me, occupied)
        for angle, speed in shots:
            ev, _, k, _ = simulate(self.rink, me[0], me[1], angle, speed, occupied)
            if ev == ("goal", "right") and k <= self.SHOT_STEPS:      # crosses before the keeper can react to it
                return self.adapter.encode_kick(angle, speed)
        pressed = any(manhattan(me, c) <= self.PRESSURE for c, _ in opps)
        if mate is not None and (pressed or mate[0] >= me[0] + self.AHEAD):
            for angle, speed in passes:
                ev = simulate(self.rink, me[0], me[1], angle, speed, occupied)[0]
                if ev == ("taken", "mate") and mate[0] >= me[0] - 1:
                    return self.adapter.encode_kick(angle, speed)
        # dribble: towards the nearest goal row, then forward; keep out of cells next to opponents when possible
        gy = min(self.rink.goal_rows, key=lambda r: abs(r - me[1]))
        target = (self.rink.width - 1 - self.crease, gy)
        danger = {(c[0] + dx, c[1] + dy) for c, _ in opps for dx, dy in MOVE_DELTA.values()}
        m = self._toward(me, target, occupied | {d: "danger" for d in danger if d not in occupied})
        if m == STAY:
            m = self._toward(me, target, occupied)
        if m == STAY:                                       # at the shooting spot but the lane is blocked: change row
            for row in sorted(self.rink.goal_rows, key=lambda r: abs(r - me[1])):
                if row != me[1]:
                    m = self._toward(me, (me[0], row), occupied)
                    if m != STAY: break
        if m == STAY and me[0] < self.rink.width - 1 - self.crease:   # boxed in: shoot anyway if it at least goes forward
            for angle, speed in shots:
                if abs(angle) < 1e-6:
                    return self.adapter.encode_kick(angle, speed)
        return self.adapter.encode_move(m)

    def _clear(self, me, mate, opps, occupied):
        """The keeper with the ball: pass to the field player if the lane is clear, else the forward kick that no
        opponent can cut out (dry-run every candidate), else the one that travels farthest from our goal."""
        shots, passes = self._kick_candidates(me, mate)
        occupied = self._kick_occ(me, occupied)
        for angle, speed in passes:
            if simulate(self.rink, me[0], me[1], angle, speed, occupied)[0] == ("taken", "mate"):
                return self.adapter.encode_kick(angle, speed)
        if self.adapter.name == "discrete":
            cands = [(a, s) for a, s in shots if math.cos(a) > -0.1]
        else:
            cands = [(math.radians(d), s) for d in (0, 30, -30, 55, -55) for s in (3.0, 2.0)]
        best, best_x = None, -1
        for angle, speed in cands:
            ev, b, *_ = simulate(self.rink, me[0], me[1], angle, speed, occupied)
            if ev is not None and ev[0] == "taken" and ev[1].startswith("opp"):
                continue
            if ev is not None and ev[0] == "goal": return self.adapter.encode_kick(angle, speed)
            if b.x > best_x: best, best_x = (angle, speed), b.x
        if best is None:
            best = max(cands, key=lambda k: (math.cos(k[0]), k[1]))
        return self.adapter.encode_kick(*best)

    def _support(self, me, mate, opps, occupied):
        rows = list(self.rink.goal_rows)
        def crowd(r): return min([manhattan((mate[0] + 4, r), c) for c, _ in opps] or [99])
        row = max(rows, key=crowd)
        target = (min(mate[0] + 4, self.rink.width - 1 - self.crease), row)
        if manhattan(me, target) <= 1: return self.adapter.encode_move(STAY)
        return self.adapter.encode_move(self._toward(me, target, occupied))

    def _loose(self, me, mate, opps, occupied, ball):
        bx, by, vx, vy = ball
        rest = (int(round(bx)), int(round(by)))
        speed = math.hypot(vx, vy)
        if speed > 1e-6:                                             # where will it stop?
            ev, b, *_ = predict(self.rink, bx, by, vx, vy, occupied | {me: "me"})
            rest = (b.cx, b.cy)
        return self.adapter.encode_move(self._toward(me, rest, occupied))

    def _keeper(self, me, occupied, ball, opp_holder):
        """Stay on the goal line, on the row the ball will arrive at; step out only for a loose ball or carrier next door."""
        bx, by, vx, vy = ball
        rows = self.rink.goal_rows
        if opp_holder is not None and manhattan(me, opp_holder) == 1:
            return self._toward(me, opp_holder, occupied, allow=(opp_holder,))             # tackle
        if vx < -1e-6:                                               # coming our way: meet it on our line
            _, _, _, pts = predict(self.rink, bx, by, vx, vy, {})
            y = min(pts, key=lambda p: p[0])[1]
        elif opp_holder is None and math.hypot(vx, vy) < 1e-6 and manhattan(me, (round(bx), round(by))) <= 2 and bx <= 3:
            return self._toward(me, (int(round(bx)), int(round(by))), occupied)             # a loose ball nearby: take it
        elif bx > self.rink.width / 2:
            y = (self.rink.height - 1) / 2                             # far away: hold the centre
        else:
            y = by
        row = int(min(max(round(y), rows.start), rows.stop - 1))
        if me[0] == 0 and me[1] == row: return STAY
        if me[0] == 0 or me[1] != row:                               # fix the row first, then the column
            m = self._toward(me, (me[0], row), occupied)
            if m != STAY: return m
        return self._toward(me, (0, row), occupied)
