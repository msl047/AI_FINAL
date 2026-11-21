"""grid2v2/env.py — the rules.  A 2v2 (or 1v1) ball game on a walled grid, as a PettingZoo ParallelEnv.

Red attacks to the right (+x), blue to the left.  Every agent observes the same things, in its own frame: its team
always attacks towards +x, so one policy can play either colour.  Actions are decoded by a per-agent `Adapter`
(see actions.py), so teams with different action spaces play each other on the same rink.

Step order (all agents act simultaneously):
  1. decode intents;  2. kicks release the ball (the kicker does not move this step);  3. moves, with the slip, resolved
  so that no two robots share a cell (random order, swaps and head-ons blocked); a blocked mover whose target holds
  the ball tackles with probability p_tackle (with `block=True`, one whose target *kicked this step* blocks the kick
  with p_block; off by default); a failed tackle stuns the tackler for `stun` steps (it cannot move); a ball carrier
  only moves with
  probability p_dribble, so dribbling averages half a cell per step while a pass covers 2 to 4; nobody may enter the
  crease (the `crease` columns in front of the goal mouth) of the goal they attack, so every goal is a shot from range;  4. a blocked mover whose target cell
  holds an opponent carrying the ball tackles with probability p_tackle;  5. a loose ball in a robot's cell is taken,
  otherwise it flies (physics.step_ball): bounces, is intercepted by the first robot in its path, or scores; the kicker
  itself cannot take its own kick back while it is still moving (no stalling by rebounding the ball off a wall), it can
  pick it up once it has stopped; a teammate standing next to the kicker cannot receive the kick (`min_pass`), so a
  pass always crosses at least one open cell where a defender can stand;
  6. after a goal, both teams return to their kick-off cells and the conceding team's robot 0 holds the ball;  7. clock.

Rewards: +1 to each robot of the scoring team, -1 to each robot of the other, nothing else.
"""
from __future__ import annotations
import math
import numpy as np
from gymnasium import spaces
try:
    from pettingzoo import ParallelEnv
except ImportError:                       # HW4 uses the single-agent views only; the project installs pettingzoo
    class ParallelEnv:  # minimal stand-in so the module imports without pettingzoo
        metadata = {}

from .physics import Rink, Ball, step_ball, EPS
from .actions import Intent, make_adapter, MOVE_DELTA, STAY, N, E, S, W

TEAMS = ("red", "blue")
ATTACK = {"red": "right", "blue": "left"}
P_MOVE = 0.8
P_TACKLE = 0.5
P_DRIBBLE = 0.5      # a ball carrier's move only happens with this probability: dribbling is slow, passing is fast
CREASE = 2           # the attacking team may not enter the `crease` columns in front of the goal mouth it attacks
STUN = 1             # a failed tackle or block leaves the tackler unable to move for this many steps
BLOCK = False        # moving into an opponent as he kicks blocks the kick with p_block (off: it stopped learners from scoring)
MIN_PASS = 2         # a teammate standing next to the kicker cannot receive the kick: a pass needs room (kills the adjacent ping-pong stall)
P_BLOCK = 0.5        # probability that a block attempt succeeds
MAX_STEPS = 600
PERP = {N: (E, W), S: (E, W), E: (N, S), W: (N, S)}


class Grid2v2(ParallelEnv):
    metadata = {"name": "grid2v2_v0", "render_modes": ["ansi", "rgb_array"], "is_parallelizable": True}

    def __init__(self, width: int = 21, height: int = 11, players_per_team: int = 2, tiers="discrete",
                 p_move: float = P_MOVE, p_tackle: float = P_TACKLE, p_dribble: float = P_DRIBBLE, crease: int = CREASE, stun: int = STUN, block: bool = BLOCK, p_block: float = P_BLOCK, min_pass: int = MIN_PASS, max_steps: int = MAX_STEPS,
                 goals_to_win: int | None = None, kick_noise: float = 0.0, ball_first: bool = True, render_mode=None):
        assert width % 2 == 1 and height % 2 == 1, "odd width and height keep the centre cell and the goals centred"
        self.rink = Rink(width, height)
        self.n = players_per_team
        self.p_move, self.p_tackle, self.p_dribble, self.crease, self.stun, self.block, self.p_block = p_move, p_tackle, p_dribble, crease, stun, block, p_block
        self.min_pass = min_pass
        self.max_steps, self.goals_to_win, self.kick_noise, self.ball_first = max_steps, goals_to_win, kick_noise, ball_first
        self.render_mode = render_mode
        self.possible_agents = [f"{t}_{i}" for t in TEAMS for i in range(self.n)]
        self.team = {a: a.split("_")[0] for a in self.possible_agents}
        self.index = {a: int(a.split("_")[1]) for a in self.possible_agents}
        self.members = {t: [a for a in self.possible_agents if self.team[a] == t] for t in TEAMS}
        self.adapters = self._make_adapters(tiers)
        W, H = width, height
        self._obs_space = spaces.Dict({
            "self": spaces.Box(np.array([0, 0, 0, 0], np.float32), np.array([W - 1, H - 1, 1, 9], np.float32), dtype=np.float32),
            "mate": spaces.Box(np.array([-1, -1, -1, -1], np.float32), np.array([W - 1, H - 1, 1, 9], np.float32), dtype=np.float32),
            "opponents": spaces.Box(np.full((self.n, 4), -1, np.float32), np.tile(np.array([W - 1, H - 1, 1, 9], np.float32), (self.n, 1)), dtype=np.float32),
            "ball": spaces.Box(np.array([-0.5, -0.5, -4, -4], np.float32), np.array([W - 0.5, H - 0.5, 4, 4], np.float32), dtype=np.float32),
            "score": spaces.Box(0, 99, shape=(2,), dtype=np.float32),
            "t": spaces.Box(0, max_steps, shape=(1,), dtype=np.float32),
        })
        self.rng = np.random.default_rng()
        self.agents = []

    # ----------------------------------------------------------------------------------------------- spaces & setup
    def _make_adapters(self, tiers):
        if isinstance(tiers, dict):
            return {a: make_adapter(tiers.get(a, tiers.get(self.team[a], "discrete"))) for a in self.possible_agents}
        return {a: make_adapter(tiers) for a in self.possible_agents}

    def observation_space(self, agent): return self._obs_space
    def action_space(self, agent): return self.adapters[agent].space
    def tier(self, agent): return self.adapters[agent].name

    def kickoff_cells(self) -> dict:
        W, H = self.rink.width, self.rink.height
        cy = (H - 1) // 2
        cells = {}
        for i in range(self.n):
            x = W // 2 - 2 if i == 0 else 2 * i          # robot 0 contests the centre, the others start near their own goal
            cells[f"red_{i}"] = (x, cy)
            cells[f"blue_{i}"] = (W - 1 - x, cy)
        return cells

    def _kickoff(self, ball_to: str | None):
        self.pos = dict(self.kickoff_cells())
        c = ((self.rink.width - 1) // 2, (self.rink.height - 1) // 2)
        if ball_to is None:
            self.ball = Ball.at_cell(*c); self.holder = None
        else:
            self.holder = f"{ball_to}_0"
            self.ball = Ball.at_cell(*self.pos[self.holder])
        self.events.append(("kickoff", ball_to))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.score = {"red": 0, "blue": 0}
        self.t = 0
        self.events = []
        self.last_kicker = None
        self.stunned = {a: 0 for a in self.agents}
        self._kickoff(None)
        obs = {a: self.observe(a) for a in self.agents}
        infos = {a: {"events": list(self.events)} for a in self.agents}
        return obs, infos

    # ------------------------------------------------------------------------------------------------- observation
    def _row(self, agent, mirror: bool) -> np.ndarray:
        x, y = self.pos[agent]
        if mirror: x = self.rink.width - 1 - x
        return np.array([x, y, 1.0 if self.holder == agent else 0.0, self.stunned.get(agent, 0)], np.float32)

    def observe(self, agent) -> dict:
        team = self.team[agent]; mirror = team == "blue"
        mates = [a for a in self.members[team] if a != agent]
        opps = self.members["blue" if team == "red" else "red"]
        bx, by = self.ball.x, self.ball.y
        vx, vy = self.ball.dx * self.ball.speed, self.ball.dy * self.ball.speed
        if mirror:
            bx = self.rink.width - 1 - bx; vx = -vx
        return {
            "self": self._row(agent, mirror),
            "mate": self._row(mates[0], mirror) if mates else np.full(4, -1, np.float32),
            "opponents": np.stack([self._row(o, mirror) for o in opps]).astype(np.float32),
            "ball": np.array([bx, by, vx, vy], np.float32),
            "score": np.array([self.score[team], self.score["blue" if team == "red" else "red"]], np.float32),
            "t": np.array([self.t], np.float32),
        }

    def state(self) -> dict:
        """True-frame snapshot for logs and the renderer."""
        return {"t": self.t, "pos": {a: list(p) for a, p in self.pos.items()}, "holder": self.holder, "stunned": {a: v for a, v in self.stunned.items() if v},
                "ball": [self.ball.x, self.ball.y, self.ball.dx * self.ball.speed, self.ball.dy * self.ball.speed],
                "score": dict(self.score), "events": list(self.events)}

    def occupied(self) -> dict:
        return {p: a for a, p in self.pos.items()}

    def in_attacking_crease(self, team: str, cell) -> bool:
        """True when `cell` is inside the crease of the goal that `team` attacks (forbidden ground for that team)."""
        x, y = cell
        if y not in self.rink.goal_rows: return False
        return x >= self.rink.width - self.crease if team == "red" else x < self.crease

    # ------------------------------------------------------------------------------------------------------ step
    def _slip(self, move: int) -> int:
        if move == STAY: return STAY
        r = self.rng.random()
        if r < self.p_move: return move
        return PERP[move][0 if r < self.p_move + (1 - self.p_move) / 2 else 1]

    def _occ_for_ball(self, just_kicked: bool = False) -> dict:
        """Cells that can take the ball: every robot, except the kicker while its own kick is still moving, and, on
        the step of the kick, the kicker's teammates standing within min_pass - 1 cells of him (a pass needs room)."""
        occ = self.occupied()
        if self.ball.moving and self.last_kicker in occ.values():      # no self-pass off the wall
            occ = {c: a for c, a in occ.items() if a != self.last_kicker}
        if just_kicked and self.min_pass > 1 and self.last_kicker is not None:
            kx, ky = self.pos[self.last_kicker]; team = self.team[self.last_kicker]
            occ = {c: a for c, a in occ.items() if not (self.team[a] == team and abs(c[0] - kx) + abs(c[1] - ky) < self.min_pass)}
        return occ

    def _fly(self, rewards, just_kicked: bool = False) -> bool:
        """One step of free flight.  Returns True on a goal (positions are then reset for the kick-off)."""
        f = step_ball(self.rink, self.ball, self._occ_for_ball(just_kicked))
        if f.bounces: self.events.append(("bounce", f.bounces))
        if f.event is not None and f.event[0] == "taken":
            self.holder = f.event[1]; self.ball = Ball.at_cell(*self.pos[self.holder]); self.events.append(("intercept", self.holder))
        elif f.event is not None and f.event[0] == "goal":
            scorer_team = "red" if f.event[1] == "right" else "blue"
            self.score[scorer_team] += 1
            for a in self.agents:
                rewards[a] = 1.0 if self.team[a] == scorer_team else -1.0
            self.events.append(("goal", scorer_team, self.last_kicker))
            self._kickoff("blue" if scorer_team == "red" else "red")
            return True
        return False

    def _move(self, intents, kickers, holder_at_start=None):
        targets = {}
        for a, it in intents.items():
            if a in kickers or it.move == STAY or self.stunned.get(a, 0) > 0: continue
            if a == self.holder and self.rng.random() >= self.p_dribble: continue      # dribbling is slow
            dx, dy = MOVE_DELTA[self._slip(it.move)]
            tgt = (self.pos[a][0] + dx, self.pos[a][1] + dy)
            if self.rink.inside(*tgt) and not self.in_attacking_crease(self.team[a], tgt): targets[a] = tgt
        pending = list(targets)
        changed = True
        while changed and pending:
            changed = False
            taken = set(self.pos.values())
            for a in self.rng.permutation(pending):
                a = str(a)
                if targets[a] not in taken:
                    taken.discard(self.pos[a]); taken.add(targets[a]); self.pos[a] = targets[a]
                    pending.remove(a); changed = True
        occ = self.occupied()
        for a in self.rng.permutation(pending):
            a = str(a)
            victim = occ.get(targets[a])
            if victim is None or self.team[victim] == self.team[a]: continue
            if victim == self.holder:
                if self.rng.random() < self.p_tackle:
                    self.holder = a; self.ball = Ball.at_cell(*self.pos[a])
                    self.events.append(("tackle", a, victim))
                else:
                    self.stunned[a] = self.stun; self.events.append(("tackle_failed", a, victim))
            elif self.block and victim == holder_at_start and victim in kickers and (self.holder is None or self.team[self.holder] == self.team[victim]):
                # the defender reached the kicker as the kick went off: with p_tackle the kick is blocked and the
                # defender comes away with the ball (so a passer under pressure is not safe just because he passed)
                if self.rng.random() < self.p_block:
                    self.holder = a; self.ball = Ball.at_cell(*self.pos[a])
                    self.events.append(("block", a, victim))
                else:
                    self.stunned[a] = self.stun; self.events.append(("tackle_failed", a, victim))

    def step(self, actions: dict):
        self.events = []
        rewards = {a: 0.0 for a in self.agents}
        # 1. intents (in the true frame)
        intents = {}
        for a in self.agents:
            act = actions.get(a, self.adapters[a].noop()) if actions is not None else self.adapters[a].noop()
            it = self.adapters[a].decode(act, self.holder == a)
            intents[a] = it.mirrored() if self.team[a] == "blue" else it
        # 2. kicks
        holder_at_start = self.holder
        before = dict(self.stunned)                 # stuns set during this step start counting down next step
        kickers = set()
        for a, it in intents.items():
            if it.kick is not None and self.holder == a:
                angle, speed = it.kick
                if self.kick_noise > 0:
                    angle += self.rng.normal(0.0, self.kick_noise * speed)
                self.ball = Ball.at_cell(*self.pos[a]); self.ball.kick(angle, speed)
                self.holder = None; self.last_kicker = a; kickers.add(a)
                self.events.append(("kick", a, round(angle, 4), round(speed, 3)))
        goal = False
        if self.ball_first and self.holder is None and self.ball.moving:
            goal = self._fly(rewards, just_kicked=bool(kickers))   # 3. the ball moves before anyone can react to it
        if not goal:
            self._move(intents, kickers, holder_at_start)       # 4. moves, with the slip; 5. tackles and blocks by blocked movers
            if self.holder is not None:
                self.ball = Ball.at_cell(*self.pos[self.holder])
            else:                                               # 6. a loose ball in a robot's cell is taken
                occ = self._occ_for_ball()
                who = None if (kickers and not self.ball_first) else occ.get((self.ball.cx, self.ball.cy))
                if who is not None:
                    self.holder = who; moving = self.ball.moving; self.ball = Ball.at_cell(*self.pos[who])
                    self.events.append(("intercept" if moving else "pickup", who))
                elif self.ball.moving and not self.ball_first:
                    self._fly(rewards, just_kicked=bool(kickers))
        # 7. clock (a stun set this step lasts through the next `stun` steps)
        for a in self.stunned:
            if self.stunned[a] > 0 and self.stunned[a] == before.get(a, 0): self.stunned[a] -= 1
        self.t += 1
        done = self.goals_to_win is not None and max(self.score.values()) >= self.goals_to_win
        trunc = self.t >= self.max_steps
        obs = {a: self.observe(a) for a in self.agents}
        terminations = {a: done for a in self.agents}
        truncations = {a: (trunc and not done) for a in self.agents}
        infos = {a: {"events": list(self.events), "intent": intents[a]} for a in self.agents}
        if done or trunc:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    # ---------------------------------------------------------------------------------------------------- render
    def render(self):
        if self.render_mode == "rgb_array":                     # the broadcast view of the current state (slow; not for training)
            from .render import Renderer
            if not hasattr(self, "_renderer"):
                self._renderer = Renderer(self.rink, self.crease); self._facing = {}; self._prev = dict(self.pos)
            for a, p in self.pos.items():
                q = self._prev.get(a, p)
                if p != q and abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1: self._facing[a] = (p[0] - q[0], p[1] - q[1])
            self._prev = dict(self.pos)
            im = self._renderer.frame(self.pos, self.holder, (self.ball.x, self.ball.y), self._facing, self.score, self.t, self.max_steps,
                                      stunned={a for a, v in self.stunned.items() if v})
            return np.asarray(im)
        W, H = self.rink.width, self.rink.height
        grid = [["." for _ in range(W)] for _ in range(H)]
        for a, (x, y) in self.pos.items():
            grid[y][x] = ("R" if self.team[a] == "red" else "B") if self.holder != a else ("r" if self.team[a] == "red" else "b")
        if self.holder is None:
            grid[self.ball.cy][self.ball.cx] = "o"
        rows = []
        for y in range(H - 1, -1, -1):
            edge = "|" if y in self.rink.goal_rows else "#"
            rows.append(edge + "".join(grid[y]) + edge)
        top = "#" * (W + 2)
        s = f"t={self.t}  red {self.score['red']} : {self.score['blue']} blue   holder={self.holder}\n" + top + "\n" + "\n".join(rows) + "\n" + top
        if self.render_mode == "ansi":
            return s
        print(s)

    def close(self): pass
