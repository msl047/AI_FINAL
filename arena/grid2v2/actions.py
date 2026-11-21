"""grid2v2/actions.py — action tiers.

Every action, whatever its space, decodes to an `Intent`: a move (STAY, N, E, S, W) or a kick (angle, speed).  The
environment only ever sees intents, so teams may choose different action spaces and still play each other.  Intents
are expressed in the agent's own frame (its team attacks towards +x); the environment flips them for the blue team.

Tier 1  DiscreteKick   Discrete(9): STAY, N, E, S, W, kick N, kick E, kick S, kick W at a fixed speed (3 -> range 6).
Tier 2  BoxKick        Box(4) in [-1, 1]: (mx, my, kx, ky).  If |k| > deadzone and the robot holds the ball, kick at
                       angle atan2(ky, kx) with speed 1 + 3 min(1, |k|) (range 1.6 to 10); otherwise move along the
                       dominant axis of (mx, my), or STAY when that too is inside the deadzone.
Tier 1.5 FineDiscrete  Discrete(5 + n_dirs * len(speeds)): a worked example of writing your own adapter.

Write your own by subclassing `Adapter`: `space`, `decode(action, has_ball) -> Intent`, `noop()`.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces

STAY, N, E, S, W = 0, 1, 2, 3, 4
MOVE_NAMES = ["STAY", "N", "E", "S", "W"]
MOVE_DELTA = {STAY: (0, 0), N: (0, 1), E: (1, 0), S: (0, -1), W: (-1, 0)}
AXIS_ANGLE = {N: math.pi / 2, E: 0.0, S: -math.pi / 2, W: math.pi}
KICK_SPEED = 3.0          # Tier 1 fixed kick speed: 3 + 2 + 1 = 6 cells
BOX_DEADZONE = 0.2
BOX_SPEED_MIN, BOX_SPEED_MAX = 1.0, 3.0


@dataclass(frozen=True)
class Intent:
    move: int = STAY                    # one of STAY..W; ignored when kick is set
    kick: tuple | None = None           # (angle, speed) or None

    def mirrored(self) -> "Intent":
        """The same intent seen from the other end of the rink (x -> -x)."""
        move = {E: W, W: E}.get(self.move, self.move)
        kick = None if self.kick is None else (math.atan2(math.sin(self.kick[0]), -math.cos(self.kick[0])), self.kick[1])
        return Intent(move, kick)


class Adapter:
    name = "base"
    space: spaces.Space

    def decode(self, action, has_ball: bool) -> Intent: raise NotImplementedError
    def noop(self): raise NotImplementedError
    def encode_move(self, move: int): raise NotImplementedError
    def encode_kick(self, angle: float, speed: float): raise NotImplementedError


class DiscreteKick(Adapter):
    name = "discrete"

    def __init__(self, kick_speed: float = KICK_SPEED):
        self.kick_speed = float(kick_speed)
        self.space = spaces.Discrete(9)

    def decode(self, action, has_ball: bool) -> Intent:
        a = int(action)
        if a < 5:
            return Intent(a)
        if not has_ball:
            return Intent(STAY)                          # kicking thin air: a wasted step
        d = a - 4
        return Intent(STAY, (AXIS_ANGLE[d], self.kick_speed))

    def noop(self): return STAY
    def encode_move(self, move: int): return int(move)

    def encode_kick(self, angle: float, speed: float):
        """Nearest axis; speed is fixed in this tier."""
        d = min(AXIS_ANGLE, key=lambda k: abs(math.atan2(math.sin(angle - AXIS_ANGLE[k]), math.cos(angle - AXIS_ANGLE[k]))))
        return 4 + d

    @property
    def kick_actions(self): return [5, 6, 7, 8]


class BoxKick(Adapter):
    name = "box"

    def __init__(self, deadzone: float = BOX_DEADZONE, speed_min: float = BOX_SPEED_MIN, speed_max: float = BOX_SPEED_MAX):
        self.deadzone, self.speed_min, self.speed_max = deadzone, speed_min, speed_max
        self.space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def speed_of(self, mag: float) -> float:
        return self.speed_min + (self.speed_max - self.speed_min) * min(1.0, mag)

    def decode(self, action, has_ball: bool) -> Intent:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(4), -1.0, 1.0)
        mx, my, kx, ky = a
        mag = math.hypot(kx, ky)
        if has_ball and mag > self.deadzone:
            return Intent(STAY, (math.atan2(ky, kx), self.speed_of(mag)))
        if max(abs(mx), abs(my)) <= self.deadzone:
            return Intent(STAY)
        if abs(mx) >= abs(my):
            return Intent(E if mx > 0 else W)
        return Intent(N if my > 0 else S)

    def noop(self): return np.zeros(4, dtype=np.float32)

    def encode_move(self, move: int):
        dx, dy = MOVE_DELTA[move]
        return np.array([dx, dy, 0.0, 0.0], dtype=np.float32)

    def encode_kick(self, angle: float, speed: float):
        mag = (min(max(speed, self.speed_min), self.speed_max) - self.speed_min) / (self.speed_max - self.speed_min)
        mag = max(mag, self.deadzone + 1e-3)
        return np.array([0.0, 0.0, mag * math.cos(angle), mag * math.sin(angle)], dtype=np.float32)


class FineDiscrete(Adapter):
    """Example middle rung: n_dirs kick directions x a few speeds, still a Discrete space (works with DQN)."""
    name = "fine"

    def __init__(self, n_dirs: int = 16, speeds=(2.0, 3.0, 4.0)):
        self.n_dirs, self.speeds = n_dirs, tuple(float(s) for s in speeds)
        self.space = spaces.Discrete(5 + n_dirs * len(self.speeds))

    def decode(self, action, has_ball: bool) -> Intent:
        a = int(action)
        if a < 5:
            return Intent(a)
        if not has_ball:
            return Intent(STAY)
        k = a - 5
        d, s = divmod(k, len(self.speeds))
        return Intent(STAY, (2 * math.pi * d / self.n_dirs, self.speeds[s]))

    def noop(self): return STAY
    def encode_move(self, move: int): return int(move)

    def encode_kick(self, angle: float, speed: float):
        d = int(round(angle / (2 * math.pi) * self.n_dirs)) % self.n_dirs
        s = min(range(len(self.speeds)), key=lambda i: abs(self.speeds[i] - speed))
        return 5 + d * len(self.speeds) + s


TIERS = {"discrete": DiscreteKick, "box": BoxKick, "fine": FineDiscrete}


def make_adapter(spec) -> Adapter:
    if isinstance(spec, Adapter):
        return spec
    if isinstance(spec, str):
        return TIERS[spec]()
    raise ValueError(f"unknown action tier {spec!r}; use one of {list(TIERS)} or an Adapter instance")
