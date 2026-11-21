"""grid2v2/physics.py — the ball, as a pure function.

Robots live on integer cells; cell (cx, cy) is the unit square centred on (cx, cy).  The ball has a float position,
a unit direction and a speed in cells per step.  Each step it travels `speed` cells, reflecting off the four walls,
and then loses FRICTION cells/step.  A ball that crosses a goal mouth is a goal; a ball that enters a cell holding a
robot is taken by that robot.  Nothing here knows about teams, rewards or randomness, so agents may call `simulate`
to look ahead (students: this is a model of the world; using it is a design decision you will defend).

Frame: x to the right, y up.  The left goal is the segment x = -0.5, GOAL_LO <= y <= GOAL_HI; the right goal is the
same segment at x = W - 0.5.  `Rink` is immutable; there is one per environment.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

FRICTION = 1.0       # cells/step lost per step
EPS = 1e-9
INF = float("inf")


@dataclass(frozen=True)
class Rink:
    width: int = 21
    height: int = 11
    goal_cells: int = 3          # height of each goal mouth, in cells, centred on the middle row

    @property
    def goal_lo(self) -> float: return (self.height - 1) / 2 - self.goal_cells / 2

    @property
    def goal_hi(self) -> float: return (self.height - 1) / 2 + self.goal_cells / 2

    @property
    def goal_rows(self) -> range:
        c = (self.height - 1) // 2; h = self.goal_cells // 2
        return range(c - h, c + h + 1)

    def inside(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.width and 0 <= cy < self.height

    def mirror_x(self, x: float) -> float:
        return (self.width - 1) - x

    def goal_centre(self, side: str) -> tuple[float, float]:
        return ((-0.5 if side == "left" else self.width - 0.5), (self.height - 1) / 2)


@dataclass
class Ball:
    x: float
    y: float
    cx: int
    cy: int
    dx: float = 0.0          # unit direction (0, 0 when still)
    dy: float = 0.0
    speed: float = 0.0       # cells per step

    @classmethod
    def at_cell(cls, cx: int, cy: int) -> "Ball":
        return cls(float(cx), float(cy), cx, cy)

    @property
    def moving(self) -> bool: return self.speed > EPS

    def stop(self) -> None:
        self.dx = self.dy = 0.0; self.speed = 0.0

    def kick(self, angle: float, speed: float) -> None:
        self.dx, self.dy, self.speed = math.cos(angle), math.sin(angle), float(speed)

    def copy(self) -> "Ball":
        return Ball(self.x, self.y, self.cx, self.cy, self.dx, self.dy, self.speed)


@dataclass
class Flight:
    """What happened while the ball travelled: `event` is None, ("goal", "left"|"right") or ("taken", occupant)."""
    event: tuple | None = None
    bounces: int = 0
    path: list = field(default_factory=list)   # (x, y) after each boundary crossing, for the renderer


def advance(rink: Rink, ball: Ball, distance: float, occupied: dict, path: bool = False) -> Flight:
    """Move `ball` up to `distance` cells along its direction.  `occupied` maps (cx, cy) -> occupant for every robot.
    The ball's own starting cell is never checked (a kick leaves the kicker's cell).  Mutates `ball`."""
    out = Flight()
    W, H = rink.width, rink.height
    r = distance
    while r > EPS:
        tx = INF if abs(ball.dx) < EPS else ((ball.cx + (0.5 if ball.dx > 0 else -0.5)) - ball.x) / ball.dx
        ty = INF if abs(ball.dy) < EPS else ((ball.cy + (0.5 if ball.dy > 0 else -0.5)) - ball.y) / ball.dy
        t = min(tx, ty)
        if t >= r - EPS:                                   # stops inside the current cell
            ball.x += ball.dx * r; ball.y += ball.dy * r
            return out
        ball.x += ball.dx * t; ball.y += ball.dy * t; r -= t
        if abs(tx - t) < EPS:                              # crossing a vertical boundary
            nx = ball.cx + (1 if ball.dx > 0 else -1)
            if nx < 0 or nx >= W:
                if rink.goal_lo - EPS <= ball.y <= rink.goal_hi + EPS:
                    out.event = ("goal", "left" if nx < 0 else "right")
                    if path: out.path.append((ball.x, ball.y))
                    return out
                ball.dx = -ball.dx; out.bounces += 1
            else:
                ball.cx = nx
        if abs(ty - t) < EPS:                              # crossing a horizontal boundary (both at a corner)
            ny = ball.cy + (1 if ball.dy > 0 else -1)
            if ny < 0 or ny >= H:
                ball.dy = -ball.dy; out.bounces += 1
            else:
                ball.cy = ny
        if path: out.path.append((ball.x, ball.y))
        who = occupied.get((ball.cx, ball.cy))
        if who is not None:
            out.event = ("taken", who)
            ball.x, ball.y = float(ball.cx), float(ball.cy)
            return out
    return out


def step_ball(rink: Rink, ball: Ball, occupied: dict, path: bool = False) -> Flight:
    """One environment step of free flight: travel `speed` cells, then apply friction."""
    if not ball.moving:
        return Flight()
    f = advance(rink, ball, ball.speed, occupied, path)
    if f.event is not None:
        ball.stop()
    else:
        ball.speed -= FRICTION
        if ball.speed <= EPS:
            ball.stop()
    return f


def simulate(rink: Rink, cx: int, cy: int, angle: float, speed: float, occupied: dict, max_steps: int = 20):
    """Dry-run a kick from cell (cx, cy): returns (event, ball, steps, path).  `event` is None when the ball comes to
    rest, else ("goal", side) or ("taken", occupant).  `occupied` must not contain the kicker's cell (or it is skipped
    only on the first segment anyway).  Robots are assumed to stand still."""
    b = Ball.at_cell(cx, cy); b.kick(angle, speed)
    pts = [(b.x, b.y)]
    for k in range(1, max_steps + 1):
        f = step_ball(rink, b, occupied, path=True)
        pts += f.path
        if f.event is not None:
            return f.event, b, k, pts
        if not b.moving:
            pts.append((b.x, b.y))
            return None, b, k, pts
    return None, b, max_steps, pts


def predict(rink: Rink, x: float, y: float, vx: float, vy: float, occupied: dict, max_steps: int = 20):
    """Like `simulate`, but for a ball already in flight at float position (x, y) with velocity (vx, vy) cells/step."""
    speed = math.hypot(vx, vy)
    b = Ball(x, y, int(math.floor(x + 0.5)), int(math.floor(y + 0.5)), vx / speed, vy / speed, speed)
    pts = [(b.x, b.y)]
    for k in range(1, max_steps + 1):
        f = step_ball(rink, b, occupied, path=True)
        pts += f.path
        if f.event is not None:
            return f.event, b, k, pts
        if not b.moving:
            pts.append((b.x, b.y))
            return None, b, k, pts
    return None, b, max_steps, pts


def kick_range(speed: float) -> float:
    """Total distance a kick at `speed` travels with FRICTION: s + (s-1) + ... down to zero."""
    d, s = 0.0, speed
    while s > EPS:
        d += s; s -= FRICTION
    return d
