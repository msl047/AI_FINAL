"""grid2v2/render.py — the broadcast view: frames from a match log (or a live env), GIF or MP4.

Off by default: training never touches this module.  Record a match with `--log match.json`, then

    python -m grid2v2.render match.json --out match.gif                    # whole match, rink skin
    python -m grid2v2.render match.json --out match.mp4 --skin pitch --fps 15 --interp 3
    python -m grid2v2.render match.json --out goal.gif --range 180:230      # a stretch of steps
    python -m grid2v2.render match.json --out highlights.gif --highlights   # 25 steps around every goal

Between two logged steps the robots glide cell to cell and the ball follows its true path (re-run through the physics,
so bank shots bend at the wall).  Facing is the last direction a robot moved.  Banners name kicks (SHOT / PASS /
CLEARANCE by what happened next), tackles, saves (an interception inside the keeper's crease) and goals.
Needs Pillow; MP4 needs imageio[ffmpeg].
"""
from __future__ import annotations
import argparse, json, math, os
from PIL import Image, ImageDraw, ImageFont
from .physics import Rink, Ball, advance, simulate

SKINS = {
    "rink": {"bg": (24, 32, 44), "ice": (226, 236, 242), "ice2": (218, 229, 236), "line": (205, 40, 50), "line2": (40, 90, 170),
             "board": (255, 255, 255), "crease": (160, 200, 235), "text": (240, 244, 248), "sub": (170, 180, 192)},
    "pitch": {"bg": (24, 32, 44), "ice": (48, 140, 86), "ice2": (44, 130, 80), "line": (255, 255, 255), "line2": (255, 255, 255),
              "board": (255, 255, 255), "crease": (70, 160, 100), "text": (240, 244, 248), "sub": (170, 180, 192)},
}
TEAM = {"red": (178, 24, 54), "blue": (22, 120, 160)}
TEAM_LIGHT = {"red": (236, 170, 150), "blue": (160, 215, 230)}
BANNER = {"GOAL": (205, 40, 50), "SHOT": (240, 110, 20), "PASS": (240, 110, 20), "CLEARANCE": (240, 110, 20), "BACK PASS": (240, 110, 20), "KICK": (240, 110, 20),
          "TACKLE": (120, 80, 200), "BLOCK": (120, 80, 200), "SAVE": (30, 150, 110), "INTERCEPT": (30, 150, 110), "KICK-OFF": (90, 100, 120)}


def _font(size, bold=False):
    for p in (f"/usr/share/fonts/opentype/inter/Inter-{'Bold' if bold else 'Medium'}.otf",
              f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
              "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Renderer:
    def __init__(self, rink: Rink, crease: int = 2, skin: str = "rink", cell: int = 48, names=None):
        self.rink, self.crease, self.cell = rink, crease, cell
        self.c = SKINS[skin]; self.skin = skin
        self.names = names or {"red": "RED", "blue": "BLUE"}
        self.W, self.H = rink.width * cell, rink.height * cell
        self.top, self.side = 120, 70
        self.size = (self.W + 2 * self.side, self.H + self.top + 40)
        self.f_big, self.f_mid, self.f_small, self.f_num = _font(34, True), _font(20, True), _font(16), _font(18, True)
        self.board = self._board()

    # ------------------------------------------------------------------------------------------------ geometry
    def xy(self, x, y):
        """Rink coordinates (cell centres at integers, y up) -> pixel centre."""
        return (self.side + (x + 0.5) * self.cell, self.top + (self.rink.height - 1 - y + 0.5) * self.cell)

    def _board(self):
        """The static background: boards, ice, lines, goals, creases."""
        c = self.c; rk = self.rink; cs = self.cell
        im = Image.new("RGB", self.size, c["bg"]); d = ImageDraw.Draw(im, "RGBA")
        x0, y0, x1, y1 = self.side, self.top, self.side + self.W, self.top + self.H
        d.rounded_rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], radius=22 if self.skin == "rink" else 2, fill=c["board"])
        d.rounded_rectangle([x0, y0, x1, y1], radius=18 if self.skin == "rink" else 0, fill=c["ice"])
        for i in range(rk.width):                              # faint cell stripes so positions read as cells
            if i % 2: d.rectangle([x0 + i * cs, y0, x0 + (i + 1) * cs, y1], fill=c["ice2"])
        for i in range(rk.width + 1):                          # grid
            d.line([x0 + i * cs, y0, x0 + i * cs, y1], fill=(0, 0, 0, 14))
        for j in range(rk.height + 1):
            d.line([x0, y0 + j * cs, x1, y0 + j * cs], fill=(0, 0, 0, 14))
        # creases (the forbidden cells, drawn as they are)
        for row in rk.goal_rows:
            for k in range(self.crease):
                for x in (k, rk.width - 1 - k):
                    px, py = self.xy(x, row)
                    d.rectangle([px - cs / 2, py - cs / 2, px + cs / 2, py + cs / 2], fill=c["crease"] + (150,))
        # lines
        mid = x0 + self.W / 2
        d.line([mid, y0, mid, y1], fill=c["line"], width=4)
        d.ellipse([mid - 2.2 * cs, y0 + self.H / 2 - 2.2 * cs, mid + 2.2 * cs, y0 + self.H / 2 + 2.2 * cs], outline=c["line2"], width=3)
        for x in (x0 + self.crease * cs, x1 - self.crease * cs):
            d.line([x, y0, x, y1], fill=c["line2"], width=3)
        # goal mouths: a net box outside the boards
        gy0 = self.top + (rk.height - 1 - rk.goal_hi + 0.5) * cs; gy1 = self.top + (rk.height - 1 - rk.goal_lo + 0.5) * cs
        for team, gx0, gx1 in (("red", x0 - 40, x0 - 6), ("blue", x1 + 6, x1 + 40)):     # red defends the left goal
            d.rectangle([gx0, gy0, gx1, gy1], fill=(235, 235, 235), outline=TEAM[team], width=4)
            for k in range(1, 4): d.line([gx0, gy0 + k * (gy1 - gy0) / 4, gx1, gy0 + k * (gy1 - gy0) / 4], fill=(190, 190, 190))
            for k in range(1, 3): d.line([gx0 + k * (gx1 - gx0) / 3, gy0, gx0 + k * (gx1 - gx0) / 3, gy1], fill=(190, 190, 190))
        return im

    # -------------------------------------------------------------------------------------------------- frames
    def frame(self, pos: dict, holder, ball_xy, facing: dict, score: dict, t: int, max_steps: int,
              trail=(), banner=None, possession=None, stunned=()) -> Image.Image:
        im = self.board.copy(); d = ImageDraw.Draw(im, "RGBA"); c = self.c; cs = self.cell
        # scoreboard
        s = f"{self.names['red']}  {score['red']}  :  {score['blue']}  {self.names['blue']}"
        w = d.textlength(s, font=self.f_big); d.text(((self.size[0] - w) / 2, 18), s, font=self.f_big, fill=c["text"])
        sub = f"step {t} / {max_steps}"
        if possession is not None: sub += f"      possession  red {possession:.0%}"
        w = d.textlength(sub, font=self.f_small); d.text(((self.size[0] - w) / 2, 62), sub, font=self.f_small, fill=c["sub"])
        d.text((self.side, 30), "RED", font=self.f_mid, fill=TEAM_LIGHT["red"])
        w = d.textlength("BLUE", font=self.f_mid); d.text((self.size[0] - self.side - w, 30), "BLUE", font=self.f_mid, fill=TEAM_LIGHT["blue"])
        # banner
        if banner:
            kind, text = banner
            col = BANNER.get(kind, (90, 100, 120)); label = f"{kind}   {text}".strip()
            w = d.textlength(label, font=self.f_mid)
            bx = (self.size[0] - w) / 2
            d.rounded_rectangle([bx - 18, 84, bx + w + 18, 112], radius=6, fill=col)
            d.text((bx, 87), label, font=self.f_mid, fill=(255, 255, 255))
        # ball trail
        n = len(trail)
        for k, (x, y) in enumerate(trail):
            px, py = self.xy(x, y); r = 4 + 3 * k / max(1, n); a = int(40 + 120 * k / max(1, n))
            d.ellipse([px - r, py - r, px + r, py + r], fill=(255, 255, 255, a))
        # robots
        R = cs * 0.36
        for a, (x, y) in pos.items():
            team = a.split("_")[0]; num = int(a.split("_")[1]) + 1
            px, py = self.xy(x, y)
            fx, fy = facing.get(a, (1 if team == "red" else -1, 0))
            tip = (px + fx * R * 1.55, py - fy * R * 1.55)
            l = (px + fx * R * 0.9 - fy * R * 0.5, py - fy * R * 0.9 - fx * R * 0.5)
            r_ = (px + fx * R * 0.9 + fy * R * 0.5, py - fy * R * 0.9 + fx * R * 0.5)
            d.polygon([tip, l, r_], fill=(255, 255, 255))
            d.ellipse([px - R, py - R, px + R, py + R], fill=TEAM[team], outline=(255, 255, 255), width=3)
            w = d.textlength(str(num), font=self.f_num); d.text((px - w / 2, py - 11), str(num), font=self.f_num, fill=(255, 255, 255))
            if a in stunned:                                            # stunned: a ring of stars
                for k in range(6):
                    ang = k * math.pi / 3; sx, sy = px + math.cos(ang) * R * 1.35, py + math.sin(ang) * R * 1.35
                    d.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=(255, 210, 60))
        # ball
        if holder is not None and holder in pos:
            x, y = pos[holder]; fx, fy = facing.get(holder, (1, 0)); px, py = self.xy(x, y)
            px, py = px + fx * R * 1.1, py - fy * R * 1.1
        else:
            px, py = self.xy(*ball_xy)
        d.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(20, 20, 20) if self.skin == "rink" else (255, 255, 255), outline=(255, 255, 255) if self.skin == "rink" else (30, 30, 30), width=2)
        return im


# ---------------------------------------------------------------------------------------------------- from logs
def _ball_path(rink, f0, f1):
    """The ball's polyline between two logged frames, from the physics (bank shots bend at the wall)."""
    x, y, vx, vy = f0["ball"]
    kick = next((e for e in f1["events"] if e[0] == "kick"), None)
    if kick is not None:                                           # released this step: start from the kicker's cell
        kx, ky = f0["pos"][kick[1]]
        b = Ball.at_cell(int(kx), int(ky)); b.kick(kick[2], kick[3]); speed = kick[3]; x, y = float(kx), float(ky)
    elif f0["holder"] is not None or (abs(vx) < 1e-9 and abs(vy) < 1e-9):
        return [(x, y), tuple(f1["ball"][:2])]
    else:
        speed = math.hypot(vx, vy)
        b = Ball(x, y, int(math.floor(x + 0.5)), int(math.floor(y + 0.5)), vx / speed, vy / speed, speed)
    occ = {tuple(p): a for a, p in f1["pos"].items()}            # robots at their end-of-step cells
    fl = advance(rink, b, speed, occ, path=True)
    pts = [(x, y)] + fl.path + [(b.x, b.y)]
    if fl.event is not None and fl.event[0] == "goal":
        pts.append(pts[-1])
    else:
        pts[-1] = tuple(f1["ball"][:2]) if f1["holder"] is None else tuple(f1["pos"][f1["holder"]])
    return pts


def _along(pts, u):
    """Point at fraction u in [0, 1] of the polyline's arc length."""
    segs = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(segs)
    if total < 1e-9: return pts[-1]
    target = u * total
    for (a, b), L in zip(zip(pts[:-1], pts[1:]), segs):
        if target <= L:
            k = target / L if L > 0 else 1
            return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)
        target -= L
    return pts[-1]


def _name(agent: str) -> str:
    """blue_1 -> 'blue 2': robots are numbered 1 and 2 on screen."""
    t, i = agent.split("_"); return f"{t} {int(i) + 1}"


def _banner_for(events, frames, i, rink, crease):
    """The most newsworthy event of step i, with a label that says what the kick turned out to be."""
    out = None
    for e in events:
        k = e[0]
        if k == "goal":
            return ("GOAL", f"{e[1].upper()}  ({_name(e[2]) if e[2] else ''})")
        if k == "kick":
            # what kind of kick: a dry run through the physics with the robots where they stood
            f0 = frames[i - 1] if i > 0 else frames[i]
            kx, ky = f0["pos"][e[1]]; team = e[1].split("_")[0]
            occ = {tuple(p): a for a, p in f0["pos"].items() if a != e[1]}
            ev, b, _, _ = simulate(rink, int(kx), int(ky), e[2], e[3], occ)
            towards = (rink.width - 1 - b.cx) if team == "red" else b.cx
            if ev is not None and ev[0] == "goal": kind = "SHOT"
            elif ev is not None and ev[1].split("_")[0] == team: kind = "PASS"
            elif ev is not None and towards <= crease + 1: kind = "SHOT"
            elif towards <= 2: kind = "SHOT"
            elif (b.cx - kx) * (1 if team == "red" else -1) > 0: kind = "CLEARANCE" if abs(kx - (0 if team == "red" else rink.width - 1)) <= 3 else "KICK"
            else: kind = "BACK PASS" if ev is None else "KICK"
            out = out or (kind, _name(e[1]) + f"  speed {e[3]:.0f}")
        elif k == "tackle":
            out = ("TACKLE", f"{_name(e[1])} takes it from {_name(e[2])}")
        elif k == "block":
            out = ("BLOCK", f"{_name(e[1])} blocks {_name(e[2])}'s kick")
        elif k == "intercept":
            a = e[1]; x, y = frames[i]["pos"][a]
            own_crease = (y in rink.goal_rows) and ((a.startswith("red") and x < crease) or (a.startswith("blue") and x >= rink.width - crease))
            out = out or (("SAVE", _name(a)) if own_crease else ("INTERCEPT", _name(a)))
        elif k == "kickoff" and e[1] is not None:
            out = out or ("KICK-OFF", f"{e[1]} ball")
    return out


def render_log(path, out, skin="rink", fps=10, interp=2, cell=48, step_range=None, highlights=False, hold=8, max_frames=None):
    with open(path) as f:
        log = json.load(f)
    rink = Rink(*log["rink"][:2], log["rink"][2]); frames = log["frames"]
    crease = log.get("crease", 2); max_steps = log.get("max_steps", 600)
    names = {t: n.split(".")[-1].upper() for t, n in log["result"]["names"].items()}
    R = Renderer(rink, crease, skin, cell, names)
    idx = list(range(len(frames) - 1))
    if step_range:
        a, b = step_range; idx = [i for i in idx if a <= i < b]
    if highlights:
        goals = [i for i, f in enumerate(frames) if any(e[0] == "goal" for e in f["events"])]
        keep = set()
        for g in goals: keep.update(range(max(0, g - 22), min(len(frames) - 1, g + 3)))
        idx = sorted(keep)
    facing = {}
    trail = []
    images = []
    banner, banner_left = None, 0
    held = 0
    for i in idx:
        f0, f1 = frames[i], frames[i + 1]
        for a, p in f1["pos"].items():
            q = f0["pos"][a]
            if p != q:
                dx, dy = p[0] - q[0], p[1] - q[1]
                if abs(dx) + abs(dy) <= 1: facing[a] = (dx, dy)
        b = _banner_for(f1["events"], frames, i + 1, rink, crease)
        pts = _ball_path(rink, f0, f1)
        goal_now = any(e[0] == "goal" for e in f1["events"])
        poss = sum(1 for f in frames[:i + 1] if f["holder"] and f["holder"].startswith("red")) / max(1, sum(1 for f in frames[:i + 1] if f["holder"]))
        for k in range(interp):
            u = (k + 1) / interp
            pos = {}
            for a, p in f1["pos"].items():
                q = f0["pos"][a]
                if goal_now: pos[a] = tuple(q) if u < 1 else tuple(p)       # teleport to kick-off on the last subframe
                else: pos[a] = (q[0] + (p[0] - q[0]) * u, q[1] + (p[1] - q[1]) * u)
            if goal_now and u >= 1:
                ball = tuple(f1["ball"][:2]); holder = f1["holder"]; trail.clear()
            else:
                ball = _along(pts, u); holder = f0["holder"] if (f0["holder"] is not None and u < 1) else (f1["holder"] if u >= 1 else None)
                if holder is None: trail.append(ball)
                else: trail.clear()
            del trail[:-10]
            if b and u >= 1: banner, banner_left = b, hold + (hold if b[0] == "GOAL" else 0)     # events land at the end of the step
            score = f1["score"] if u >= 1 else f0["score"]
            images.append(R.frame(pos, holder, ball, facing, score, f1["t"], max_steps, list(trail),
                                  banner if banner_left > 0 else None, poss, stunned=set(f1.get("stunned", {}))))
            banner_left -= 1
            if max_frames and len(images) >= max_frames: break
        if max_frames and len(images) >= max_frames: break
    _save(images, out, fps)
    return len(images)


def _save(images, out, fps):
    if out.lower().endswith(".gif"):
        images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=False)
    else:
        import imageio.v2 as imageio, numpy as np
        with imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=None) as w:
            for im in images: w.append_data(np.asarray(im))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log"); p.add_argument("--out", default="match.gif")
    p.add_argument("--skin", choices=list(SKINS), default="rink"); p.add_argument("--fps", type=int, default=10)
    p.add_argument("--interp", type=int, default=2); p.add_argument("--cell", type=int, default=48)
    p.add_argument("--range", default=None, help="a:b steps"); p.add_argument("--highlights", action="store_true")
    a = p.parse_args()
    rng = tuple(int(v) for v in a.range.split(":")) if a.range else None
    n = render_log(a.log, a.out, a.skin, a.fps, a.interp, a.cell, rng, a.highlights)
    print(f"{n} frames -> {a.out}")


if __name__ == "__main__":
    main()
