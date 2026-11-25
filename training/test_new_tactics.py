"""Compare keeper outlets and near-goal shot creation against p05414."""

import argparse
import csv
import json
import math
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arena"))

from grid2v2 import E, N, S, W
from grid2v2.actions import MOVE_DELTA
from grid2v2.agents import ScriptedAgent, manhattan
from grid2v2.match import load_team, play
from grid2v2.physics import simulate
from teams import team2_chip, team2_explore

ATTACK = ("wide_south/y_first", "race/y_first", "carry/script", "chip_north")
KEEPER_MODES = ("outlet", "territory")
FIELD_MODES = ("angle_scan", "shot_setup")
OPPONENTS = (
    "baselines.learned", "baselines.scripted_box", "teams.team1",
    "two_field_box", "two_field_fine", "p05414",
)
CLEAR_ANGLES = tuple(math.radians(angle) for angle in range(-90, 91, 15))
CLEAR_POWERS = (1.5, 2.0, 3.0)


class NewKeeper(ScriptedAgent):
    def __init__(self, original, mode, trace):
        super().__init__(original.adapter, original.rink, index=original.index)
        self.mode = mode
        self.trace = trace

    def _clear(self, me, mate, opponents, occupied):
        fallback = super()._clear(me, mate, opponents, occupied)
        if mate is None or not opponents or self.i_am_stunned:
            return fallback
        self.trace["opportunities"] += 1
        angles = (*CLEAR_ANGLES, math.atan2(mate[1] - me[1], mate[0] - me[0]))
        occupied = self._kick_occ(me, occupied)
        best = None

        for angle in angles:
            for power in CLEAR_POWERS:
                event, rest, turns, _ = simulate(self.rink, *me, angle, power, occupied)
                received = event == ("taken", "mate")
                if event is not None and not received:
                    continue
                landing = (rest.cx, rest.cy)
                gain = landing[0] - me[0]
                if gain < 1 or self.forbidden(landing):
                    continue

                ours = turns if received else max(turns, manhattan(mate, landing))
                theirs = min(manhattan(pos, landing) for pos, _ in opponents)
                margin = theirs - ours
                if self.mode == "outlet":
                    if not received and margin < 1:
                        continue
                    score = (received, margin, gain)
                else:
                    if not received and (margin < -1 or gain < 3):
                        continue
                    score = (gain, received, margin)
                if best is None or score > best[0]:
                    best = (score, angle, power, received)

        if best is None:
            return fallback
        _, angle, power, received = best
        action = self.adapter.encode_kick(angle, power)
        if not np.array_equal(action, fallback):
            self.trace["changes"] += 1
            self.trace["pass_choices" if received else "space_choices"] += 1
        return action


class NewField(team2_explore.ExploreField):
    def __init__(self, original, mode, trace):
        super().__init__(original, ATTACK, None)
        self.mode = mode
        self.changes = trace

    def shots(self, origin, occupied):
        goal_x = self.rink.width - 0.5
        for fraction in (0.50, 0.35, 0.65, 0.20, 0.80, 0.08, 0.92):
            y = self.rink.goal_lo + fraction * (self.rink.goal_hi - self.rink.goal_lo)
            angle = math.atan2(y - origin[1], goal_x - origin[0])
            event, _, turns, _ = simulate(
                self.rink, *origin, angle, 3.0, self._kick_occ(origin, occupied), max_steps=2,
            )
            if event == ("goal", "right"):
                yield turns, angle

    def act(self, obs):
        fallback = super().act(obs)
        if (
            obs["self"][2] < 0.5 or obs["self"][3] > 0.5
            or obs["self"][0] < self.rink.width - 7
        ):
            return fallback
        me, _, _, _, opponents, occupied, _ = self._parse(obs)
        self.changes["opportunities"] += 1

        kick = self.adapter.decode(fallback, True).kick
        if kick is not None:
            event, *_ = simulate(self.rink, *me, *kick, self._kick_occ(me, occupied))
            if event in (("goal", "right"), ("taken", "mate")):
                return fallback

        shots_now = list(self.shots(me, occupied))
        if shots_now:
            _, angle = min(shots_now, key=lambda shot: shot[0])
            action = self.adapter.encode_kick(angle, 3.0)
            if not np.array_equal(action, fallback):
                self.changes["changes"] += 1
                self.changes["extra_shots"] += 1
            return action
        if self.mode == "angle_scan" or not opponents:
            return fallback

        choices = []
        for move in (E, N, S, W):
            dx, dy = MOVE_DELTA[move]
            cell = (me[0] + dx, me[1] + dy)
            if not self.legal(cell, occupied) or cell[0] >= self.rink.width - self.crease:
                continue
            gap = min(manhattan(cell, pos) for pos, _ in opponents)
            if gap < 2:
                continue
            openings = list(self.shots(cell, occupied))
            if openings:
                rank = (-min(turns for turns, _ in openings), len(openings), gap, cell[0])
                choices.append((rank, move))
        if not choices:
            return fallback

        _, move = max(choices, key=lambda choice: choice[0])
        action = self.adapter.encode_move(move)
        if not np.array_equal(action, fallback):
            self.changes["changes"] += 1
            self.changes["setup_moves"] += 1
        return action


def make_candidate(mode, stats):
    module = ModuleType(mode)
    module.TIER = team2_explore.TIER

    def make_team(side, ids, env):
        if mode in FIELD_MODES:
            bots = team2_chip.make_team(side, ids, env)
            bots[ids[0]] = NewField(bots[ids[0]], mode, stats)
        else:
            bots = team2_explore.build_team(side, ids, env, ATTACK)
            if mode in KEEPER_MODES:
                bots[ids[1]] = NewKeeper(bots[ids[1]], mode, stats)
        return bots

    module.make_team = make_team
    return module


@cache
def load_opponent(name):
    if name == "p05414":
        return make_candidate("control", None)
    if not name.startswith("two_field_"):
        return load_team(name)

    module = ModuleType(name)
    module.TIER = name.removeprefix("two_field_")
    module.make_team = lambda side, ids, env: {
        agent: ScriptedAgent(env.adapters[agent], env.rink, index=0) for agent in ids
    }
    return module


def run_match(job):
    mode, name, seed, swap = job
    stats = dict(opportunities=0, changes=0, pass_choices=0, space_choices=0,
                 extra_shots=0, setup_moves=0)
    result = play(make_candidate(mode, stats), load_opponent(name), seed=seed, swap=swap)
    side = "blue" if swap else "red"
    other = "red" if swap else "blue"
    gf, ga = result["score"][side], result["score"][other]
    return dict(
        mode=mode, opponent=name, seed=seed, side=side, gf=gf, ga=ga,
        points=3 if gf > ga else int(gf == ga), strikes=sum(result["strikes"].values()),
        forfeit=result["forfeit"], **stats,
    )


def summarize(rows, modes):
    groups = {(row["mode"], row["opponent"], row["seed"], row["side"]): row for row in rows}
    report = {}
    for mode in modes:
        report[mode] = {}
        for name in (*OPPONENTS, "overall"):
            selected = [row for row in rows if row["mode"] == mode
                        and (name == "overall" or row["opponent"] == name)]
            controls = [groups["control", r["opponent"], r["seed"], r["side"]] for r in selected]
            pairs = list(zip(selected, controls))
            report[mode][name] = dict(
                games=len(pairs),
                wdl=[sum(r["points"] == points for r in selected) for points in (3, 1, 0)],
                points=sum(r["points"] for r in selected),
                control_points=sum(r["points"] for r in controls),
                points_gain=sum(a["points"] - b["points"] for a, b in pairs),
                gf_gain=sum(a["gf"] - b["gf"] for a, b in pairs),
                ga_gain=sum(a["ga"] - b["ga"] for a, b in pairs),
                **{key: sum(row[key] for row in selected) for key in (
                    "opportunities", "changes", "pass_choices", "space_choices",
                    "extra_shots", "setup_moves", "strikes",
                )},
                forfeits=sum(row["forfeit"] is not None for row in selected),
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", choices=("distribution", "finishing", "all"), default="all")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=32, help="seeds per block and opponent")
    parser.add_argument("--seed", type=int, default=807200)
    args = parser.parse_args()
    if not (1 <= args.workers <= 61 and 1 <= args.seeds <= 10_000 and args.seed >= 0):
        parser.error("Use 1-61 workers, 1-10000 seeds, and a nonnegative seed.")

    if args.test == "distribution":
        modes = KEEPER_MODES
    elif args.test == "finishing":
        modes = FIELD_MODES
    else:
        modes = (*KEEPER_MODES, *FIELD_MODES)
    arms = ("control", *modes)
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"new_tactics_{args.test}_", dir=data))
    print(f"Results folder: {out}")
    reports = []

    with (
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        (out / "matches.csv").open("w", newline="", encoding="utf-8", buffering=1) as file,
    ):
        writer = None
        for block in range(2):
            start = args.seed + 10_000 * block
            jobs = [(mode, name, seed, swap) for seed in range(start, start + args.seeds)
                    for name in OPPONENTS for swap in (False, True) for mode in arms]
            rows = []
            for index, row in enumerate(pool.map(run_match, jobs, chunksize=2), 1):
                row = {"block": block + 1, **row}
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
                if index == 1 or index % 200 == 0 or index == len(jobs):
                    print(f"Block {block + 1}: {index}/{len(jobs)} matches.")
            report = summarize(rows, arms)
            reports.append(report)
            for mode in modes:
                r = report[mode]["overall"]
                print(
                    f"{mode}: {r['points_gain']:+} points, {r['changes']} changes, "
                    f"{r['gf_gain']:+} goals scored, {r['ga_gain']:+} conceded, "
                    f"{r['strikes']} strikes, {r['forfeits']} forfeits"
                )

    (out / "summary.json").write_text(json.dumps(
        {"config": vars(args), "attack": ATTACK, "blocks": reports}, indent=2
    ), encoding="utf-8")
    print(f"Summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
