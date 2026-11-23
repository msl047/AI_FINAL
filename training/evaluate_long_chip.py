"""Compare the fixed chip rule with a long-chip-only variant."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arena"))

from grid2v2.match import play
from grid2v2.physics import simulate
from teams import team2_chip


class ChipTestField(team2_chip.ChipField):
    def __init__(self, original, long_gain, long_only, counts):
        super().__init__(
            original.adapter,
            original.rink,
            index=original.index,
            policies=original.policies,
            chip_policy={},
        )
        self.long_gain = long_gain
        self.long_only = long_only
        self.counts = counts
        self.candidate = None

    def find_chip(self, me, opps, occupied, fallback):
        candidate = super().find_chip(me, opps, occupied, fallback)
        if candidate is None:
            return None

        action, _ = candidate
        angle, speed = self.adapter.decode(action, True).kick
        _, ball, _, _ = simulate(self.rink, *me, angle, speed, occupied)
        gain = ball.cx - me[0]

        if self.long_only and gain < self.long_gain:
            self.counts["short_candidates_filtered"] += 1
            return None

        self.candidate = (action, gain)
        return candidate

    def act(self, obs):
        self.candidate = None
        chosen = super().act(obs)

        if self.candidate is not None:
            action, gain = self.candidate
            if np.array_equal(chosen, action):
                length = "long_chips" if gain >= self.long_gain else "short_chips"
                self.counts[length] += 1
        return chosen


def make_test_team(long_gain, long_only, counts):
    module = ModuleType("long_only" if long_only else "fixed_chip")
    module.TIER = team2_chip.TIER

    def make_team(side, ids, env):
        bots = team2_chip.make_team(side, ids, env)
        field_id = ids[0]
        bots[field_id] = ChipTestField(
            bots[field_id], long_gain, long_only, counts
        )
        return bots

    module.make_team = make_team
    return module


def play_series(opponent, args, long_only):
    counts = Counter()
    team = make_test_team(args.long_gain, long_only, counts)
    games = []

    for seed in range(args.seed, args.seed + args.games // 2):
        for swap in (False, True):
            result = play(team, opponent, seed=seed, swap=swap)
            ours, theirs = ("blue", "red") if swap else ("red", "blue")
            games.append({
                "seed": seed,
                "side": ours,
                "gf": result["score"][ours],
                "ga": result["score"][theirs],
                "strikes": result["strikes"][ours],
                "forfeit": result["forfeit"],
            })

    wins = sum(game["gf"] > game["ga"] for game in games)
    draws = sum(game["gf"] == game["ga"] for game in games)
    losses = len(games) - wins - draws
    gf = sum(game["gf"] for game in games)
    ga = sum(game["ga"] for game in games)

    print(
        f"{team.__name__} vs {opponent}: {wins}-{draws}-{losses} W-D-L, "
        f"{gf}-{ga} goals, GD {gf - ga:+d}, "
        f"{counts['short_chips']} short and {counts['long_chips']} long chips, "
        f"{counts['short_candidates_filtered']} short candidates filtered, "
        f"{sum(game['strikes'] for game in games)} strikes",
        flush=True,
    )

    return {
        "variant": team.__name__,
        "opponent": opponent,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "gf": gf,
        "ga": ga,
        "short_chips": counts["short_chips"],
        "long_chips": counts["long_chips"],
        "short_candidates_filtered": counts["short_candidates_filtered"],
        "games": games,
    }


def test_long_chip(args):
    output = ROOT / "data"
    output.mkdir(exist_ok=True)
    report = output / f"long_chip_seed{args.seed}_gain{args.long_gain}.json"
    rows = []

    for opponent in args.opponents:
        for long_only in (False, True):
            rows.append(play_series(opponent, args, long_only))
            report.write_text(
                json.dumps({"settings": vars(args), "results": rows}, indent=2),
                encoding="utf-8",
            )

    print(f"Saved {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--long-gain", type=int, choices=range(3, 7), default=4)
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["baselines.learned", "baselines.scripted_box"],
    )
    args = parser.parse_args()

    if args.games < 2 or args.games % 2 or args.seed < 0:
        parser.error("games must be even and at least 2, and seed must be nonnegative")

    test_long_chip(args)