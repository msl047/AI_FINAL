"""Learn when to use Team 2's chip from paired simulator rollouts."""

import argparse
from collections import defaultdict
from itertools import count
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arena"))

from grid2v2 import Ball, Grid2v2
from grid2v2.match import load_team, play, tier_of
from teams import team2, team2_chip

OPPONENTS = ("baselines.learned", "baselines.scripted_box", "baselines.scripted")
POLICY_FILE = team2_chip.TEAM_DIR / "chip_policy.json"


def make_fixed_chip(side, ids, arena):
    bots = team2_chip.make_team(side, ids, arena)
    bots[ids[0]].chip_policy = {}
    return bots


FIXED_CHIP = ModuleType("fixed_chip")
FIXED_CHIP.TIER = team2_chip.TIER
FIXED_CHIP.make_team = make_fixed_chip


def restore_position(frame, opponent, seed):
    arena = Grid2v2(tiers={"red": team2_chip.TIER, "blue": tier_of(opponent)})
    arena.reset(seed=seed)
    arena.pos.update({name: tuple(cell) for name, cell in frame["pos"].items()})
    arena.stunned.update(frame["stunned"])
    arena.score.update(frame["score"])
    arena.t = frame["t"]
    arena.holder = "red_0"
    arena.ball = Ball.at_cell(*arena.pos["red_0"])
    return arena


def prepare_position(frame, opponent):
    arena = restore_position(frame, opponent, seed=0)
    field = make_fixed_chip("red", arena.members["red"], arena)["red_0"]
    obs = arena.observe("red_0")
    key = field.chip_key(obs)
    if key is None:
        return None

    field.reset()
    chip = field.act(obs)
    field.chip_policy = {key: False}
    field.reset()
    script = field.act(obs)
    return None if np.array_equal(chip, script) else key


def rollout(frame, opponent, seed, key, use_chip, horizon):
    arena = restore_position(frame, opponent, seed)
    bots = make_fixed_chip("red", arena.members["red"], arena)
    bots.update(opponent.make_team("blue", arena.members["blue"], arena))
    for bot in bots.values():
        bot.reset()

    field = bots["red_0"]
    field.chip_policy = {} if use_chip else {key: False}
    obs = {name: arena.observe(name) for name in arena.agents}

    for _ in range(horizon):
        actions = {name: bots[name].act(obs[name]) for name in arena.agents}
        # Both branches use the fixed chip policy after the first decision.
        field.chip_policy = {}
        obs, rewards, _, truncated, _ = arena.step(actions)
        reward = float(rewards["red_0"])
        if reward != 0 or truncated["red_0"]:
            return reward
    return 0.0


def save_policy(buckets, args):
    summary = {}
    for key, bucket in sorted(buckets.items()):
        pairs = bucket["pairs"]
        script = bucket["script"] / pairs
        chip = bucket["chip"] / pairs
        summary[key] = {
            "positions": bucket["positions"],
            "pairs": pairs,
            "script_value": script,
            "chip_value": chip,
            "use_chip": bucket["positions"] < 8 or chip >= script - 0.05,
        }

    saved = {"settings": vars(args), "buckets": summary}
    temp = POLICY_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    temp.replace(POLICY_FILE)
    return sum(not row["use_chip"] for row in summary.values())


def train_chip_selector(args):
    rng = np.random.default_rng(args.seed)
    deadline = monotonic() + 60 * args.minutes
    buckets = defaultdict(
        lambda: {"positions": 0, "pairs": 0, "script": 0.0, "chip": 0.0}
    )
    positions = 0
    seen = set()

    with TemporaryDirectory() as folder:
        replay = Path(folder) / "training_match.json"
        for game in count():
            if monotonic() >= deadline:
                break

            opponent_name = OPPONENTS[game % len(OPPONENTS)]
            opponent = load_team(opponent_name)
            collector = team2 if (game // len(OPPONENTS)) % 2 == 0 else FIXED_CHIP
            match_seed = 10_000 + 100_000 * args.seed + game
            result = play(collector, opponent, seed=match_seed, log=replay)
            if result["forfeit"]:
                raise RuntimeError(result["forfeit"])

            frames = json.loads(replay.read_text(encoding="utf-8"))["frames"]
            candidates = [
                frame for frame in frames[::3]
                if frame["holder"] == "red_0"
                and frame["pos"]["red_0"][0] < 13
                and frame["t"] <= 600 - args.horizon
            ]
            rng.shuffle(candidates)

            sampled = 0
            for frame in candidates:
                if sampled >= args.per_match or monotonic() >= deadline:
                    break

                signature = (
                    opponent_name,
                    tuple(tuple(frame["pos"][name]) for name in sorted(frame["pos"])),
                    tuple(sorted(frame["stunned"].items())),
                )
                if signature in seen:
                    continue
                seen.add(signature)

                key = prepare_position(frame, opponent)
                if key is None:
                    continue

                bucket = buckets[key]
                for _ in range(args.trials):
                    trial_seed = int(rng.integers(2**31))
                    bucket["script"] += rollout(
                        frame, opponent, trial_seed, key, False, args.horizon
                    )
                    bucket["chip"] += rollout(
                        frame, opponent, trial_seed, key, True, args.horizon
                    )
                    bucket["pairs"] += 1

                bucket["positions"] += 1
                positions += 1
                sampled += 1

            if game % 10 == 9 and positions:
                skipped = save_policy(buckets, args)
                remaining = max(0, (deadline - monotonic()) / 60)
                print(
                    f"{positions} positions, {skipped} script buckets, "
                    f"{remaining:.1f} min left"
                )

    if not positions:
        raise RuntimeError("No eligible chip decisions sampled. Increase --minutes.")

    skipped = save_policy(buckets, args)
    print(
        f"Saved {positions} positions across {len(buckets)} buckets. "
        f"Learned to skip the chip in {skipped} buckets."
    )


def evaluate_chip_selector(args):
    rows = []
    for name in OPPONENTS:
        for label, team in (
            ("team2", team2),
            ("fixed_chip", FIXED_CHIP),
            ("learned_chip", team2_chip),
        ):
            games = []
            for index in range(args.games // 2):
                for swap in (False, True):
                    seed = args.eval_seed + index
                    result = play(team, name, seed=seed, swap=swap)
                    ours, theirs = ("blue", "red") if swap else ("red", "blue")
                    games.append({
                        "seed": seed,
                        "side": ours,
                        "gf": result["score"][ours],
                        "ga": result["score"][theirs],
                        "strikes": result["strikes"][ours],
                        "forfeit": result["forfeit"],
                    })

            wins = sum(g["gf"] > g["ga"] for g in games)
            draws = sum(g["gf"] == g["ga"] for g in games)
            losses = len(games) - wins - draws
            gf = sum(g["gf"] for g in games)
            ga = sum(g["ga"] for g in games)
            print(f"{label} vs {name}: {wins}-{draws}-{losses} W-D-L, {gf}-{ga} goals")
            rows.append({"team": label, "opponent": name, "games": games})

    output = ROOT / "data"
    output.mkdir(exist_ok=True)
    (output / f"chip_results_seed{args.seed}.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=55)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--per-match", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=6000)
    args = parser.parse_args()

    if args.minutes <= 0 or args.trials < 1 or args.per_match < 1:
        parser.error("minutes, trials, and per-match must be positive")
    if not 1 <= args.horizon <= 300 or args.games < 2 or args.games % 2:
        parser.error("horizon must be 1–300 and games must be even and at least 2")

    train_chip_selector(args)
    evaluate_chip_selector(args)