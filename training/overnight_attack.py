"""Train two attack policies and evaluate them on fresh games overnight."""

import argparse
from collections import Counter, deque
import csv
import json
from math import isfinite
from pathlib import Path
import sys
from tempfile import TemporaryDirectory, mkdtemp
from time import monotonic
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arena"))

from grid2v2 import Ball, Grid2v2
from grid2v2.match import load_team, play, tier_of
from teams import team2_attack as attack, team2_chip

LEARNED = "baselines.learned"
BOX = "baselines.scripted_box"
MIRROR = "teams.team2_chip"

POOLS = {
    "staff_only": (LEARNED, BOX),
    "with_self_play": (LEARNED, BOX, MIRROR),
}
OPPONENTS = POOLS["with_self_play"]
HORIZON = 18
BOX_TOLERANCE = 0.10


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def variant(weights=None, samples=None, counts=None):
    team = ModuleType("attack_candidate")
    team.TIER = attack.TIER

    def build(side, ids, env):
        return attack.build_team(side, ids, env, weights, samples, counts)

    team.make_team = build
    return team


class ReplayPositions:
    def __init__(self, opponent, seed, replay):
        self.opponent = opponent
        self.seed = seed
        self.replay = replay
        self.rng = np.random.default_rng(seed)
        self.pending = deque()
        self.seen = set()
        self.forfeits = 0

    def take(self, deadline):
        while not self.pending and monotonic() < deadline:
            samples = {}
            result = play(
                variant(samples=samples),
                self.opponent,
                seed=self.seed,
                log=self.replay,
            )
            self.seed += 1

            if result["forfeit"]:
                self.forfeits += 1
                print(
                    f"Skipping collection forfeit vs {self.opponent}: "
                    f"{result['forfeit']}",
                    flush=True,
                )
                if self.forfeits >= 3:
                    raise RuntimeError(
                        f"Three consecutive collection forfeits vs {self.opponent}."
                    )
                continue
            self.forfeits = 0

            data = json.loads(self.replay.read_text(encoding="utf-8"))
            times = [t for t in samples if t <= data["max_steps"] - HORIZON]
            self.rng.shuffle(times)

            for t in times:
                frame = data["frames"][t]
                signature = (
                    tuple(tuple(cell) for cell in frame["pos"].values()),
                    tuple(sorted(frame["stunned"].items())),
                )
                if signature in self.seen:
                    continue

                self.seen.add(signature)
                self.pending.append((frame, *samples[t]))
                if len(self.pending) == 8:
                    break

        return self.pending.popleft() if self.pending else None


def make_simulator(name):
    opponent = load_team(name)
    arena = Grid2v2(tiers={"red": attack.TIER, "blue": tier_of(opponent)})
    bots = attack.build_team("red", arena.members["red"], arena)
    bots.update(opponent.make_team("blue", arena.members["blue"], arena))
    return arena, bots


def rollout(simulator, case, seed, plan):
    arena, bots = simulator
    frame, _, _, field_rng = case

    arena.reset(seed=int(seed))
    arena.t = frame["t"]
    arena.score.update(frame["score"])
    arena.pos.update({a: tuple(cell) for a, cell in frame["pos"].items()})
    arena.stunned.update(frame["stunned"])
    arena.holder = "red_0"
    arena.ball = Ball.at_cell(*arena.pos["red_0"])

    for bot in bots.values():
        bot.reset()

    field = bots["red_0"]
    field.policy_rng.bit_generator.state = field_rng
    if plan is not None:
        field.start_plan(plan)

    obs = {a: arena.observe(a) for a in arena.agents}
    for _ in range(HORIZON):
        actions = {a: bots[a].act(obs[a]) for a in arena.agents}
        obs, reward, _, truncated, _ = arena.step(actions)

        if reward["red_0"]:
            return int(reward["red_0"] > 0)
        if arena.holder and arena.team[arena.holder] == "blue":
            return 0
        if truncated["red_0"]:
            break

    return 0


def goals(simulator, case, seeds, plan=None):
    return sum(rollout(simulator, case, seed, plan) for seed in seeds)


class PlanValues:
    def __init__(self):
        size = attack.N_INPUTS
        self.matrix = {plan: 3 * np.eye(size) for plan in attack.PLANS}
        self.target = {plan: np.zeros(size) for plan in attack.PLANS}
        self.samples = Counter()

    def train(self, simulator, case, seeds):
        _, features, options, _ = case
        current = goals(simulator, case, seeds)
        products = np.outer(features, features)

        for plan in options:
            gain = (goals(simulator, case, seeds, plan) - current) / len(seeds)
            self.matrix[plan] += products
            self.target[plan] += features * gain
            self.samples[plan] += 1

    def weights(self):
        return {
            plan: np.linalg.solve(self.matrix[plan], self.target[plan])
            for plan in attack.PLANS
            if self.samples[plan] >= 6
        }


def record(weights, arm, positions):
    return {
        "arm": arm,
        "positions": positions,
        "weights": {name: row.tolist() for name, row in weights.items()},
    }


def train_arms(args, output, simulators, replay, deadline):
    rng = np.random.default_rng(23)
    models = {arm: PlanValues() for arm in POOLS}
    streams = {
        (arm, name): ReplayPositions(
            name, 210000 + 100000 * OPPONENTS.index(name), replay
        )
        for arm, pool in POOLS.items()
        for name in pool
    }
    counts = {arm: Counter() for arm in POOLS}
    positions = 0
    checkpoint = monotonic() + 300

    def snapshot():
        weights = {arm: model.weights() for arm, model in models.items()}
        for arm, fitted in weights.items():
            row = record(fitted, arm, positions)
            row["positions_by_opponent"] = dict(counts[arm])
            save(output / f"{arm}.json", row)

        print(f"Saved both arms at {positions} training positions each.", flush=True)
        return weights

    while monotonic() < deadline:
        pair = {}
        for arm, pool in POOLS.items():
            name = pool[positions % len(pool)]
            case = streams[arm, name].take(deadline)
            if case is None:
                break
            pair[arm] = (name, case)

        if len(pair) != len(POOLS) or monotonic() >= deadline:
            break

        seeds = rng.integers(2**31, size=args.trials)
        for arm, (name, case) in pair.items():
            models[arm].train(simulators[name], case, seeds)
            counts[arm][name] += 1
        positions += 1

        if monotonic() >= checkpoint:
            snapshot()
            checkpoint = monotonic() + 300
        if args.smoke_test and positions >= 12:
            break

    if not positions:
        raise RuntimeError("No paired training positions collected.")

    return snapshot(), positions


def test_positions(args, output, weights, simulators, replay, deadline):
    rng = np.random.default_rng(82)
    total = 2 if args.smoke_test else 60
    trials = 3 if args.smoke_test else 12
    results = {}

    for i, name in enumerate(OPPONENTS):
        stream = ReplayPositions(name, 800000 + i * 10000, replay)
        rows = {arm: Counter() for arm in POOLS}

        for _ in range(total):
            if monotonic() >= deadline:
                break
            case = stream.take(deadline)
            if case is None:
                break

            _, features, options, _ = case
            seeds = rng.integers(2**31, size=trials)
            current = goals(simulators[name], case, seeds)

            for arm, model in weights.items():
                plan = attack.choose_plan(model, features, options)
                scored = goals(simulators[name], case, seeds, plan) if plan else current
                rows[arm].update(
                    positions=1,
                    rollouts=trials,
                    current_goals=current,
                    learned_goals=scored,
                    plans_chosen=bool(plan),
                )

        results[name] = {arm: dict(row) for arm, row in rows.items()}
        print(f"Position test {name}: {results[name]}", flush=True)

    save(output / "position_test.json", results)
    return results, total


def test_matches(args, output, weights, deadline):
    blocks = []
    seeds = 1 if args.smoke_test else args.eval_seeds

    with (output / "matches.csv").open("w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow([
            "block", "arm", "opponent", "seed", "side", "gf", "ga",
            "points", "strikes", "plan_starts", "changed_moves", "forfeit",
        ])

        for block in range(2):
            results = {}
            first_seed = 900000 + 10000 * block

            for arm in ("current", *POOLS):
                results[arm] = {}

                for name in OPPONENTS:
                    scores = Counter()

                    for seed in range(first_seed, first_seed + seeds):
                        for swap in (False, True):
                            if monotonic() >= deadline:
                                break

                            moves = Counter()
                            team = (
                                team2_chip if arm == "current"
                                else variant(weights[arm], counts=moves)
                            )
                            match = play(team, name, seed=seed, swap=swap)
                            us, them = ("blue", "red") if swap else ("red", "blue")
                            gf, ga = match["score"][us], match["score"][them]
                            points = 3 if gf > ga else int(gf == ga)
                            starts = sum(moves[plan] for plan in attack.PLANS)
                            strikes = match["strikes"][us]

                            scores.update(
                                games=1, points=points, wins=gf > ga, draws=gf == ga,
                                losses=gf < ga, gf=gf, ga=ga, strikes=strikes,
                                forfeit=bool(match["forfeit"]), plans=starts,
                                changed_moves=moves["changed_moves"],
                            )
                            writer.writerow([
                                block + 1, arm, name, seed, us, gf, ga, points,
                                strikes, starts, moves["changed_moves"], match["forfeit"],
                            ])
                            log.flush()

                        if monotonic() >= deadline:
                            break

                    results[arm][name] = dict(scores)
                    print(
                        f"Block {block + 1} {arm} vs {name}: {dict(scores)}",
                        flush=True,
                    )

            blocks.append(results)
            save(output / "match_blocks.json", blocks)

    return blocks, 2 * seeds


def select_policy(position_test, cases, blocks, games):
    if any(
        row.get("games", 0) != games or row["forfeit"]
        for block in blocks for row in block["current"].values()
    ):
        return None

    candidates = []
    for arm in POOLS:
        scenarios = position_test[LEARNED][arm]
        if (
            scenarios.get("positions", 0) != cases
            or scenarios["learned_goals"] <= scenarios["current_goals"]
        ):
            continue

        if any(
            row.get("games", 0) != games or row["forfeit"]
            for block in blocks for row in block[arm].values()
        ):
            continue

        differences = [
            block[arm][LEARNED]["points"] - block["current"][LEARNED]["points"]
            for block in blocks
        ]
        box_gain = sum(
            block[arm][BOX]["points"] - block["current"][BOX]["points"]
            for block in blocks
        )

        if min(differences) > 0 and box_gain >= -BOX_TOLERANCE * games * len(blocks):
            candidates.append((sum(differences), box_gain, arm))

    return max(candidates)[2] if candidates else None


def run(args, output):
    start = monotonic()
    budget = 600 if args.smoke_test else 3600 * args.hours
    simulators = {name: make_simulator(name) for name in OPPONENTS}

    with TemporaryDirectory() as folder:
        replay = Path(folder) / "match.json"
        weights, trained = train_arms(
            args, output, simulators, replay, start + budget * 15 / 16
        )
        positions, cases = test_positions(
            args, output, weights, simulators, replay, start + budget * 31 / 32
        )

    blocks, games = test_matches(args, output, weights, start + budget)
    complete = all(
        row.get("positions", 0) == cases
        for opponent in positions.values() for row in opponent.values()
    ) and all(
        row.get("games", 0) == games
        for block in blocks for arm in block.values() for row in arm.values()
    )

    recommended = (
        select_policy(positions, cases, blocks, games)
        if complete and not args.smoke_test else None
    )

    save(output / "summary.json", {
        "smoke_test": args.smoke_test,
        "complete": complete,
        "positions_per_arm": trained,
        "test_games_per_block": games,
        "recommended": recommended,
        "position_test": positions,
        "match_blocks": blocks,
    })

    if recommended:
        save(
            output / "recommended_policy.json",
            record(weights[recommended], recommended, trained),
        )

    status = "Complete" if complete else "Evaluation incomplete"
    print(f"{status}. Recommended arm: {recommended}. Results: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if not isfinite(args.hours) or args.hours < 1 or min(args.trials, args.eval_seeds) < 1:
        parser.error("--hours must be at least 1 and trial and seed counts must be positive")

    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    output = Path(mkdtemp(prefix="attack_overnight_", dir=data))
    print(f"Results folder: {output}", flush=True)
    save(output / "config.json", vars(args))
    run(args, output)