"""Explore tactical combinations, keep distinct styles, and test finalists."""

import argparse
from collections import Counter
import csv
import json
from math import isfinite, prod
from pathlib import Path
from random import Random
import sys
from tempfile import mkdtemp
from time import monotonic
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arena"))

from grid2v2.match import play
from teams import team2_explore as soccer

LEARNED = "baselines.learned"
BOX = "baselines.scripted_box"
MIRROR = "teams.team2_chip"
TRAIN_OPPONENTS = (LEARNED, BOX)
TEST_OPPONENTS = (LEARNED, BOX, MIRROR)
DEFAULT = soccer.DEFAULT
CHANGES = ("kick_changes", "chase_changes", "restart_changes")
MIN_LEARNED_GAIN = 0.10
SECONDARY_TOLERANCE = 0.10


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def as_team(genome, trace):
    module = ModuleType("explore_candidate")
    module.TIER = soccer.TIER

    def build(side, ids, env):
        return soccer.build_team(side, ids, env, genome, trace)

    module.make_team = build
    return module


class Experiment:
    def __init__(self, log):
        self.log = log
        self.writer = csv.writer(log)
        self.ids = {}
        self.writer.writerow([
            "stage", "policy", "opponent", "seed", "side", "goals_for",
            "goals_against", "points", "strikes", "forfeit", "chips", "wide",
            *CHANGES,
        ])

    def policy_id(self, genome):
        if genome not in self.ids:
            self.ids[genome] = f"p{len(self.ids):05d}"
        return self.ids[genome]

    def evaluate(self, genome, stage, opponents, start, seeds, deadline):
        results = {}
        pid = self.policy_id(genome)

        for opponent in opponents:
            counts = Counter()

            for seed in range(start, start + seeds):
                for swap in (False, True):
                    if monotonic() >= deadline:
                        return None

                    trace = Counter()
                    match = play(
                        as_team(genome, trace), opponent, seed=seed, swap=swap
                    )
                    us, them = ("blue", "red") if swap else ("red", "blue")
                    gf, ga = match["score"][us], match["score"][them]
                    points = 3 if gf > ga else int(gf == ga)
                    strikes = match["strikes"][us]

                    counts.update(
                        games=1,
                        points=points,
                        gf=gf,
                        ga=ga,
                        wins=gf > ga,
                        draws=gf == ga,
                        losses=gf < ga,
                        strikes=strikes,
                        forfeits=bool(match["forfeit"]),
                    )
                    counts.update(trace)
                    self.writer.writerow([
                        stage, pid, opponent, seed, us, gf, ga, points, strikes,
                        match["forfeit"], trace["chips"], trace["wide"],
                        *(trace[name] for name in CHANGES),
                    ])
                    self.log.flush()

            results[opponent] = counts

        return results


def quality(results):
    learned, box = results[LEARNED], results[BOX]
    return learned["points"], box["points"], learned["gf"] - learned["ga"]


def clean(results):
    return all(row["forfeits"] == 0 for row in results.values())


def niche(results):
    chips = sum(row["chips"] for row in results.values())
    wide = sum(row["wide"] for row in results.values())
    games = sum(row["games"] for row in results.values())
    pace = chips / games
    width = wide / max(1, chips)

    return (
        int(pace >= 4) + int(pace >= 9),
        int(width >= 0.20) + int(width >= 0.45),
    )


def changed(results):
    return sum(row[name] for row in results.values() for name in CHANGES)


def random_genome(rng):
    return tuple(rng.choice(options) for options in soccer.GENE_OPTIONS)


def screen_genomes(rng):
    seeded = [
        (gene, gene, DEFAULT[2], DEFAULT[3]) for gene in soccer.GENES
    ]
    seeded += [
        (*DEFAULT[:2], f"{style}/y_first", DEFAULT[3])
        for style in soccer.STYLES
        if style not in ("current", "carry")
    ]
    seeded += [(*DEFAULT[:3], opening) for opening in soccer.RESTARTS]
    random_seeds = [random_genome(rng) for _ in range(32)]
    return list(dict.fromkeys([DEFAULT, *seeded, *random_seeds]))


def mutate(rng, parent, size):
    child = list(parent)
    for slot in rng.sample(range(len(parent)), size):
        child[slot] = rng.choice([
            gene for gene in soccer.GENE_OPTIONS[slot] if gene != child[slot]
        ])
    return tuple(child)


def checkpoint(output, experiment, archive, screened, searched, duplicates, proposals):
    save(output / "checkpoint.json", {
        "screened": screened,
        "searched": searched,
        "proposals_without_changed_actions": duplicates,
        "search_sources": dict(proposals),
        "policies": {pid: genome for genome, pid in experiment.ids.items()},
        "archive": [
            {
                "cell": cell,
                "policy": experiment.policy_id(genome),
                "genome": genome,
                "results": scores,
            }
            for cell, (genome, scores) in sorted(archive.items())
        ],
    })


def recommend(final, finalists):
    if any(DEFAULT not in block or not clean(block[DEFAULT]) for block in final):
        return None

    ranked = []
    for genome in finalists:
        if any(genome not in block or not clean(block[genome]) for block in final):
            continue

        learned_gains = [
            block[genome][LEARNED]["points"] - block[DEFAULT][LEARNED]["points"]
            for block in final
        ]
        games = sum(block[genome][LEARNED]["games"] for block in final)
        secondary_gains = [
            sum(
                block[genome][opp]["points"] - block[DEFAULT][opp]["points"]
                for block in final
            )
            for opp in (BOX, MIRROR)
        ]

        if (
            min(learned_gains) > 0
            and sum(learned_gains) >= MIN_LEARNED_GAIN * games
            and min(secondary_gains) >= -SECONDARY_TOLERANCE * games
        ):
            ranked.append((sum(learned_gains), sum(secondary_gains), genome))

    return max(ranked)[2] if ranked else None


def stage(experiment, genomes, label, opponents, first_seed, seeds, deadline):
    results = {}

    for genome in dict.fromkeys(genomes):
        scores = experiment.evaluate(
            genome, label, opponents, first_seed, seeds, deadline
        )
        if scores is None:
            break

        results[genome] = scores
        print(
            f"{label} {experiment.policy_id(genome)}: "
            + ", ".join(f"{name} {row['points']} pts" for name, row in scores.items())
            + f", {changed(scores)} changed actions",
            flush=True,
        )

    return results


def run(args, output, experiment):
    started = monotonic()
    budget = 600 if args.smoke_test else args.hours * 3600
    search_end = started + budget * 7 / 8
    validation_end = started + budget * 11 / 12
    finish = started + budget
    screen_seeds, search_seeds, validation_seeds, final_seeds = (
        (1, 1, 1, 1) if args.smoke_test else (4, 4, 16, 25)
    )

    rng = Random(args.seed)
    archive = {}
    candidates = screen_genomes(rng)
    if args.smoke_test:
        candidates = [
            DEFAULT, candidates[6], candidates[11], candidates[-7], candidates[-2]
        ]

    screening = stage(
        experiment, candidates, "screen", TRAIN_OPPONENTS, 400000,
        screen_seeds, min(search_end, started + budget / 8),
    )
    if DEFAULT not in screening:
        raise RuntimeError("The baseline did not finish screening.")

    save(output / "screening.json", {
        experiment.policy_id(genome): scores for genome, scores in screening.items()
    })

    tested = set(screening)
    possibilities = prod(len(options) for options in soccer.GENE_OPTIONS)
    duplicates = 0
    proposals = Counter()

    for genome, scores in screening.items():
        if genome != DEFAULT and not changed(scores):
            duplicates += 1
            continue

        if clean(scores):
            cell = niche(scores)
            if cell not in archive or quality(scores) > quality(archive[cell][1]):
                archive[cell] = genome, scores

    screened, searched = len(screening), 0
    checkpoint(output, experiment, archive, screened, searched, duplicates, proposals)
    next_save = monotonic() + 300

    while (
        monotonic() < search_end
        and len(tested) < possibilities
        and (not args.smoke_test or searched < 3)
    ):
        fraction = min(1.0, (monotonic() - started) / (search_end - started))
        fresh_probability = 0.60 - 0.45 * fraction

        genome = random_genome(rng)
        source = "random"
        if archive and rng.random() >= fresh_probability:
            parents = [entry[0] for entry in archive.values()]
            parent = rng.choice([*parents, DEFAULT])
            size = 1 if fraction > 0.70 else rng.choice((2, 3))
            genome = mutate(rng, parent, size)
            source = "mutation"

        if genome in tested:
            continue

        tested.add(genome)
        first_seed = 500000 + searched * search_seeds
        scores = experiment.evaluate(
            genome, "search", TRAIN_OPPONENTS, first_seed, search_seeds, search_end,
        )
        if scores is None:
            break

        searched += 1
        proposals[source] += 1

        if not changed(scores):
            duplicates += 1
        elif clean(scores):
            cell = niche(scores)

            if cell not in archive:
                archive[cell] = genome, scores
            else:
                incumbent, _ = archive[cell]
                reference = experiment.evaluate(
                    incumbent, "search_incumbent", TRAIN_OPPONENTS,
                    first_seed, search_seeds, search_end,
                )
                if (
                    reference is not None
                    and clean(reference)
                    and quality(scores) > quality(reference)
                ):
                    archive[cell] = genome, scores

        if monotonic() >= next_save:
            checkpoint(
                output, experiment, archive, screened, searched, duplicates, proposals
            )
            print(
                f"Search: {searched} proposals, {len(archive)}/9 styles, "
                f"{duplicates} proposals with zero changed actions.",
                flush=True,
            )
            next_save = monotonic() + 300

    checkpoint(output, experiment, archive, screened, searched, duplicates, proposals)

    contenders = [genome for genome, _ in archive.values() if genome != DEFAULT]
    validation = stage(
        experiment, [DEFAULT, *contenders], "validation", TEST_OPPONENTS,
        2000000, validation_seeds, validation_end,
    )
    finalists = sorted(
        (g for g in validation if g != DEFAULT and clean(validation[g])),
        key=lambda g: quality(validation[g]),
        reverse=True,
    )[:2]

    final = [
        stage(
            experiment, [DEFAULT, *finalists], f"final_{block + 1}",
            TEST_OPPONENTS, 3000000 + 1000 * block, final_seeds, finish,
        )
        for block in range(2)
    ]
    complete = DEFAULT in validation and all(
        all(g in block for g in [DEFAULT, *finalists]) for block in final
    )
    winner = recommend(final, finalists) if complete and not args.smoke_test else None

    def with_ids(results):
        return {experiment.policy_id(g): scores for g, scores in results.items()}

    save(output / "summary.json", {
        "complete": complete,
        "smoke_test": args.smoke_test,
        "screened": screened,
        "searched": searched,
        "archive_cells": len(archive),
        "proposals_without_changed_actions": duplicates,
        "policies": {pid: g for g, pid in experiment.ids.items()},
        "baseline": experiment.policy_id(DEFAULT),
        "validation": with_ids(validation),
        "finalists": [experiment.policy_id(g) for g in finalists],
        "final": [with_ids(block) for block in final],
        "recommended": experiment.policy_id(winner) if winner else None,
    })

    if winner:
        save(output / "recommended_policy.json", {
            "genome": winner,
            "tactics": dict(zip(soccer.SLOTS, winner)),
        })
        print(f"Recommended {experiment.policy_id(winner)}. See recommended_policy.json.")
    elif args.smoke_test and complete:
        print("Smoke test complete. See summary.json.")
    elif not complete:
        raise RuntimeError(
            "Final evaluation incomplete. Check summary.json and checkpoint.json."
        )
    else:
        print("Keep Team 2 Chip. No finalist qualified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if not isfinite(args.hours) or args.hours < 1:
        parser.error("--hours must be a finite number of at least 1")

    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    output = Path(mkdtemp(prefix="explore_overnight_", dir=data))
    print(f"Results folder: {output}", flush=True)
    save(output / "config.json", vars(args))

    with (output / "matches.csv").open("w", newline="", encoding="utf-8") as log:
        run(args, output, Experiment(log))