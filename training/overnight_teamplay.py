"""Search four team-play tactics, then test combinations on fresh matches."""

import argparse
import json
import random
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arena"))

from grid2v2.agents import ScriptedAgent
from grid2v2.match import load_team, make_env, play
from grid2v2.physics import Ball
from teams import team2_teamplay as bot

TRAIN = ("baselines.scripted", "baselines.scripted_box", "baselines.learned",
         "two_field_box", "p05414", "territory")
FINAL = (*TRAIN, "teams.team1", "two_field_fine")
CONTROL = (bot.DEFAULT, bot.TERRITORY)

# Positions in the candidate team's frame. The owner is an index into the four positions.
POSITIONS = {
    "outlet": (((5, 5), (7, 6), (6, 5), (11, 4)), 0),
    "lead": (((7, 4), (9, 6), (8, 4), (12, 5)), 0),
    "wall": (((8, 1), (10, 3), (9, 1), (9, 2)), 0),
    "blocked": (((8, 5), (11, 5), (9, 5), (12, 6)), 0),
    "double_press": (((6, 5), (2, 5), (7, 6), (6, 4)), 2),
    "turnover": (((12, 7), (8, 4), (7, 6), (9, 5)), 2),
    "keeper": (((6, 7), (1, 5), (3, 5), (8, 4)), 1),
    "keeper_wide": (((5, 7), (2, 3), (4, 3), (5, 4)), 1),
}
SCHEDULE = ("passing", "defense", "passing", "clearance", "defense",
            "passing", "restart", "clearance", "defense", "passing")


def team(genome, trace=None):
    module = ModuleType("teamplay_candidate")
    module.TIER = bot.TIER
    module.make_team = lambda side, ids, env: bot.build_team(side, ids, env, genome, trace)
    return module


@cache
def opponent(name):
    if name == "p05414":
        return team(bot.DEFAULT)
    if name == "territory":
        return team(bot.TERRITORY)
    if not name.startswith("two_field_"):
        return load_team(name)
    module = ModuleType(name)
    module.TIER = name.removeprefix("two_field_")
    module.make_team = lambda side, ids, env: {
        a: ScriptedAgent(env.adapters[a], env.rink, index=0) for a in ids
    }
    return module


def short_game(candidate, rival, seed, swap, scenario):
    side, other = ("blue", "red") if swap else ("red", "blue")
    modules = {side: candidate, other: rival}
    env = make_env(modules["red"], modules["blue"])
    env.reset(seed=seed)
    concession = int(scenario == "restart")
    if concession:
        env.score[other] = 1
        env._kickoff(side)
    else:
        positions, owner = POSITIONS[scenario]
        ids = env.members[side] + env.members[other]
        cells = [(env.rink.width - 1 - x if swap else x, y) for x, y in positions]
        env.pos = dict(zip(ids, cells))
        env.holder = ids[owner]
        env.ball = Ball.at_cell(*cells[owner])
    agents = {}
    for color, module in modules.items():
        agents.update(module.make_team(color, env.members[color], env))
    for agent in agents.values():
        agent.reset()
    for _ in range(24):
        actions = {a: agents[a].act(env.observe(a)) for a in env.agents}
        env.step(actions)
    gf, ga = env.score[side], env.score[other] - concession
    x = env.ball.x if not swap else env.rink.width - 1 - env.ball.x
    holder_side = env.team[env.holder] if env.holder is not None else None
    possession = 0.0
    if holder_side == side:
        possession = 1 + x / env.rink.width
    elif holder_side == other:
        possession = -2 + x / env.rink.width
    return gf, ga, 8 * (gf - ga) + possession


def match_job(job):
    genome, case, deadline = job
    if time.monotonic() >= deadline:
        return None
    name, seed, swap, scenario = case
    trace = [Counter(), Counter()]
    candidate, rival = team(genome, trace), opponent(name)
    if scenario:
        gf, ga, reward = short_game(candidate, rival, seed, swap, scenario)
        strikes, forfeit, slowest = 0, None, 0.0
    else:
        result = play(candidate, rival, seed=seed, swap=swap)
        side, other = ("blue", "red") if swap else ("red", "blue")
        gf, ga = result["score"][side], result["score"][other]
        strikes = result["strikes"][side]
        slowest = result["slowest_act_ms"][side]
        forfeit, reward = result["forfeit"], 0.0
    points = 3 if gf > ga else int(gf == ga)
    return genome, case, {
        "points": points, "gf": gf, "ga": ga, "scenario_score": reward,
        "strikes": strikes, "forfeit": forfeit, "slowest_ms": slowest,
        "changes": dict(trace[0] + trace[1]),
    }


def deck(names, seed, count):
    return [(name, s, swap, None) for name in names for s in range(seed, seed + count)
            for swap in (False, True)]


def evaluate(pool, genomes, cases, deadline):
    results = {g: [] for g in dict.fromkeys(genomes)}
    jobs = ((g, case, deadline) for case in cases for g in results)
    for item in pool.map(match_job, jobs, chunksize=2):
        if item is None:
            return None
        genome, case, row = item
        results[genome].append({"case": case, **row})
    return results


def summarize(rows):
    counts = Counter()
    for row in rows:
        counts.update(row["changes"])
    return dict(
        games=len(rows), points=sum(r["points"] for r in rows),
        wdl=[sum(r["points"] == k for r in rows) for k in (3, 1, 0)],
        gf=sum(r["gf"] for r in rows), ga=sum(r["ga"] for r in rows),
        strikes=sum(r["strikes"] for r in rows), forfeits=sum(bool(r["forfeit"]) for r in rows),
        scenario_score=round(sum(r["scenario_score"] for r in rows), 3),
        slowest_ms=max((r["slowest_ms"] for r in rows), default=0.0),
        changes=dict(counts),
    )


def rank(row, scenarios=False):
    primary = row["scenario_score"] if scenarios else row["points"]
    return (-row["forfeits"], primary, row["gf"] - row["ga"], -row["strikes"])


def active_families(genome):
    return [name for name, slots in bot.FAMILIES.items() if genome[slots.start] != "current"]


def propose(rng, slots, parents, progress):
    genome = list(bot.DEFAULT)
    if not parents or rng.random() < 0.65 - 0.40 * progress:
        genome[slots] = [rng.choice(bot.OPTIONS[i]) for i in range(slots.start, slots.stop)]
    else:
        genome = list(rng.choice(parents))
        if rng.random() < 0.35:
            other = rng.choice(parents)
            for i in range(slots.start, slots.stop):
                if rng.random() < 0.5:
                    genome[i] = other[i]
        for i in rng.sample(range(slots.start, slots.stop), rng.randint(1, len(genome[slots]))):
            genome[i] = rng.choice(bot.OPTIONS[i])
    return bot.normalize(genome)


def combine(rng, qualified):
    genome = list(bot.DEFAULT)
    for family in rng.sample(list(qualified), rng.randint(2, len(qualified))):
        slots = bot.FAMILIES[family]
        genome[slots] = rng.choice(qualified[family])[slots]
        if rng.random() < 0.55:
            i = rng.randrange(slots.start + 1, slots.stop)
            genome[i] = rng.choice(bot.OPTIONS[i])
    return bot.normalize(genome)


def components(genome):
    pieces = []
    for name in active_families(genome):
        slots = bot.FAMILIES[name]
        piece = list(bot.DEFAULT)
        piece[slots] = genome[slots]
        pieces.append(tuple(piece))
    return pieces


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def search(pool, out, state, start, deadline, stage, batch):
    archive = {k: tuple(g) for k, g in state["archive"].items()}
    seen = {tuple(g) for g in state["seen"]}
    qualified = {key: [tuple(g) for g in gs] for key, gs in state["qualified"].items()}
    smoke = state["smoke"]
    last_report = time.monotonic()
    with (out / "search.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        while time.monotonic() < deadline and (not smoke or state["round"] < 2):
            round_id = state["round"]
            rng = random.Random(state["seed"] + 97 * round_id + 100_000 * state["phase"])
            progress = min(1.0, (time.monotonic() - start) / (0.875 * state["budget"]))
            proposals = {}
            for i in range(batch * 32):
                if stage == "combine":
                    genome = combine(rng, qualified)
                    key = "combo/" + "+".join(active_families(genome))
                else:
                    family = SCHEDULE[(round_id + i) % len(SCHEDULE)]
                    slots = bot.FAMILIES[family]
                    parents = [g for key, g in archive.items() if key.startswith(family + "/")]
                    genome = propose(rng, slots, parents, progress)
                    key = family + "/" + genome[slots.start]
                if genome in CONTROL or genome in seen or genome in proposals:
                    continue
                proposals[genome] = key
                if len(proposals) == batch:
                    break
            if not proposals:
                break

            seed = state["seed"] + state["phase"] * 3_000_000 + 3 * round_id
            if stage == "scenarios":
                cases = [(name, seed, swap, case) for case in (*POSITIONS, "restart")
                         for name in ("two_field_box", "p05414") for swap in (False, True)]
            else:
                cases = deck(TRAIN, seed, 1 if smoke else 2)
            incumbents = [archive[key] for key in dict.fromkeys(proposals.values()) if key in archive]
            rows = evaluate(pool, [*CONTROL, *proposals, *incumbents], cases, deadline)
            if rows is None:
                break
            scores = {g: summarize(rows[g]) for g in rows}
            for genome, key in proposals.items():
                row = scores[genome]
                old = archive.get(key)
                if not row["forfeits"] and (old is None or rank(
                    row, stage == "scenarios"
                ) > rank(scores[old], stage == "scenarios")):
                    archive[key] = genome
                log.write(json.dumps({
                    "stage": stage, "batch": round_id + 1, "family": key,
                    "genome": genome, "score": row, "p05414": scores[bot.DEFAULT],
                    "territory": scores[bot.TERRITORY],
                }) + "\n")
            seen.update(proposals)
            state["searched"] += len(proposals)
            state["round"] += 1
            if round_id == 0 or smoke or time.monotonic() - last_report >= 90:
                last_report = time.monotonic()
                print(f"{stage}: {state['searched']} proposals total, {len(archive)} archive slots.")
                checkpoint(out, state, start, archive, seen)
    checkpoint(out, state, start, archive, seen)
    return archive


def checkpoint(out, state, start, archive=None, seen=None):
    state["elapsed"] = time.monotonic() - start
    if archive is not None:
        state["archive"] = archive
    if seen is not None:
        state["seen"] = sorted(seen)
    save(out / "checkpoint.json", state)


def shortlist(pool, genomes, seed, count, deadline):
    if not genomes:
        return [], {}
    rows = evaluate(pool, [*CONTROL, *genomes], deck(TRAIN, seed, count), deadline)
    if rows is None:
        return [], {}
    scores = {g: summarize(rr) for g, rr in rows.items()}
    ranked = sorted((g for g in genomes if not scores[g]["forfeits"]),
                    key=lambda g: rank(scores[g]), reverse=True)
    return ranked, scores


def run(pool, args, out, state):
    start = time.monotonic() - state["elapsed"]
    budget, smoke, seed = state["budget"], state["smoke"], state["seed"]
    batch = min(args.batch, 5) if smoke else args.batch
    if state["phase"] == 0:
        search(pool, out, state, start, start + budget * 0.20, "scenarios", batch)
        state.update(phase=1, round=0, seen=[])
        checkpoint(out, state, start)
    if state["phase"] == 1:
        search(pool, out, state, start, start + budget * 0.70, "matches", batch)
        genomes = list(dict.fromkeys(tuple(g) for g in state["archive"].values()))
        ranked, scores = shortlist(pool, genomes, seed + 17_000_000,
                                   1 if smoke else 8, start + budget * 0.74)
        qualified = {}
        if scores:
            for g in ranked:
                family = active_families(g)[0]
                if rank(scores[g]) > rank(scores[bot.DEFAULT]):
                    qualified.setdefault(family, []).append(g)
            if rank(scores[bot.TERRITORY]) >= rank(scores[bot.DEFAULT]):
                qualified.setdefault("clearance", []).insert(0, bot.TERRITORY)
        state.update(phase=2, round=0, seen=[], qualified={
            key: values[:2] for key, values in qualified.items()
        }, singles=ranked[:4] or genomes[:4], archive={})
        checkpoint(out, state, start)
    if state["phase"] == 2:
        if len(state["qualified"]) >= 2:
            search(pool, out, state, start, start + budget * 0.85, "combine", batch)
        combos = list(dict.fromkeys(tuple(g) for g in state["archive"].values()))
        parts = list(dict.fromkeys(part for g in combos for part in components(g)))
        combo_rank, scores = shortlist(pool, combos + parts, seed + 18_000_000,
                                       1 if smoke else 8, start + budget * 0.875)
        comparisons = [
            {"genome": g, "combined": scores[g],
             "parts": [{"genome": part, "score": scores[part]} for part in components(g)]}
            for g in combo_rank if g in combos
        ]
        save(out / "combination_checks.json", comparisons)
        combos_to_test = [
            row["genome"] for row in comparisons if all(
                rank(row["combined"]) > rank(part["score"]) for part in row["parts"]
            )
        ]
        choices = list(dict.fromkeys([*(tuple(g) for g in state["singles"]), *combos_to_test[:2]]))
        state.update(phase=3, finalists=choices[:2] if smoke else choices, final_blocks=[])
        checkpoint(out, state, start)

    finalists = [tuple(g) for g in state["finalists"]]
    for block in range(len(state["final_blocks"]), 2):
        if not finalists:
            break
        cases = deck(FINAL, seed + 20_000_000 + 10_000 * block, 1 if smoke else 48)
        candidates = list(dict.fromkeys([*CONTROL, *finalists]))
        print(f"Final block {block + 1}: {len(cases) * len(candidates)} matches.")
        rows = evaluate(pool, candidates, cases, start + budget)
        if rows is None:
            break
        report = []
        for genome in candidates:
            matches = rows[genome]
            report.append({
                "genome": genome, "overall": summarize(matches),
                "opponents": {name: summarize([r for r in matches if r["case"][0] == name])
                              for name in FINAL},
            })
            totals = report[-1]["overall"]
            label = {bot.DEFAULT: "p05414", bot.TERRITORY: "territory"}.get(genome)
            if label is None:
                label = f"Finalist {len(report) - 2} ({', '.join(active_families(genome))})"
            print(f"{label}: {totals['wdl']} W/D/L, {totals['points']} points.")
        save(out / f"final_matches_{block + 1}.json", [
            {"genome": g, "matches": rows[g]} for g in candidates
        ])
        state["final_blocks"].append(report)
        checkpoint(out, state, start)

    comparisons = []
    for i, genome in enumerate(finalists, 2):
        blocks = state["final_blocks"]
        gains = [[block[i]["overall"]["points"] - block[j]["overall"]["points"]
                  for j in range(2)] for block in blocks]
        clean = all(not (report["overall"]["forfeits"] or report["overall"]["strikes"])
                    for block in blocks for report in (block[0], block[1], block[i]))
        scripted = [block[i]["opponents"]["baselines.scripted"]["wdl"] for block in blocks]
        totals = [sum(b[j] for b in gains) for j in range(2)]
        comparisons.append(dict(genome=genome, point_gain_by_block=gains, scripted_wdl=scripted,
                                clean=clean, total_gain=totals,
                                repeated_gain=len(gains) == 2 and all(min(b) > 0 for b in gains)))
    passing = [row for row in comparisons if row["clean"] and len(state["final_blocks"]) == 2
               and min(row["total_gain"]) > 0]
    winner = max(passing, key=lambda row: min(row["total_gain"])) if passing else None
    if winner and not smoke:
        genome = winner["genome"]
        save(out / "recommended_policy.json", {
            "genome": genome, "tactics": dict(zip(bot.NAMES, genome)),
        })
    save(out / "summary.json", dict(
        searched=state["searched"], minutes=(time.monotonic() - start) / 60,
        final_blocks=state["final_blocks"], comparisons=comparisons,
        recommendation=None if smoke or winner is None else winner["genome"],
    ))
    if smoke:
        if len(state["final_blocks"]) != 2 or any(
            row["overall"]["forfeits"] or row["overall"]["strikes"]
            for block in state["final_blocks"] for row in block
        ):
            raise RuntimeError("Smoke test did not complete cleanly.")
        print("Smoke test passed.")
    elif winner:
        print(f"Exported: {out / 'recommended_policy.json'}")
    print(f"Summary: {out / 'summary.json'}")


def benchmark():
    genes = list(bot.DEFAULT)
    genes[0], genes[4] = "lead", "goal_side"
    cases = deck(("baselines.learned", "two_field_box", "p05414"), 274800, 8)
    for workers in (12, 16, 20, 24):
        start = time.monotonic()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            result = evaluate(pool, [*CONTROL, tuple(genes)], cases, start + 180)
        if result is None:
            print(f"{workers} workers: benchmark hit its time limit.")
            continue
        reports = [summarize(rows) for rows in result.values()]
        games = sum(row["games"] for row in reports)
        strikes = sum(row["strikes"] for row in reports)
        forfeits = sum(row["forfeits"] for row in reports)
        print(f"{workers} workers: {games / (time.monotonic() - start):.1f} games/sec, "
              f"{strikes} strikes, {forfeits} forfeits.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=112401)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not 0 < args.hours < float("inf") or min(args.workers, args.batch) < 1 or args.seed < 0:
        parser.error("Use positive hours, workers and batch size, and a nonnegative seed.")
    if args.benchmark:
        benchmark()
        return
    if args.resume:
        out = args.resume.resolve()
        state = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
    else:
        data = ROOT / "data"
        data.mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="teamplay_overnight_", dir=data))
        state = dict(phase=0, round=0, searched=0, elapsed=0, archive={}, seen=[], qualified={},
                     seed=args.seed, smoke=args.smoke_test, budget=900 if args.smoke_test else args.hours * 3600)
    print(f"Results folder: {out}")
    print(f"Match workers: {args.workers}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        run(pool, args, out, state)


if __name__ == "__main__":
    main()