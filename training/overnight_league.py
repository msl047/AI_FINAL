"""Evolve situational tactics against fixed teams and past league opponents."""

import argparse
import json
import math
import random
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path
from types import ModuleType

import overnight_press as prior
from grid2v2.match import make_env, play
from grid2v2.physics import Ball
from teams import team2_league as bot

old = prior.old
TRAIN = ("baselines.scripted", "baselines.learned", "teamplay",
         "box", "fine", "press")
FINAL = (*TRAIN, "baselines.scripted_box", "teams.team1", "wide_two_field")
CONTROLS = ("box", "fine", "teamplay", "press")
ABLATIONS = ("press_no_attack", "press_no_press", "press_no_recover")

# Positions, holder index, loose-ball (x, y, vx, vy), previous opponent holder.
STARTS = {
    "rolling_race": (((5, 4), (3, 7), (7, 5), (8, 7)), None, (8, 5, -2.5, 0), None),
    "wall_intercept": (((5, 2), (3, 4), (7, 2), (6, 5)), None, (7, 1, -2, -0.5), None),
    "far_post": (((5, 4), (3, 7), (6, 6), (4, 3)), 2, None, None),
    "next_dribble": (((7, 4), (3, 5), (6, 4), (6, 7)), 2, None, None),
    "next_receiver": (((5, 6), (3, 3), (5, 4), (4, 6)), 2, None, None),
    "counter_lane": (((8, 5), (10, 6), (9, 4), (7, 6)), 0, None, 3),
    "counter_space": (((6, 6), (8, 7), (5, 6), (7, 5)), 0, None, 2),
}


@cache
def module(key, press_genome):
    if key == "box":
        return prior.opponent("two_field_box")
    if key == "fine":
        return prior.opponent("two_field_fine")
    if key == "teamplay":
        return prior.opponent("teamplay")
    if key == "press":
        return prior.team(press_genome)
    if key in ABLATIONS:
        slots = prior.bot.FAMILIES[key.removeprefix("press_no_")]
        genes = list(press_genome)
        genes[slots] = prior.bot.DEFAULT[slots]
        return prior.team(tuple(genes))
    if isinstance(key, str):
        return prior.opponent(key)
    return candidate(key, press_genome, None)


def candidate(genome, press_genome, trace):
    mod = ModuleType("team2_league")
    mod.TIER = "fine" if genome[0] == "fine" else "box"
    mod.make_team = lambda side, ids, env: bot.build_team(
        side, ids, env, genome, press_genome, trace
    )
    return mod


def scenario_game(our_team, their_team, seed, swap, scene):
    side, other = ("blue", "red") if swap else ("red", "blue")
    teams = {side: our_team, other: their_team}
    env = make_env(teams["red"], teams["blue"])
    env.reset(seed=seed)
    cells, owner, flight, prior_owner = STARTS[scene]
    ids = env.members[side] + env.members[other]
    mirror_y = seed % 2

    def position(x, y):
        return (env.rink.width - 1 - x if swap else x,
                env.rink.height - 1 - y if mirror_y else y)

    env.pos = dict(zip(ids, (position(x, y) for x, y in cells)))
    agents = {}
    for color, mod in teams.items():
        agents.update(mod.make_team(color, env.members[color], env))
    for agent in agents.values():
        agent.reset()

    if prior_owner is not None:
        env.holder = ids[prior_owner]
        env.ball = Ball.at_cell(*env.pos[env.holder])
        for a in env.agents:
            agents[a].act(env.observe(a))
        env.t += 1

    if owner is not None:
        env.holder = ids[owner]
        env.ball = Ball.at_cell(*env.pos[env.holder])
    else:
        env.holder = None
        x, y, vx, vy = flight
        env.ball = Ball.at_cell(*position(x, y))
        env.ball.kick(
            math.atan2(-vy if mirror_y else vy, -vx if swap else vx),
            math.hypot(vx, vy),
        )

    for _ in range(22):
        env.step({a: agents[a].act(env.observe(a)) for a in env.agents})

    gf, ga = env.score[side], env.score[other]
    x = env.ball.x if not swap else env.rink.width - 1 - env.ball.x
    possession = 0.0
    if env.holder is not None:
        possession = (1 if env.team[env.holder] == side else -2) + x / env.rink.width
    return gf, ga, 8 * (gf - ga) + possession


def job(args):
    key, case, press_genome = args
    rival, seed, swap, scene = case
    trace = [Counter(), Counter()]
    own = module(key, press_genome) if isinstance(key, str) else candidate(key, press_genome, trace)
    theirs = module(rival, press_genome)

    if scene is None:
        result = play(own, theirs, seed=seed, swap=swap)
        side, other = ("blue", "red") if swap else ("red", "blue")
        gf, ga = result["score"][side], result["score"][other]
        strikes, forfeit = result["strikes"][side], result["forfeit"]
        peak, reward = result["slowest_act_ms"][side], 0.0
    else:
        gf, ga, reward = scenario_game(own, theirs, seed, swap, scene)
        strikes, forfeit, peak = 0, None, 0.0

    return key, {
        "case": case, "gf": gf, "ga": ga, "points": 3 if gf > ga else int(gf == ga),
        "scenario_score": reward, "strikes": strikes, "forfeit": forfeit,
        "slowest_ms": peak, "changes": dict(trace[0] + trace[1]),
    }


def evaluate(pool, keys, cases, press_genome):
    results = {key: [] for key in dict.fromkeys(keys)}
    jobs = ((key, case, press_genome) for case in cases for key in results)
    for key, row in pool.map(job, jobs, chunksize=2):
        results[key].append(row)
    return results


def active(genome):
    return tuple(name for name, i in bot.FAMILIES.items() if genome[i] != "base")


def propose(rng, archive, seen, n, progress, preferred):
    results = {}
    parents = list(archive.values())
    for _ in range(n * 60):
        if not parents or rng.random() < 0.65 - 0.40 * progress:
            g = [rng.choice(options) for options in bot.OPTIONS]
            if rng.random() < 0.40:
                g[0] = preferred
            if rng.random() < 0.50:
                chosen = rng.choice(list(bot.FAMILIES.values()))
                for i in bot.FAMILIES.values():
                    if i != chosen:
                        g[i] = "base"
        else:
            g = list(rng.choice(parents))
            if rng.random() < 0.35:
                other = rng.choice(parents)
                for start, stop in ((1, 4), (4, 7), (7, 10)):
                    if rng.random() < 0.5:
                        g[start:stop] = other[start:stop]
            for i in rng.sample(range(len(g)), rng.randint(1, 4 if progress < 0.6 else 2)):
                g[i] = rng.choice([v for v in bot.OPTIONS[i] if v != g[i]])

        genome = bot.normalize(g)
        if active(genome) and genome not in seen:
            results[genome] = (genome[0], *active(genome))
        if len(results) == n:
            break
    return results


def checkpoint(out, state):
    old.save(out / "checkpoint.json", state)


def search(pool, out, state, batch):
    budget = state["search_budget"]
    archive = {tuple(slot): tuple(g) for slot, g in state["archive"]}
    seen = {tuple(g) for g in state["seen"]}
    hall = [tuple(g) for g in state["hall"]]
    press_genome = tuple(state["press_genome"])
    smoke = state["smoke"]
    last_status = time.monotonic()

    with (out / "search.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        while state["elapsed"] < budget and (not smoke or state["round"] < 3):
            started = time.monotonic()
            r = state["round"]
            seed = state["seed"] + 5 * r
            rng = random.Random(seed)
            progress = min(1.0, state["elapsed"] / budget)
            forced = r < 2 if smoke else progress < 0.14 or r % 12 == 0
            candidates = propose(rng, archive, seen, batch, progress, state["preferred"])
            if not candidates:
                break

            if forced:
                scenes = list(STARTS) if smoke else rng.sample(list(STARTS), 5)
                cases = [
                    (rival, seed + parity, swap, scene) for scene in scenes
                    for rival in ("box", "press") for swap in (False, True)
                    for parity in (0, 1)
                ]
            else:
                rivals = (*TRAIN, rng.choice(hall)) if hall else TRAIN
                cases = old.deck(rivals, seed, 1 if smoke else 2)

            incumbents = [archive[slot] for slot in dict.fromkeys(candidates.values()) if slot in archive]
            rows = evaluate(pool, [*CONTROLS, *candidates, *incumbents], cases, press_genome)
            scores = {key: old.summarize(games) for key, games in rows.items()}
            for g, slot in candidates.items():
                row, elite = scores[g], archive.get(slot)
                if not row["forfeits"] and not row["strikes"] and row["changes"] and (
                    elite is None or scores[elite]["strikes"]
                    or old.rank(row, forced) > old.rank(scores[elite], forced)
                ):
                    archive[slot] = g
                log.write(json.dumps({
                    "batch": r + 1, "forced": forced, "genome": g, "result": row,
                    "controls": {key: scores[key] for key in CONTROLS},
                }) + "\n")

            if not forced and (not hall or r % 8 == 0):
                contenders = [g for g in candidates if g not in hall and scores[g]["changes"]
                              and not (scores[g]["strikes"] or scores[g]["forfeits"])]
                if contenders:
                    hall.append(max(contenders, key=lambda g: old.rank(scores[g])))
                    if len(hall) > 20:
                        hall = hall[:4] + hall[-16:]

            seen.update(candidates)
            state["round"] += 1
            state["searched"] += len(candidates)
            state["elapsed"] += time.monotonic() - started
            state.update(archive=list(archive.items()), seen=sorted(seen), hall=hall)
            checkpoint(out, state)

            if smoke or r == 0 or time.monotonic() - last_status >= 90:
                last_status = time.monotonic()
                print(f"{state['searched']} proposals | {len(archive)} styles | {len(hall)} league snapshots")


def test_press_components(pool, out, state):
    press_genome = tuple(state["press_genome"])
    opponents = ("box", "fine", "wide_two_field", "teamplay")
    count = 1 if state["smoke"] else 8
    reports = []
    for block in range(2):
        cases = old.deck(opponents, state["seed"] + 4_000_000 + 10_000 * block, count)
        rows = evaluate(pool, ["press", *ABLATIONS], cases, press_genome)
        reports.append({
            name: {
                "overall": old.summarize(games),
                "opponents": {op: old.summarize([r for r in games if r["case"][0] == op])
                              for op in opponents},
            }
            for name, games in rows.items()
        })

    old.save(out / "press_components.json", reports)
    print("Recovered winner component checks complete.")


def pick_finalists(pool, out, state):
    press_genome = tuple(state["press_genome"])
    archive = list(dict.fromkeys([
        *(tuple(g) for _, g in state["archive"]), *(tuple(g) for g in state["hall"]),
    ]))
    if not archive:
        return []

    rows = evaluate(pool, [*CONTROLS, *archive], old.deck(
        TRAIN, state["seed"] + 7_000_000, 1 if state["smoke"] else 8
    ), press_genome)
    scores = {k: old.summarize(rr) for k, rr in rows.items()}
    clean = [g for g in archive if not (scores[g]["strikes"] or scores[g]["forfeits"])]
    order = sorted(clean, key=lambda g: old.rank(scores[g]), reverse=True)
    old.save(out / "validation.json", {
        "controls": {name: scores[name] for name in CONTROLS},
        "candidates": [{"genome": g, **scores[g]} for g in order],
    })
    if state["smoke"]:
        return order[:2]

    parts = []
    if order:
        for name in active(order[0]):
            modified = list(order[0])
            modified[bot.FAMILIES[name]] = "base"
            genome = bot.normalize(modified)
            if active(genome):
                parts.append(genome)
    by_base = [next((g for g in order if g[0] == base), None) for base in bot.OPTIONS[0]]
    by_tactic = [next((g for g in order if active(g) == (name,)), None) for name in bot.FAMILIES]
    return list(dict.fromkeys(g for g in [*order[:2], *parts, *by_base, *by_tactic] if g))[:9]


def finals(pool, out, state):
    finalists = [tuple(g) for g in state["finalists"]]
    press_genome = tuple(state["press_genome"])
    for i in range(len(state["blocks"]), 2):
        if not finalists:
            break
        count = 1 if state["smoke"] else 32
        cases = old.deck(FINAL, state["seed"] + 12_000_000 + i * 10_000, count)
        print(f"Final block {i + 1}/2: {len(cases) * (len(CONTROLS) + len(finalists))} games.")
        rows = evaluate(pool, [*CONTROLS, *finalists], cases, press_genome)

        report = []
        for key, games in rows.items():
            report.append(dict(
                policy=key,
                overall=old.summarize(games),
                opponents={name: old.summarize([r for r in games if r["case"][0] == name])
                           for name in FINAL},
            ))
            label = key if isinstance(key, str) else f"{key[0]} / {', '.join(active(key))}"
            print(f"{label}: {report[-1]['overall']['points']} points")

        old.save(out / f"final_matches_{i + 1}.json",
                 [{"policy": key, "matches": games} for key, games in rows.items()])
        state["blocks"].append(report)
        checkpoint(out, state)


def finish(out, state):
    blocks = state["blocks"]
    controls = [dict((r["policy"], r["overall"]) for r in b[:len(CONTROLS)]) for b in blocks]
    comparisons = []

    for i, g in enumerate(state["finalists"], len(CONTROLS)):
        results = [b[i]["overall"] for b in blocks]
        gain = {
            name: [row["points"] - control[name]["points"]
                   for row, control in zip(results, controls)] for name in CONTROLS
        }
        clean = all(not (entry["strikes"] or entry["forfeits"])
                    for ctrl, row in zip(controls, results) for entry in [row, *ctrl.values()])
        comparisons.append(dict(
            genome=g, clean=clean, gain_by_block=gain,
            total={name: sum(values) for name, values in gain.items()},
        ))

    eligible = [row for row in comparisons if row["clean"] and len(blocks) == 2]
    best = max(eligible, key=lambda r: min(r["total"].values()), default=None)
    export = best is not None and min(best["total"].values()) > 0 and not state["smoke"]
    if export:
        old.save(out / "recommended_policy.json", {
            "genome": best["genome"], "press_genome": state["press_genome"],
            "tactics": dict(zip(bot.NAMES, best["genome"])), "point_gain": best["total"],
        })

    old.save(out / "summary.json", {
        "source": state["source"], "searched": state["searched"], "styles": len(state["archive"]),
        "snapshots": len(state["hall"]), "search_minutes": state["elapsed"] / 60,
        "blocks": blocks, "comparisons": comparisons, "best_candidate": best,
        "recommendation": best if export else None,
    })

    if state["smoke"]:
        if len(blocks) != 2 or any(
            r["overall"]["strikes"] or r["overall"]["forfeits"]
            for bloc in blocks for r in bloc
        ):
            raise RuntimeError("Smoke test incomplete or had strikes. See summary.json.")
        print("Smoke test passed.")
    print(f"Summary: {out / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=970113)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--final-only", action="store_true")
    args = parser.parse_args()

    if not 0 < args.hours < float("inf") or not 1 <= args.workers <= 61 or args.batch <= 0 or args.seed < 0:
        parser.error("Use positive hours and batch size, 1–61 workers, and a nonnegative seed.")

    if args.resume:
        out = args.resume.resolve()
        state = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
    else:
        if not args.recovery or args.final_only:
            parser.error("Use --recovery SUMMARY for a new run, or --resume FOLDER to finalize a saved run.")
        source = args.recovery.resolve()
        if not source.is_file():
            parser.error(f"Recovery summary missing: {source}")

        reference = json.loads(source.read_text(encoding="utf-8"))
        best = reference.get("best_hybrid")
        if len(reference.get("blocks", [])) != 2 or not best or not best["clean"]:
            parser.error("Give the completed press recovery summary with a clean best hybrid.")

        totals = Counter()
        for block in reference["blocks"]:
            for row in block:
                key = row["policy"]
                if isinstance(key, str):
                    totals[key] += row["overall"]["points"]

        labels = {"two_field_box": "box", "two_field_fine": "fine", "teamplay": "teamplay"}
        preferred = labels[max(labels, key=lambda name: totals[name])]
        if min(best["total"].values()) > 0:
            preferred = "press"

        data = old.ROOT / "data"
        data.mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="league_overnight_", dir=data))
        state = dict(
            source=str(source), press_genome=best["genome"], preferred=preferred,
            seed=args.seed, smoke=args.smoke_test, phase="search", round=0,
            search_budget=720 if args.smoke_test else args.hours * 3600,
            elapsed=0, searched=0, archive=[], seen=[], hall=[], finalists=[], blocks=[],
        )

    print(f"Results folder: {out}")
    print(f"Preferred foundation: {state['preferred']} | workers: {args.workers}")
    checkpoint(out, state)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if state["phase"] == "search":
            if not args.final_only:
                if not (out / "press_components.json").exists():
                    test_press_components(pool, out, state)
                search(pool, out, state, min(args.batch, 4) if state["smoke"] else args.batch)

            print("Validating league candidates on fresh seeds.")
            state["finalists"] = pick_finalists(pool, out, state)
            state["phase"] = "finals"
            checkpoint(out, state)

        finals(pool, out, state)

    finish(out, state)


if __name__ == "__main__":
    main()