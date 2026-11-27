"""Eight-hour search for adaptive two-field play."""

import argparse
import json
import random
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path
from types import ModuleType

import overnight_teamplay as old

from grid2v2.agents import ScriptedAgent, manhattan
from grid2v2.match import play
from teams import team2_press as bot

TRAIN = ("baselines.scripted", "baselines.scripted_box", "baselines.learned",
         "p05414", "teamplay", "two_field_box")
FINAL = (*TRAIN, "teams.team1", "two_field_fine", "wide_two_field")
CONTROLS = ("teamplay", "two_field_box", "two_field_fine")
STARTS = {
    "wide_attack": (((9, 2), (10, 4), (10, 2), (15, 5)), 0),
    "two_on_one": (((11, 7), (13, 5), (10, 7), (17, 6)), 0),
    "high_trap": (((12, 1), (9, 3), (11, 0), (14, 2)), 2),
    "outlet_trap": (((10, 3), (14, 5), (12, 3), (8, 5)), 2),
    "press_broken": (((10, 3), (5, 6), (6, 4), (4, 7)), 2),
    "late_cover": (((5, 6), (4, 7), (4, 4), (6, 2)), 2),
    "wing_shot": (((7, 2), (4, 5), (4, 3), (3, 6)), 2),
}
# Only the training harness receives these starts. The team receives observations.
old.POSITIONS.update(STARTS)
old.POSITIONS.update({
    name + "_mirror": (tuple((x, 10 - y) for x, y in cells), holder)
    for name, (cells, holder) in STARTS.items()
})
CASES = (*STARTS, *(name + "_mirror" for name in STARTS))


class WideField(ScriptedAgent):
    def _support(self, me, mate, opponents, occupied):
        x = min(mate[0] + 2, self.rink.width - self.crease - 1)
        cells = [
            (x, min(self.rink.height - 1, max(0, mate[1] + dy)))
            for dy in (-3, 3, -2, 2)
        ]
        valid = [c for c in cells if c not in occupied and not self.forbidden(c)]
        if not valid or not opponents:
            return super()._support(me, mate, opponents, occupied)
        target = max(valid, key=lambda c: (
            min(manhattan(c, p) for p, _ in opponents), -manhattan(me, c),
        ))
        return self.adapter.encode_move(self._toward(me, target, occupied))


def team(genome, trace=None):
    module = ModuleType("adaptive_press")
    module.TIER = bot.TIER

    def make_team(side, ids, env):
        return bot.build_team(side, ids, env, genome, trace)

    module.make_team = make_team
    return module


@cache
def opponent(name):
    if name == "teamplay":
        return team(bot.DEFAULT)
    if name != "wide_two_field":
        return old.opponent(name)

    module = ModuleType(name)
    module.TIER = "box"

    def make_team(side, ids, env):
        return {a: (WideField if i else ScriptedAgent)(
            env.adapters[a], env.rink, index=0
        ) for i, a in enumerate(ids)}

    module.make_team = make_team
    return module


def run_job(job):
    key, case, deadline = job
    if time.monotonic() >= deadline:
        return None
    rival, seed, swap, scene = case
    trace = [Counter(), Counter()]
    candidate = opponent(key) if isinstance(key, str) else team(key, trace)
    adversary = opponent(rival)
    if scene is None:
        result = play(candidate, adversary, seed=seed, swap=swap)
        side, other = ("blue", "red") if swap else ("red", "blue")
        gf, ga = result["score"][side], result["score"][other]
        strikes, forfeit = result["strikes"][side], result["forfeit"]
        slowest, reward = result["slowest_act_ms"][side], 0.0
    else:
        gf, ga, reward = old.short_game(candidate, adversary, seed, swap, scene)
        strikes, forfeit, slowest = 0, None, 0.0
    return key, {
        "case": case, "gf": gf, "ga": ga,
        "points": 3 if gf > ga else int(gf == ga), "scenario_score": reward,
        "strikes": strikes, "forfeit": forfeit, "slowest_ms": slowest,
        "changes": dict(trace[0] + trace[1]),
    }


def evaluate(pool, policies, cases, deadline):
    results = {key: [] for key in dict.fromkeys(policies)}
    jobs = ((key, case, deadline) for case in cases for key in results)
    for item in pool.map(run_job, jobs, chunksize=2):
        if item is None:
            return None
        key, row = item
        results[key].append(row)
    return results


def styles(genome):
    return tuple(genome[i] for i in (0, 4, 8))


def groups(genome):
    return [name for name, slots in bot.FAMILIES.items() if genome[slots.start] != "current"]


def proposals_for(rng, archive, seen, count, progress, force):
    proposals = {}
    scopes = list(bot.FAMILIES)
    for attempt in range(count * 55):
        focus = scopes[attempt % len(scopes)] if force or rng.random() < 0.30 else None
        slots = bot.FAMILIES[focus] if focus else slice(0, len(bot.OPTIONS))
        parents = [g for g in archive.values() if not focus or groups(g) == [focus]]
        if not parents or rng.random() < 0.68 - 0.43 * progress:
            genes = list(bot.DEFAULT)
            genes[slots] = [rng.choice(bot.OPTIONS[i]) for i in range(slots.start, slots.stop)]
        else:
            genes = list(rng.choice(parents))
            if rng.random() < 0.40:
                other = rng.choice(parents)
                for field, part in bot.FAMILIES.items():
                    if (focus is None or field == focus) and rng.random() < 0.5:
                        genes[part] = other[part]
            indices = range(slots.start, slots.stop)
            for i in rng.sample(indices, rng.randint(1, min(4, len(indices)))):
                genes[i] = rng.choice([v for v in bot.OPTIONS[i] if v != genes[i]])
        genome = bot.normalize(genes)
        if genome != bot.DEFAULT and genome not in seen:
            proposals[genome] = styles(genome)
        if len(proposals) >= count:
            break
    return proposals


def scenario_deck(rng, seed, small):
    names = rng.sample(CASES, 6 if small else 10)
    rivals = ("two_field_box", "teamplay")
    return [(rival, seed, swap, scene) for scene in names
            for rival in rivals for swap in (False, True)]


def checkpoint(out, state, start):
    state["elapsed"] = time.monotonic() - start
    old.save(out / "checkpoint.json", state)


def search(pool, out, state, start, end, batch, force):
    archive = {tuple(key): tuple(g) for key, g in state["archive"]}
    seen = {tuple(g) for g in state["seen"]}
    stage = "situations" if force else "full_matches"
    last = time.monotonic()
    with (out / "search.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        while time.monotonic() < end and (not state["smoke"] or state["round"] < 2):
            index = state["round"]
            seed = state["seed"] + 3_000_000 * state["phase"] + 5 * index
            rng = random.Random(seed)
            progress = min(1.0, (time.monotonic() - start) / (0.92 * state["budget"]))
            proposals = proposals_for(rng, archive, seen, batch, progress, force)
            if not proposals:
                break
            cases = (
                scenario_deck(rng, seed, state["smoke"]) if force
                else old.deck(TRAIN, seed, 1 if state["smoke"] else 2)
            )
            incumbents = [archive[key] for key in set(proposals.values()) if key in archive]
            rows = evaluate(pool, [*CONTROLS, *proposals, *incumbents], cases, end)
            if rows is None:
                break
            scores = {key: old.summarize(games) for key, games in rows.items()}
            for genome, key in proposals.items():
                row = scores[genome]
                elite = archive.get(key)
                if not row["forfeits"] and row["changes"] and (
                    elite is None or old.rank(row, force) > old.rank(scores[elite], force)
                ):
                    archive[key] = genome
                log.write(json.dumps({
                    "stage": stage, "batch": index + 1, "genome": genome, "score": row,
                    "controls": {name: scores[name] for name in CONTROLS},
                }) + "\n")
            seen.update(proposals)
            state["round"] += 1
            state["searched"] += len(proposals)
            if index == 0 or state["smoke"] or time.monotonic() - last >= 90:
                last = time.monotonic()
                print(f"{stage}: {state['searched']} proposals total, {len(archive)} styles.")
                state["archive"], state["seen"] = list(archive.items()), sorted(seen)
                checkpoint(out, state, start)
    state["archive"], state["seen"] = list(archive.items()), sorted(seen)
    checkpoint(out, state, start)


def remove_part(genome, family):
    slots = bot.FAMILIES[family]
    genes = list(genome)
    genes[slots] = bot.DEFAULT[slots]
    return tuple(genes)


def get_finalists(pool, out, state, end):
    archive = list(dict.fromkeys(tuple(g) for _, g in state["archive"]))
    if not archive:
        return []
    count = 1 if state["smoke"] else 8
    cases = old.deck(TRAIN, state["seed"] + 12_000_000, count)
    rows = evaluate(pool, [*CONTROLS, *archive], cases, end)
    if rows is None:
        return []
    score = {key: old.summarize(games) for key, games in rows.items()}
    ordered = sorted((g for g in archive if not score[g]["forfeits"]),
                     key=lambda g: old.rank(score[g]), reverse=True)
    old.save(out / "validation.json", {
        "controls": {key: score[key] for key in CONTROLS},
        "policies": [{"genome": g, **score[g]} for g in ordered],
    })
    if state["smoke"]:
        return ordered[:2]
    specialists = [next((g for g in ordered if groups(g) == [name]), None)
                   for name in bot.FAMILIES]
    parts = [remove_part(ordered[0], name) for name in groups(ordered[0])] if ordered else []
    return list(dict.fromkeys(
        g for g in [*ordered[:3], *specialists, *parts] if g and g != bot.DEFAULT
    ))


def final_block(pool, finalists, seed, count, deadline):
    rows = evaluate(pool, [*CONTROLS, *finalists], old.deck(FINAL, seed, count), deadline)
    if rows is None:
        return None, None
    report = [
        {"policy": key, "overall": old.summarize(games),
         "opponents": {name: old.summarize([m for m in games if m["case"][0] == name])
                       for name in FINAL}}
        for key, games in rows.items()
    ]
    raw = [{"policy": key, "matches": games} for key, games in rows.items()]
    return report, raw


def finish(out, state, start):
    blocks = [{json.dumps(r["policy"]): r for r in block} for block in state["blocks"]]
    comparisons = []
    for genome in (tuple(g) for g in state["finalists"]):
        key = json.dumps(genome)
        gain = [{name: block[key]["overall"]["points"] - block[json.dumps(name)]["overall"]["points"]
                 for name in CONTROLS} for block in blocks]
        total = {name: sum(row[name] for row in gain) for name in CONTROLS}
        clean = all(
            not (block[k]["overall"]["strikes"] or block[k]["overall"]["forfeits"])
            for block in blocks for k in [key, *(json.dumps(name) for name in CONTROLS)]
        )
        repeated = len(gain) == 2 and all(min(row.values()) > 0 for row in gain)
        comparisons.append(dict(genome=genome, gains=gain, total=total,
                                clean=clean, repeated_gain=repeated))
    choices = [row for row in comparisons if row["clean"] and len(blocks) == 2]
    best = max(choices, key=lambda row: min(row["total"].values()), default=None)
    recommended = best if best and min(best["total"].values()) > 0 and not state["smoke"] else None
    if recommended:
        genome = recommended["genome"]
        old.save(out / "recommended_policy.json", {
            "genome": genome, "tactics": dict(zip(bot.NAMES, genome)),
            "point_gain": recommended["total"],
        })
    old.save(out / "summary.json", {
        "searched": state["searched"], "minutes": (time.monotonic() - start) / 60,
        "controls": CONTROLS, "archive_size": len(state["archive"]),
        "blocks": state["blocks"], "comparisons": comparisons,
        "best_hybrid": best, "recommendation": recommended,
    })
    if state["smoke"]:
        if len(blocks) != 2 or any(
            row["overall"]["strikes"] or row["overall"]["forfeits"]
            for block in state["blocks"] for row in block
        ):
            raise RuntimeError("Smoke test incomplete or had strikes. See summary.json.")
        print("Smoke test passed.")
    elif recommended:
        print(f"Exported: {out / 'recommended_policy.json'}")
    elif len(blocks) == 2:
        print("Finals complete. See the best hybrid and matchup gains in summary.json.")
    else:
        print("Final evaluation incomplete. See summary.json and checkpoint.json.")
    print(f"Summary: {out / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=752910)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not 0 < args.hours < float("inf") or not 1 <= args.workers <= 61 or args.batch < 1 or args.seed < 0:
        parser.error("Use positive hours and batch size, 1-61 workers, and a nonnegative seed.")

    if args.resume:
        out = args.resume.resolve()
        state = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
    else:
        data = old.ROOT / "data"
        data.mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="press_overnight_", dir=data))
        state = dict(seed=args.seed, smoke=args.smoke_test, budget=720 if args.smoke_test else args.hours * 3600,
                     elapsed=0, phase=0, round=0, searched=0, archive=[], seen=[], finalists=[], blocks=[])
    start = time.monotonic() - state["elapsed"]
    budget = state["budget"]
    batch = min(args.batch, 5) if state["smoke"] else args.batch
    print(f"Results folder: {out}")
    print(f"Workers: {args.workers} | proposals per batch: {batch}")
    checkpoint(out, state, start)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if state["phase"] == 0:
            search(pool, out, state, start, start + 0.18 * budget, batch, True)
            state.update(phase=1, round=0, seen=[])
            checkpoint(out, state, start)
        if state["phase"] == 1:
            search(pool, out, state, start, start + 0.92 * budget, batch, False)
            print("Validating archived styles before the final matches.")
            state["finalists"] = get_finalists(pool, out, state, start + 0.95 * budget)
            state["phase"] = 2
            checkpoint(out, state, start)
        finalists = [tuple(g) for g in state["finalists"]]
        for index in range(len(state["blocks"]), 2):
            if not finalists:
                break
            print(f"Final block {index + 1}/2, {len(finalists)} candidates.")
            report, raw = final_block(
                pool, finalists, state["seed"] + 20_000_000 + index * 10_000,
                1 if state["smoke"] else 32, start + budget,
            )
            if report is None:
                break
            for number, row in enumerate(report, 1 - len(CONTROLS)):
                key = row["policy"]
                label = key if isinstance(key, str) else f"Finalist {number} " + "/".join(styles(key))
                result = row["overall"]
                print(f"{label}: {result['wdl']} W/D/L, {result['points']} points")
            old.save(out / f"final_matches_{index + 1}.json", raw)
            state["blocks"].append(report)
            checkpoint(out, state, start)
    finish(out, state, start)


if __name__ == "__main__":
    main()