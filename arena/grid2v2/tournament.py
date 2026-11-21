"""grid2v2/tournament.py — round robin between team modules, both sides, Elo ladder, one table.

    python -m grid2v2.tournament baselines.scripted baselines.scripted_box teams.template --games 4 --seed 0
    python -m grid2v2.tournament --dir teams --games 10 --logs logs/      (every team package in a folder, plus the baselines)

Each pair plays `games` matches with sides alternating.  Standings: points (3/1/0), goal difference, then Elo
(K = 32, updated match by match in a fixed order, so the ladder is reproducible).  The tier each team uses is reported
next to its name; teams with different action spaces play on the same rink.
"""
from __future__ import annotations
import argparse, importlib, itertools, json, os, sys
from .match import play, load_team, tier_of

K_ELO = 32


def expected(ra, rb): return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def run(modules, games=4, seed=0, players_per_team=2, logs=None, env_kwargs=None, verbose=True):
    mods = [load_team(m) for m in modules]
    names = [m.__name__ for m in mods]
    table = {n: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "pts": 0, "elo": 1000.0, "strikes": 0, "tier": tier_of(m) if isinstance(tier_of(m), str) else tier_of(m).name}
             for n, m in zip(names, mods)}
    results = []
    for (ia, a), (ib, b) in itertools.combinations(enumerate(mods), 2):
        for g in range(games):
            swap = g % 2 == 1
            log = None if logs is None else os.path.join(logs, f"{names[ia].split('.')[-1]}_vs_{names[ib].split('.')[-1]}_{g}.json")
            r = play(a, b, seed=seed + g, swap=swap, players_per_team=players_per_team, log=log, env_kwargs=env_kwargs)
            sa, sb = (r["score"]["blue"], r["score"]["red"]) if swap else (r["score"]["red"], r["score"]["blue"])
            ka, kb = (r["strikes"]["blue"], r["strikes"]["red"]) if swap else (r["strikes"]["red"], r["strikes"]["blue"])
            for n, gf, ga, st in ((names[ia], sa, sb, ka), (names[ib], sb, sa, kb)):
                t = table[n]; t["P"] += 1; t["GF"] += gf; t["GA"] += ga; t["strikes"] += st
                t["W"] += gf > ga; t["D"] += gf == ga; t["L"] += gf < ga; t["pts"] += 3 if gf > ga else (1 if gf == ga else 0)
            ea = expected(table[names[ia]]["elo"], table[names[ib]]["elo"])
            score_a = 1.0 if sa > sb else (0.5 if sa == sb else 0.0)
            table[names[ia]]["elo"] += K_ELO * (score_a - ea); table[names[ib]]["elo"] -= K_ELO * (score_a - ea)
            results.append({"a": names[ia], "b": names[ib], "game": g, "swap": swap, "score": [sa, sb], "steps": r["steps"]})
            if verbose:
                print(f"  {names[ia]} {sa}-{sb} {names[ib]}" + ("  (sides swapped)" if swap else "")); sys.stdout.flush()
    order = sorted(table, key=lambda n: (-table[n]["pts"], -(table[n]["GF"] - table[n]["GA"]), -table[n]["elo"]))
    return order, table, results


def format_table(order, table) -> str:
    head = f"{'#':>2} {'team':<28}{'tier':<10}{'P':>3}{'W':>3}{'D':>3}{'L':>3}{'GF':>5}{'GA':>5}{'GD':>5}{'pts':>5}{'Elo':>7}{'strikes':>9}"
    rows = [head, "-" * len(head)]
    for i, n in enumerate(order, 1):
        t = table[n]
        rows.append(f"{i:>2} {n:<28}{t['tier']:<10}{t['P']:>3}{t['W']:>3}{t['D']:>3}{t['L']:>3}{t['GF']:>5}{t['GA']:>5}{t['GF'] - t['GA']:>5}{t['pts']:>5}{t['elo']:>7.0f}{t['strikes']:>9}")
    return "\n".join(rows)


def discover(folder):
    """Every package or module directly inside `folder` that defines make_team, imported as `<folder>.<name>`."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(folder)))
    pkg = os.path.basename(os.path.normpath(folder))
    found = []
    for name in sorted(os.listdir(folder)):
        if name.startswith(("_", ".")): continue
        mod = name[:-3] if name.endswith(".py") else (name if os.path.isdir(os.path.join(folder, name)) else None)
        if mod is None: continue
        try:
            m = importlib.import_module(f"{pkg}.{mod}")
            if hasattr(m, "make_team"): found.append(m)
        except Exception as e:                                            # a team that will not import plays no games
            print(f"  skipping {pkg}.{mod}: {type(e).__name__}: {e}")
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("teams", nargs="*", help="team modules, e.g. baselines.scripted teams.template")
    p.add_argument("--dir", default=None, help="folder of team packages to discover instead")
    p.add_argument("--games", type=int, default=4); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--players", type=int, default=2); p.add_argument("--logs", default=None)
    p.add_argument("--out", default=None, help="write standings and results as JSON here")
    p.add_argument("--no-baselines", action="store_true", help="with --dir: do not add the four baselines")
    a = p.parse_args()
    mods = discover(a.dir) if a.dir else list(a.teams)
    if a.dir and not a.no_baselines:
        mods += ["baselines.random", "baselines.scripted", "baselines.scripted_box"]
        if os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baselines", "learned")):
            mods.append("baselines.learned")
    if len(mods) < 2:
        sys.exit("need at least two teams")
    if a.logs: os.makedirs(a.logs, exist_ok=True)
    order, table, results = run(mods, games=a.games, seed=a.seed, players_per_team=a.players, logs=a.logs)
    print(); print(format_table(order, table))
    if a.out:
        with open(a.out, "w") as f: json.dump({"order": order, "table": table, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
