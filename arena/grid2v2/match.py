"""grid2v2/match.py — play matches between two team modules, under the time budget, with optional JSON logs.

A team is a module (or package) exposing  make_team(team: str, agent_ids: list[str], env) -> {agent_id: Agent}
and, optionally, TIER = "discrete" | "box" | "fine" | an Adapter (default "discrete").  Agents may use `env.rink` and
their adapter and nothing else from `env`: every decision comes from the observation.

The budget (Berkeley Pacman rules, scaled to a 600-step match):
    setup   15 s   for make_team() plus reset(), hard: over it and the team forfeits before kick-off
    act     50 ms  soft: a slower act() still counts, the action is replaced by STAY and the team takes a strike
             1 s   hard: the call is interrupted (SIGALRM), STAY, strike
    strikes 10     per team per match; the eleventh forfeits the match (recorded as a 0-3 loss)
An exception inside act() is a strike too.  All timings are wall clock on the machine running the match; the
tournament runs on the reference machine, so test under `--budget 0.025` locally if your laptop is faster.

    python -m grid2v2.match --red teams.template --blue baselines.scripted --games 5
    python -m grid2v2.match --red baselines.scripted_box --blue baselines.scripted --games 20 --seed 1
    python -m grid2v2.match --red baselines.scripted --blue baselines.random --games 1 --log match.json   (for the renderer)
"""
from __future__ import annotations
import argparse, importlib, json, signal, time
from contextlib import contextmanager
from dataclasses import dataclass
from .env import Grid2v2, TEAMS

FORFEIT_SCORE = 3


@dataclass
class Budget:
    act: float = 0.05          # soft, seconds per act()
    hard: float = 1.0          # hard, seconds per act(): interrupted
    setup: float = 15.0        # hard, seconds for make_team() + reset()
    max_strikes: int = 10      # the (max_strikes + 1)th strike forfeits


class HardTimeout(Exception):
    pass


_HAS_ALARM = hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")


@contextmanager
def deadline(seconds: float):
    """Interrupt the block after `seconds` (main thread, POSIX).  Elsewhere it degrades to measuring only."""
    if not _HAS_ALARM or seconds is None:
        yield; return
    def handler(signum, frame): raise HardTimeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def load_team(spec):
    return importlib.import_module(spec) if isinstance(spec, str) else spec


def tier_of(module):
    return getattr(module, "TIER", "discrete")


def make_env(red, blue, players_per_team=2, **kw):
    return Grid2v2(players_per_team=players_per_team, tiers={"red": tier_of(red), "blue": tier_of(blue)}, **kw)


def play(red, blue, seed=0, players_per_team=2, budget: Budget | float | None = None, log=None, env_kwargs=None, swap=False):
    """One match.  Returns a dict: score, strikes, forfeit, steps, events, tiers, names.
    `swap=True` plays `red` as blue and `blue` as red.  `budget` may be a Budget or just the soft act budget in seconds."""
    if budget is None: budget = Budget()
    elif not isinstance(budget, Budget): budget = Budget(act=float(budget))
    red, blue = load_team(red), load_team(blue)
    mods = {"red": blue, "blue": red} if swap else {"red": red, "blue": blue}
    env = make_env(mods["red"], mods["blue"], players_per_team, **(env_kwargs or {}))
    obs, infos = env.reset(seed=seed)
    strikes = {t: 0 for t in TEAMS}
    slowest = {t: 0.0 for t in TEAMS}
    forfeit = None
    agents = {}
    for t in TEAMS:                                                  # setup, under the hard setup budget
        try:
            with deadline(budget.setup):
                t0 = time.perf_counter()
                agents.update(mods[t].make_team(t, env.members[t], env))
                for a in env.members[t]: agents[a].reset()
                if time.perf_counter() - t0 > budget.setup: raise HardTimeout()
        except Exception as e:
            forfeit = (t, f"setup: {type(e).__name__}: {e}"[:200])
    frames = [env.state()] if log else None
    counts = {}
    while env.agents and forfeit is None:
        actions = {}
        for a in env.agents:
            t = env.team[a]
            t0 = time.perf_counter()
            try:
                with deadline(budget.hard):
                    act = agents[a].act(obs[a])
                dt = time.perf_counter() - t0
                slowest[t] = max(slowest[t], dt)
                if dt > budget.act: raise TimeoutError(f"{dt * 1000:.0f} ms")
            except Exception as e:
                act = env.adapters[a].noop(); strikes[t] += 1
                env_note = ("strike", a, f"{type(e).__name__}: {e}"[:80])
                if strikes[t] > budget.max_strikes:
                    forfeit = (t, f"{strikes[t]} strikes; last: {env_note[2]}"); break
            actions[a] = act
        if forfeit is not None: break
        obs, rewards, terms, truncs, infos = env.step(actions)
        for e in env.events: counts[e[0]] = counts.get(e[0], 0) + 1
        if frames is not None: frames.append(env.state())
    score = dict(env.score)
    if forfeit is not None:
        loser = forfeit[0]; winner = "blue" if loser == "red" else "red"
        score = {loser: 0, winner: FORFEIT_SCORE}
    result = {"score": score, "strikes": strikes, "forfeit": forfeit, "slowest_act_ms": {t: round(v * 1000, 1) for t, v in slowest.items()},
              "steps": env.t, "events": counts, "tiers": {t: env.tier(f"{t}_0") for t in TEAMS}, "seed": seed, "swap": swap,
              "names": {t: mods[t].__name__ for t in TEAMS}}
    if log:
        with open(log, "w") as f:
            json.dump({"result": result, "rink": [env.rink.width, env.rink.height, env.rink.goal_cells], "crease": env.crease,
                       "max_steps": env.max_steps, "frames": frames}, f)
    return result


def series(red, blue, games=10, seed=0, **kw):
    """Alternate sides each game; report from `red`'s point of view."""
    red, blue = load_team(red), load_team(blue)
    w = d = l = 0; gf = ga = 0; strikes = {"red": 0, "blue": 0}; events = {}; forfeits = []
    for g in range(games):
        swap = g % 2 == 1
        r = play(red, blue, seed=seed + g, swap=swap, **kw)
        mine, theirs = (r["score"]["blue"], r["score"]["red"]) if swap else (r["score"]["red"], r["score"]["blue"])
        gf += mine; ga += theirs
        w += mine > theirs; d += mine == theirs; l += mine < theirs
        for k, v in r["events"].items(): events[k] = events.get(k, 0) + v
        strikes["red"] += r["strikes"]["blue" if swap else "red"]; strikes["blue"] += r["strikes"]["red" if swap else "blue"]
        if r["forfeit"]: forfeits.append((g, r["forfeit"]))
    return {"W": w, "D": d, "L": l, "GF": gf, "GA": ga, "games": games, "events": events, "strikes": strikes,
            "forfeits": forfeits, "red": red.__name__, "blue": blue.__name__}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--red", default="baselines.scripted"); p.add_argument("--blue", default="baselines.random")
    p.add_argument("--games", type=int, default=1); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--players", type=int, default=2); p.add_argument("--log", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--budget", type=float, default=Budget.act, help="soft seconds per act() (default 0.05)")
    a = p.parse_args()
    kw = {} if a.max_steps is None else {"max_steps": a.max_steps}
    b = Budget(act=a.budget)
    if a.games == 1:
        r = play(a.red, a.blue, seed=a.seed, players_per_team=a.players, log=a.log, env_kwargs=kw, budget=b)
        print(json.dumps(r, indent=1))
    else:
        r = series(a.red, a.blue, games=a.games, seed=a.seed, players_per_team=a.players, env_kwargs=kw, budget=b)
        print(f"{r['red']} vs {r['blue']} over {r['games']} games (sides alternate): "
              f"W{r['W']} D{r['D']} L{r['L']}, goals {r['GF']}-{r['GA']}, strikes {r['strikes']}"
              + (f", forfeits {r['forfeits']}" if r["forfeits"] else ""))
        print("events:", r["events"])


if __name__ == "__main__":
    main()
