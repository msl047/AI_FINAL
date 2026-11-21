"""tests/test_budget.py — the time budget, strikes and forfeits in grid2v2.match.  Run: python -m pytest tests -q"""
import os, sys, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid2v2.match import play, series, Budget, FORFEIT_SCORE
from grid2v2.agents import Agent, ScriptedAgent
import baselines.scripted as scripted


def team_module(name, agent_cls, tier="discrete", setup_sleep=0.0):
    m = types.ModuleType(name); m.TIER = tier
    def make_team(team, ids, env):
        time.sleep(setup_sleep)
        return {a: agent_cls(env.adapters[a], env.rink, index=i) for i, a in enumerate(ids)}
    m.make_team = make_team
    return m


class Slow(Agent):
    def act(self, obs): time.sleep(0.02); return 0


class Crash(Agent):
    def act(self, obs): raise ValueError("oops")


class Hang(Agent):
    def act(self, obs):
        while True: pass


class SometimesSlow(Agent):
    n = 0
    def act(self, obs):
        SometimesSlow.n += 1
        if SometimesSlow.n % 200 == 0: time.sleep(0.02)
        return 0


def test_fast_team_has_no_strikes():
    r = play(scripted, scripted, seed=0)
    assert r["strikes"] == {"red": 0, "blue": 0} and r["forfeit"] is None and r["steps"] == 600


def test_slow_act_strikes_then_forfeits():
    r = play(team_module("slow", Slow), scripted, seed=0, budget=Budget(act=0.01, max_strikes=5))
    assert r["forfeit"][0] == "red" and r["score"] == {"red": 0, "blue": FORFEIT_SCORE} and r["strikes"]["red"] == 6


def test_exception_is_a_strike():
    r = play(scripted, team_module("crash", Crash), seed=0, budget=Budget(max_strikes=3))
    assert r["forfeit"][0] == "blue" and "ValueError" in r["forfeit"][1] and r["score"]["red"] == FORFEIT_SCORE


def test_hard_timeout_interrupts_a_hang():
    t0 = time.time()
    r = play(team_module("hang", Hang), scripted, seed=0, budget=Budget(act=0.01, hard=0.05, max_strikes=2))
    assert r["forfeit"][0] == "red" and time.time() - t0 < 5


def test_occasional_slowness_is_tolerated():
    SometimesSlow.n = 0
    r = play(team_module("sometimes", SometimesSlow), scripted, seed=0, budget=Budget(act=0.01))
    assert r["forfeit"] is None and 0 < r["strikes"]["red"] <= 10


def test_setup_timeout_forfeits_before_kickoff():
    r = play(scripted, team_module("lazy", Slow, setup_sleep=0.3), seed=0, budget=Budget(setup=0.1))
    assert r["forfeit"][0] == "blue" and r["steps"] == 0


def test_series_swaps_sides_and_reports_forfeits():
    r = series(scripted, team_module("crash", Crash), games=2, seed=0, budget=Budget(max_strikes=1))
    assert r["W"] == 2 and r["GF"] == 2 * FORFEIT_SCORE and len(r["forfeits"]) == 2
