"""tests/test_rules.py — the rules, one at a time, with deterministic moves (p_move=1).  Run: python -m pytest tests -q"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from grid2v2 import Grid2v2, DiscreteKick, BoxKick, FineDiscrete, Intent, STAY, N, E, S, W, simulate, kick_range, flatten, N_FEATURES
from grid2v2.physics import Rink

KN, KE, KS, KW = 5, 6, 7, 8


def make(**kw):
    kw.setdefault("p_move", 1.0); kw.setdefault("p_dribble", 1.0)
    env = Grid2v2(**kw); env.reset(seed=0); return env


def put(env, agent, cell): env.pos[agent] = tuple(cell)


def give(env, agent):
    from grid2v2.physics import Ball
    env.holder = agent; env.ball = Ball.at_cell(*env.pos[agent])


def settle(env, first_actions):
    """Step with `first_actions`, then with no actions until the ball stops or is held; returns (rewards_sum, events)."""
    total, events = {a: 0.0 for a in env.possible_agents}, []
    acts = first_actions
    for _ in range(12):
        obs, r, term, trunc, info = env.step(acts); acts = {}
        for a in total: total[a] += r.get(a, 0.0)
        events += info[next(iter(info))]["events"] if info else []
        if env.holder is not None or not env.ball.moving or not env.agents: break
    return obs, total, term, trunc, events


def park(env):
    """Move everyone out of the way (corners)."""
    for a, c in zip(env.possible_agents, [(0, 0), (0, 10), (20, 0), (20, 10)]): put(env, a, c)


# ---------------------------------------------------------------------------------------------------------- physics
def test_rink_geometry():
    r = Rink(21, 11)
    assert (r.goal_lo, r.goal_hi) == (3.5, 6.5) and list(r.goal_rows) == [4, 5, 6]


def test_kick_range_is_triangular():
    assert kick_range(3) == 6 and kick_range(4) == 10 and abs(kick_range(2.5) - (2.5 + 1.5 + 0.5)) < 1e-9


def test_straight_shot_scores_and_off_target_bounces():
    r = Rink()
    assert simulate(r, 16, 5, 0.0, 3.0, {})[0] == ("goal", "right")
    ev, b, *_ = simulate(r, 16, 8, 0.0, 3.0, {})
    assert ev is None and abs(b.x - 19.0) < 1e-9 and b.y == 8.0          # 16 -> wall at 20.5 (4.5) -> back 1.5


def test_diagonal_bank_shot():
    r = Rink()
    # from (16, 9) aim down-right at 45 deg: hits the right wall at y = 9 - 4.5 = 4.5, inside the goal mouth
    assert simulate(r, 16, 9, -math.pi / 4, 4.0, {})[0] == ("goal", "right")
    # same kick from one row higher hits the wall at y = 5.5 (goal) too; from (16, 10) at 45 deg down: y = 5.5 -> goal
    ev, b, *_ = simulate(r, 10, 9, -math.pi / 4, 4.0, {})         # hits the floor y = -0.5 first: bounce, no goal
    assert ev is None and b.y > -0.5


def test_interception_takes_the_first_robot_on_the_path():
    r = Rink()
    assert simulate(r, 10, 5, 0.0, 3.0, {(12, 5): "a", (14, 5): "b"})[0] == ("taken", "a")
    assert simulate(r, 10, 5, 0.0, 3.0, {(17, 5): "far"})[0] is None   # range 6 ends at 16


# ---------------------------------------------------------------------------------------------------------- adapters
def test_discrete_adapter():
    ad = DiscreteKick()
    assert ad.decode(E, False) == Intent(E) and ad.decode(KE, False) == Intent(STAY)
    assert ad.decode(KN, True) == Intent(STAY, (math.pi / 2, 3.0))
    assert ad.encode_kick(0.3, 3.0) == KE and ad.encode_kick(math.pi - 0.3, 3.0) == KW


def test_box_adapter():
    ad = BoxKick()
    assert ad.decode([0.9, 0.1, 0, 0], True) == Intent(E) and ad.decode([0.1, -0.9, 0, 0], False) == Intent(S)
    assert ad.decode([0.1, 0.1, 0.1, 0.1], True) == Intent(STAY)
    it = ad.decode([0.9, 0, 1.0, 0], True)
    assert it.kick is not None and abs(it.kick[0]) < 1e-9 and it.kick[1] == 3.0
    it = ad.decode([0.9, 0, 1.0, 0], False)                      # no ball: the kick component is ignored, we move
    assert it == Intent(E)
    it = ad.decode(ad.encode_kick(1.0, 3.0), True)
    assert abs(it.kick[0] - 1.0) < 1e-6 and abs(it.kick[1] - 3.0) < 1e-6


def test_fine_adapter():
    ad = FineDiscrete(16, (2.0, 3.0, 4.0))
    assert ad.space.n == 53
    it = ad.decode(ad.encode_kick(math.pi / 2, 4.0), True)
    assert abs(it.kick[0] - math.pi / 2) < 1e-9 and it.kick[1] == 4.0


def test_mirrored_intent():
    assert Intent(E).mirrored() == Intent(W) and Intent(N).mirrored() == Intent(N)
    k = Intent(STAY, (0.0, 3.0)).mirrored().kick
    assert abs(abs(k[0]) - math.pi) < 1e-9


# --------------------------------------------------------------------------------------------------------------- env
def test_moves_walls_and_stay():
    env = make(); park(env); put(env, "red_0", (0, 5))
    env.step({"red_0": W}); assert env.pos["red_0"] == (0, 5)
    env.step({"red_0": E}); assert env.pos["red_0"] == (1, 5)
    env.step({"red_0": STAY}); assert env.pos["red_0"] == (1, 5)


def test_slip_is_perpendicular():
    env = Grid2v2(p_move=0.0, p_dribble=1.0); env.reset(seed=1); park(env)
    seen = set()
    for _ in range(40):
        put(env, "red_0", (5, 3)); env.step({"red_0": N}); seen.add(env.pos["red_0"])
    assert seen == {(4, 3), (6, 3)}


def test_blue_frame_is_mirrored():
    env = make(); park(env); put(env, "blue_0", (12, 5))
    obs, *_ = env.step({"blue_0": E})                    # "forward" for blue is -x on the rink
    assert env.pos["blue_0"] == (11, 5) and tuple(obs["blue_0"]["self"][:2]) == (9, 5)


def test_no_shared_cells_swap_and_contest():
    env = make(); park(env); put(env, "red_0", (5, 5)); put(env, "blue_0", (6, 5))
    env.step({"red_0": E, "blue_0": E}); assert env.pos["red_0"] == (5, 5) and env.pos["blue_0"] == (6, 5)   # blue's E is -x
    put(env, "red_0", (5, 5)); put(env, "blue_0", (7, 5))
    env.step({"red_0": E, "blue_0": E}); assert {env.pos["red_0"], env.pos["blue_0"]} == {(5, 5), (6, 5)} or {env.pos["red_0"], env.pos["blue_0"]} == {(6, 5), (7, 5)}


def test_pickup_dribble_and_kick():
    env = make(); park(env); put(env, "red_0", (9, 5))
    env.step({"red_0": E}); assert env.holder == "red_0" and (env.ball.cx, env.ball.cy) == (10, 5)
    env.step({"red_0": E}); assert env.pos["red_0"] == (11, 5) and (env.ball.cx, env.ball.cy) == (11, 5)
    obs, *_ = env.step({"red_0": KE})
    assert env.holder is None and env.pos["red_0"] == (11, 5) and (env.ball.cx, env.ball.cy) == (14, 5) and env.ball.speed == 2.0
    env.step({}); env.step({}); assert (env.ball.cx, env.ball.cy) == (17, 5) and not env.ball.moving


def test_kick_without_ball_is_a_noop():
    env = make(); park(env); put(env, "red_0", (3, 3))
    env.step({"red_0": KE}); assert env.pos["red_0"] == (3, 3) and env.holder is None


def test_pass_is_intercepted_by_the_first_robot_on_the_line():
    env = make(); park(env); put(env, "red_0", (8, 2)); give(env, "red_0"); put(env, "blue_0", (10, 2)); put(env, "red_1", (12, 2))
    *_, events = settle(env, {"red_0": KE})
    assert env.holder == "blue_0" and ("intercept", "blue_0") in events
    put(env, "blue_0", (10, 8)); give(env, "red_0")
    settle(env, {"red_0": KE}); assert env.holder == "red_1"


def test_goal_rewards_and_kickoff():
    env = make(); park(env); put(env, "red_0", (16, 5)); give(env, "red_0")
    obs, r, term, trunc, events = settle(env, {"red_0": KE})
    assert env.score == {"red": 1, "blue": 0} and r["red_0"] == 1 and r["red_1"] == 1 and r["blue_0"] == -1 and r["blue_1"] == -1
    assert env.holder == "blue_0" and env.pos == env.kickoff_cells() and ("goal", "red", "red_0") in events
    assert obs["blue_0"]["self"][2] == 1.0 and obs["red_0"]["ball"][0] == 12.0 and obs["blue_0"]["ball"][0] == 8.0


def test_shot_wide_bounces_back_into_play():
    env = make(); park(env); put(env, "red_0", (18, 9)); give(env, "red_0")
    *_, events = settle(env, {"red_0": KE})
    assert env.score["red"] == 0 and ("bounce", 1) in events and env.holder is None     # the rebound passes through the kicker
    assert (env.ball.cx, env.ball.cy) == (17, 9) and not env.ball.moving
    env.step({"red_0": W}); assert env.holder == "red_0"                               # stopped: now it can be picked up


def test_no_self_pass_off_the_wall_but_a_teammate_may_take_it():
    env = make(); park(env); put(env, "red_0", (18, 2)); give(env, "red_0"); put(env, "red_1", (17, 2))
    settle(env, {"red_0": KE})                     # 18 -> wall at 20.5 -> back through 18 (kicker, ignored) to 17 (mate)
    assert env.holder == "red_1"


def test_tackle():
    env = make(p_tackle=1.0); park(env); put(env, "red_0", (5, 5)); give(env, "red_0"); put(env, "blue_0", (6, 5))
    env.step({"blue_0": E}); assert env.holder == "blue_0" and env.pos["blue_0"] == (6, 5)   # blue's E is -x
    env = make(p_tackle=0.0); park(env); put(env, "red_0", (5, 5)); give(env, "red_0"); put(env, "blue_0", (6, 5))
    env.step({"blue_0": E}); assert env.holder == "red_0"
    env = make(p_tackle=1.0); park(env); put(env, "red_0", (5, 5)); give(env, "red_0"); put(env, "red_1", (6, 5))
    env.step({"red_1": W}); assert env.holder == "red_0"                                    # teammates never tackle


def test_mixed_tiers_box_blue_scores():
    env = Grid2v2(tiers={"red": "discrete", "blue": "box"}, p_move=1.0, p_dribble=1.0); env.reset(seed=0); park(env)
    put(env, "blue_0", (4, 5)); give(env, "blue_0")
    settle(env, {"blue_0": np.array([0, 0, 1.0, 0.0], np.float32)})       # +x in blue's frame = towards the left goal
    assert env.score["blue"] == 1


def test_truncation_and_goals_to_win():
    env = make(max_steps=3)
    for _ in range(3): obs, r, term, trunc, _ = env.step({})
    assert all(trunc.values()) and not env.agents
    env = make(goals_to_win=1); park(env); put(env, "red_0", (16, 5)); give(env, "red_0")
    obs, r, term, trunc, _ = settle(env, {"red_0": KE}); assert all(term.values()) and not env.agents


def test_dribbling_is_slow():
    env = Grid2v2(p_move=1.0, p_dribble=0.0); env.reset(seed=0); park(env); put(env, "red_0", (5, 5)); give(env, "red_0")
    env.step({"red_0": E}); assert env.pos["red_0"] == (5, 5)
    env = Grid2v2(p_move=1.0, p_dribble=1.0); env.reset(seed=0); park(env); put(env, "red_0", (5, 5)); give(env, "red_0")
    env.step({"red_0": E}); assert env.pos["red_0"] == (6, 5) and (env.ball.cx, env.ball.cy) == (6, 5)


def test_crease_keeps_attackers_out():
    env = make(); park(env); put(env, "red_0", (18, 5)); put(env, "blue_0", (19, 5))
    env.step({"red_0": E}); assert env.pos["red_0"] == (18, 5)                 # red may not enter its attacking crease
    env.step({"blue_0": W}); assert env.pos["blue_0"] == (20, 5)              # blue (defending there) may
    put(env, "red_0", (18, 3)); env.step({"red_0": E}); assert env.pos["red_0"] == (19, 3)   # outside the goal rows is fine
    env = make(crease=0); park(env); put(env, "red_0", (19, 5)); env.step({"red_0": E}); assert env.pos["red_0"] == (20, 5)


def test_ball_moves_before_robots_react():
    """One-touch pass-and-shoot from the crease edge beats a keeper that only now steps onto the row."""
    env = make(); park(env); put(env, "red_0", (18, 6)); give(env, "red_0"); put(env, "blue_1", (20, 5))
    env.step({"red_0": KE, "blue_1": N})          # blue's N is +y: the keeper steps to row 6 as the ball flies past
    assert env.score["red"] == 1
    env = Grid2v2(p_move=1.0, p_dribble=1.0, ball_first=False); env.reset(seed=0); park(env)
    put(env, "red_0", (18, 6)); give(env, "red_0"); put(env, "blue_1", (20, 5))
    env.step({"red_0": KE, "blue_1": N})          # robots first: the keeper gets there before the ball
    assert env.score["red"] == 0 and env.holder == "blue_1"


def test_a_pass_needs_room():
    env = make(); park(env); put(env, "red_0", (8, 5)); give(env, "red_0"); put(env, "red_1", (8, 6)); put(env, "blue_0", (12, 2))
    settle(env, {"red_0": KN})                     # the adjacent teammate is passed through; the ball rolls on and stops
    assert env.holder is None and (env.ball.cx, env.ball.cy) == (8, 10)
    env = make(); park(env); put(env, "red_0", (8, 5)); give(env, "red_0"); put(env, "red_1", (8, 7)); put(env, "blue_0", (12, 2))
    settle(env, {"red_0": KN}); assert env.holder == "red_1"                          # two cells away is a pass
    env = make(); park(env); put(env, "red_0", (8, 5)); give(env, "red_0"); put(env, "blue_0", (8, 6)); put(env, "red_1", (1, 1))
    settle(env, {"red_0": KN}); assert env.holder == "blue_0"                         # an adjacent opponent still intercepts
    env = make(min_pass=1); park(env); put(env, "red_0", (8, 5)); give(env, "red_0"); put(env, "red_1", (8, 6)); put(env, "blue_0", (12, 2))
    settle(env, {"red_0": KN}); assert env.holder == "red_1"                          # knob off


def test_a_kick_under_pressure_can_be_blocked():
    env = make(block=True, p_block=1.0); park(env); put(env, "red_0", (8, 5)); give(env, "red_0"); put(env, "red_1", (8, 6)); put(env, "blue_0", (9, 5))
    env.step({"red_0": KN, "blue_0": E})          # blue's E is -x: it moves into the kicker as the pass goes off
    assert env.holder == "blue_0" and any(e[0] == "block" for e in env.events)
    env = make(block=True, p_block=0.0); park(env); put(env, "red_0", (8, 5)); give(env, "red_0"); put(env, "red_1", (8, 7)); put(env, "blue_0", (9, 5))
    env.step({"red_0": KN, "blue_0": E}); assert env.holder == "red_1"


def test_failed_tackle_stuns_the_tackler():
    env = make(p_tackle=0.0); park(env); put(env, "red_0", (5, 5)); give(env, "red_0"); put(env, "blue_0", (6, 5))
    obs, *_ = env.step({"blue_0": E}); assert env.holder == "red_0" and obs["blue_0"]["self"][3] == 1 and obs["red_0"]["opponents"][0][3] == 1
    env.step({"blue_0": N}); assert env.pos["blue_0"] == (6, 5)            # stunned: the move is ignored
    obs, *_ = env.step({"blue_0": N}); assert env.pos["blue_0"] == (6, 6) and obs["blue_0"]["self"][3] == 0
    env = make(p_tackle=0.0, stun=0); park(env); put(env, "red_0", (5, 5)); give(env, "red_0"); put(env, "blue_0", (6, 5))
    env.step({"blue_0": E}); env.step({"blue_0": N}); assert env.pos["blue_0"] == (6, 6)   # stun 0: no cost


def test_scripted_never_passes_into_its_own_goal():
    """A pass to an adjacent keeper is not received (min_pass); the scripted player must know that."""
    from grid2v2.agents import ScriptedAgent
    env = Grid2v2(tiers="box", p_move=1.0, p_dribble=1.0); env.reset(seed=0); park(env)
    put(env, "blue_0", (18, 4)); give(env, "blue_0"); put(env, "blue_1", (19, 4)); put(env, "red_0", (17, 4)); put(env, "red_1", (13, 3))
    agent = ScriptedAgent(env.adapters["blue_0"], env.rink, index=0)
    for _ in range(30):
        obs = env.observe("blue_0"); env.step({"blue_0": agent.act(obs)})
        assert env.score["red"] == 0


def test_features():
    env = make(); obs, _ = env.reset(seed=0)
    v = flatten(obs["red_0"]); assert v.shape == (N_FEATURES,) and np.all(np.abs(v) <= 1.0 + 1e-6)


def test_one_v_one():
    env = Grid2v2(players_per_team=1, p_move=1.0); obs, _ = env.reset(seed=0)
    assert env.possible_agents == ["red_0", "blue_0"] and obs["red_0"]["mate"][0] == -1 and obs["red_0"]["opponents"].shape == (1, 4)


def test_pettingzoo_api():
    from pettingzoo.test import parallel_api_test
    parallel_api_test(Grid2v2(tiers={"red": "discrete", "blue": "box"}), num_cycles=100)
    parallel_api_test(Grid2v2(players_per_team=1, tiers="fine"), num_cycles=100)
