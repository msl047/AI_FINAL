"""baselines — the teams you compare against (read them, run them, do not edit them).

    baselines.random         uniformly random Discrete(9) actions
    baselines.scripted       Tier 1: a field player and a keeper, shots and passes chosen with physics.simulate (beat this)
    baselines.scripted_box   the same policy on Tier 2 (aimed shots and passes)
    baselines.learned        (when present) a PPO-trained Tier 1 team with numpy weights; retrained after rule changes
"""
