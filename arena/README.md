# grid2v2: the AI Explorations project arena (v0.1, engine only)

A 2v2 ball game on a walled 21 x 11 grid, in the PettingZoo Parallel API. Red attacks to the right, blue to the left.
Score by kicking the ball across the goal mouth (the 3 middle rows of each end wall). Most goals after 600 steps wins.
Walls everywhere, no out of bounds: a wild shot rebounds into play. Whether the renderer draws a pitch or a rink is a
later decision; nothing underneath changes.

## Rules

- **Robots** live on cells and move N, E, S, W with the course slip: 0.8 the commanded way, 0.1 to each side. Two robots
  never share a cell; swaps and head-ons are blocked.
- **The ball** is carried by the robot that holds it. *Dribbling is slow*: a carrier's move only happens with
  probability 0.5. A kick sends the ball at a speed s in cells per step; each step it travels s cells and then loses
  1, so it covers s + (s-1) + ... cells (speed 3: 6 cells, speed 2: 3). It bounces off walls. The **first robot in its
  path takes it**, whichever team: a pass can be cut out, a shot can be blocked. A loose ball is picked up by
  stepping onto it. The kicker cannot take its own kick back while it is still moving (no stalling off the boards).
- **The ball moves first.** Within a step the ball flies before the robots move, so nobody can react to a kick in the
  same step it is made: a keeper saves what it is already standing in front of, or what it reaches before the next
  step. A straight speed-3 shot from the crease edge (x = 18) or one column back is therefore unstoppable unless a
  defender is already on its row; from farther out the keeper gets a step.
- **Tackle.** Moving into an opponent who holds the ball takes it with probability 0.5. **A failed tackle stuns you
  for 1 step** (your next move is ignored). Tackling is not free: a chaser who lunges every step spends a quarter of
  his time standing still.
- **A pass needs room.** A teammate standing next to the kicker cannot receive the kick; the ball passes through him.
  Every pass therefore crosses at least one open cell, where a defender can stand.
- **Crease.** Nobody may enter the 2 columns in front of the goal they attack (6 cells); every goal is a shot from range.
  The keeper may stand there.
- **Goal.** +1 to each robot of the scoring team, -1 to each robot of the other. Nothing else is rewarded: shaping is
  yours to add in your own wrapper, and yours to defend in the design review. After a goal both teams return to their
  kick-off cells and the conceding team's robot 0 holds the ball. The match starts with the ball loose at centre.

## What every robot sees (the same for everyone, in its own frame: you always attack towards +x)

| key | contents |
|---|---|
| `self` | `[x, y, has_ball, stunned]` (stunned = steps left) |
| `mate` | `[x, y, has_ball, stunned]` of your teammate (all -1 in 1v1) |
| `opponents` | 2 x `[x, y, has_ball, stunned]` |
| `ball` | `[x, y, vx, vy]`, floats; velocity in cells per step, zero when held |
| `score` | `[own, theirs]` |
| `t` | `[t]` |

`grid2v2.features.flatten(obs)` is the full 23-float vector. Choosing a subset, or adding derived features (distance to
goal, is the lane to my teammate clear, ...) is a design decision; the physics is public and deterministic
(`grid2v2.physics.simulate`, `predict`), so looking ahead at your own kicks is legal.

## Actions: pick your tier

Every action decodes to the same thing, a move or a kick (angle, speed), so teams with different action spaces
play each other on the same rink. The standings record the tier next to your name.

| tier | space | meaning |
|---|---|---|
| 1 `"discrete"` | `Discrete(9)` | STAY, N, E, S, W, kick N, kick E, kick S, kick W at speed 3 (range 6). Kick without the ball: a wasted step. |
| 2 `"box"` | `Box(4)` in [-1, 1] | `(mx, my, kx, ky)`. With the ball and `|k| > 0.2`: kick at angle `atan2(ky, kx)`, speed `1 + 2 min(1, |k|)` (1.4 to 3). Otherwise move along the dominant axis of `(mx, my)`, STAY inside the deadzone. |
| 1.5 `"fine"` | `Discrete(53)` | an example middle rung: 16 directions x 3 speeds; see `grid2v2/actions.py` and write your own `Adapter`. |

Tier 1 is the HW4 space and what DQN wants; Tier 2 is what PPO and SAC want, and it buys angled shots, bank shots,
soft passes that stop at the receiver and passes that lead a runner. Both tiers top out at speed 3: the Tier 2 edge
is aim, not power. A straight shot is blocked by a keeper on its row; an angled one needs the keeper on the row it
will *arrive* at, and that is why the Tier 2 baseline scores where the Tier 1 baseline waits for a pass.

## Run it

    pip install pettingzoo stable-baselines3 supersuit   # on top of the course conda env (gymnasium, numpy, torch are in it)
    python -m pytest tests -q                       # 39 tests: physics, adapters, rules, PettingZoo's API check, the budget
    python -m grid2v2.match --red teams.template --blue baselines.scripted --games 10
    python -m grid2v2.match --red baselines.scripted_box --blue baselines.scripted --games 20
    python -m grid2v2.match --red baselines.scripted --blue baselines.random --games 1 --log match.json   # for the renderer
    python -m grid2v2.tournament --dir teams --games 4          # every team folder plus the three baselines
    python tools/export_team.py --help              # SB3 model -> teams/<name>/ (numpy only, plays within the budget)
    python tools/export_policy.py --help            # a torch state_dict -> teams/<name>/ the same way
    python -m grid2v2.render match.json --out match.mp4 --skin rink        # the broadcast view (needs Pillow; MP4 needs imageio[ffmpeg])
    python -m grid2v2.render match.json --out goals.gif --highlights       # 25 steps around every goal; --range a:b for a stretch

## Watching

Training never renders. Record a match with `--log`, then render it: robots glide cell to cell, the ball follows its
true path (bank shots bend at the wall), banners name shots, passes, clearances, tackles, saves and goals, and the
scoreboard tracks possession. `--skin rink` (default) or `pitch`; `--interp` sub-frames per step (2), `--fps` (10),
`--cell` pixels per cell (48). A full match is 1,200 frames: write MP4 (about 2 MB) rather than GIF (20 MB+); GIFs are
for `--highlights` and `--range`. `Grid2v2(render_mode="rgb_array").render()` gives the same frame live, for notebooks.

## Write a team: one folder

Copy `teams/template/` to `teams/<team_name>/`; that folder is the submission (see `teams/README.md` for the full
details). It exposes `TIER` and `make_team(team, agent_ids, env) -> {agent_id: Agent}`; an `Agent` has `reset()` and
`act(obs) -> action` in its tier's space and may keep per-robot state. Weights and helpers live in the folder and are
imported relatively. Use `env.rink` and `env.adapters[agent_id]`, nothing else from the environment. Teammates
communicate only through what they both observe. The baseline to beat is `baselines.scripted` (a field player and a
keeper); `baselines.scripted_box` is the same policy on Tier 2; `baselines.random` is the floor; `baselines.learned`,
when present, is a PPO-trained Tier 1 team (it is retrained after every rule change and dropped until it is).

**Budget** (Berkeley Pacman rules, scaled): `make_team` + `reset` 15 s, hard; `act` 50 ms soft (STAY and a strike),
1 s hard (interrupted, STAY and a strike); an exception is a strike; the eleventh strike in a match forfeits it 0-3.
Measured on the machine running the match; the tournament runs on the reference machine. For scale: a `simulate`
call is about 8 us and a 23-128-128-9 MLP forward pass about 25 us, so 50 ms buys a few hundred look-aheads.

## Layout

    grid2v2/physics.py     the ball: ray-march, bounces, goals, interceptions, simulate/predict (pure functions)
    grid2v2/actions.py     Intent and the action tiers (DiscreteKick, BoxKick, FineDiscrete); subclass Adapter for your own
    grid2v2/env.py         the rules (Grid2v2, ParallelEnv); state() for logs; ansi render
    grid2v2/agents.py      Agent, RandomAgent, ScriptedAgent (field player + keeper)
    grid2v2/features.py    flatten()
    grid2v2/single.py      ShootingDrill (one robot vs a keeper, 693 states) and SoloEnv (1v1 vs a team module), gymnasium API
    grid2v2/match.py       matches, the budget, strikes, forfeits, JSON logs
    grid2v2/tournament.py  round robin, both sides, Elo, standings, --dir discovery
    grid2v2/render.py      the broadcast view from a log: Pillow frames, GIF/MP4; rgb_array for live use
    baselines/             random, scripted, scripted_box, and learned when retrained (not yours to edit)
    teams/                 template/ (copy me) and README.md (how a team folder is put together)
    tools/                 helpers: vs_env (the pair as a PettingZoo env with flat observations, for SB3 + SuperSuit),
                           solo (single-robot envs and helpers), export_team / export_policy (a model -> team folder)
    tests/                 the rules and the budget
