# teams/ — where your agent lives

This is the only folder you edit. `grid2v2/` is the engine and `baselines/` are the opponents you compare against;
both are the same for everyone and are replaced by the reference copies before the tournament.

## Submit one folder

Copy `teams/template/` to `teams/<team_name>/` (lower case, no spaces) and work there. That folder, zipped, is your
submission: `__init__.py` exposing `TIER` and `make_team`, plus whatever it needs (weights, helper modules, your
feature function), imported relatively so the folder works wherever it is copied. Hard limits: 50 MB; imports from the
standard library, numpy, torch, gymnasium, pettingzoo, stable_baselines3 and grid2v2 only; no network, no files
outside the folder, no threads or subprocesses. Training code belongs in your repo, not in the folder.

## Run it

    python -m grid2v2.match --red teams.<team_name> --blue baselines.scripted --games 10
    python -m grid2v2.match --red teams.<team_name> --blue baselines.scripted_box --games 10
    python -m grid2v2.tournament --dir teams --games 4          # every folder in teams/ against every other

Sides alternate; `W D L` and goals are from `--red`'s point of view. Add `--log match.json` to a single game to get a
replay file for the renderer.

## Budget

| what | limit | over it |
|---|---|---|
| `make_team()` + `reset()` | 15 s | forfeit before kick-off |
| `act()` | 50 ms, soft | STAY and a strike |
| `act()` | 1 s, hard | interrupted, STAY and a strike |
| exceptions in `act()` | | STAY and a strike |
| strikes | 10 per match | the eleventh forfeits (0-3) |

Times are measured on the machine running the match; the tournament runs on the reference machine (specified in the
project brief), so leave headroom or test with `--budget 0.025`. The match result reports your slowest `act()`.

## What your agent gets, and does not get

`act(obs)` receives your robot's observation only (see the arena README): positions, the ball and its velocity,
possession and stun flags, score, time, all in your frame (you attack towards +x). `make_team` receives the `env` so you can
read `env.rink` (geometry) and `env.adapters[agent_id]` (your action space); reading anything else from it, or from
the match runner, is cheating and will be checked. Your two robots may share a model, but they decide independently:
no shared mutable state between `act()` calls of different robots. The physics is public and deterministic, so
`grid2v2.physics.simulate` and `predict` are fair game for look-ahead.
