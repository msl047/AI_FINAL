"""Compare midfield PPO checkpoints without replacing Team 2's weights."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arena"))

from baselines import scripted_box
from grid2v2 import Ball, Grid2v2
from grid2v2.actions import E, N, S
from grid2v2.agents import manhattan
from grid2v2.match import play, series
from teams import team2

N_ACTIONS = 4
ROLLOUT_STEPS = 1024


def is_midfield_possession(obs):
    x, _, has_ball, stunned = obs["self"]
    return team2.phase_at(x) == "midfield" and has_ball > 0.5 and stunned < 0.5


def tactical_actions(field, obs):
    script = field.choose_action(obs, team2.USE_SCRIPT)
    me, _, mate, _, opponents, *_ = field._parse(obs)

    def clearance(cell):
        return min((manhattan(cell, pos) for pos, _ in opponents), default=10)

    east = (me[0] + 1, me[1])
    forward = field.choose_action(obs, E) if clearance(east) >= 2 else script

    sidestep = script
    best_clearance = clearance(me)
    for move, dy in ((N, 1), (S, -1)):
        distance = clearance((me[0], me[1] + dy))
        action = field.choose_action(obs, move)
        if distance > best_clearance and np.array_equal(
            action, field.adapter.encode_move(move)
        ):
            sidestep, best_clearance = action, distance

    pass_forward = script
    if mate is not None and mate[0] > me[0] and clearance(mate) >= 2:
        pass_forward = field.choose_action(obs, team2.PASS)

    return [script, forward, sidestep, pass_forward]


def action_mask(actions):
    return np.array([
        True,
        *(not np.array_equal(action, actions[0]) for action in actions[1:]),
    ])


def collect_midfield_starts(seed):
    starts = []
    with TemporaryDirectory() as temp:
        replay = Path(temp) / "match.json"
        for index in range(8):
            result = play(
                "teams.team1", scripted_box, seed=100 * seed + index, log=replay
            )
            if result["forfeit"]:
                raise RuntimeError(result["forfeit"])

            frames = json.loads(replay.read_text(encoding="utf-8"))["frames"]
            starts.extend(
                frame for frame in frames[::4]
                if frame["holder"] == "red_0"
                and team2.phase_at(frame["pos"]["red_0"][0]) == "midfield"
                and not frame["stunned"].get("red_0")
            )

    if not starts:
        raise RuntimeError("No midfield possessions were collected.")
    return starts


class MidfieldFromMatches(gym.Env):
    def __init__(self, starts):
        self.starts = starts
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(
            -1, 1, (team2.N_INPUTS,), dtype=np.float32
        )
        self.arena = Grid2v2(tiers="box", max_steps=80)

    def refresh(self):
        obs = self.obs["red_0"]
        if is_midfield_possession(obs):
            self.actions = tactical_actions(self.field, obs)
        else:
            self.actions = [self.field.act(obs)] * N_ACTIONS
        return team2.features(obs, self.arena.rink)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.arena.reset(seed=int(self.np_random.integers(2**31)))
        frame = self.starts[self.np_random.integers(len(self.starts))]
        self.arena.pos.update(
            {name: tuple(cell) for name, cell in frame["pos"].items()}
        )
        self.arena.stunned.update(frame["stunned"])
        self.arena.holder = "red_0"
        self.arena.ball = Ball.at_cell(*self.arena.pos["red_0"])
        self.obs = {name: self.arena.observe(name) for name in self.arena.agents}

        self.others = team2.make_team("red", self.arena.members["red"], self.arena)
        self.field = self.others.pop("red_0")
        self.others.update(
            scripted_box.make_team("blue", self.arena.members["blue"], self.arena)
        )
        for bot in (self.field, *self.others.values()):
            bot.reset()
        return self.refresh(), {}

    def action_masks(self):
        return action_mask(self.actions)

    def step(self, choice):
        actions = {"red_0": self.actions[int(choice)]}
        actions.update(
            {name: bot.act(self.obs[name]) for name, bot in self.others.items()}
        )
        self.obs, rewards, _, truncated, _ = self.arena.step(actions)
        reward = float(rewards["red_0"])
        finished = reward != 0.0 or truncated["red_0"]
        return self.refresh(), reward, finished, False, {}


def experimental_team(model, masked, counts):
    module = ModuleType("team2_midfield_experiment")
    module.TIER = team2.TIER
    hidden = model.policy.mlp_extractor.policy_net[0]
    output = model.policy.action_net
    w1, b1, w2, b2 = (
        tensor.detach().cpu().numpy()
        for tensor in (hidden.weight, hidden.bias, output.weight, output.bias)
    )

    class MidfieldBot(team2.Team2Field):
        def act(self, obs):
            if not is_midfield_possession(obs):
                return super().act(obs)

            options = tactical_actions(self, obs)
            mask = action_mask(options)
            inputs = team2.features(obs, self.rink)
            logits = w2 @ np.tanh(w1 @ inputs + b1) + b2
            if masked:
                logits = np.where(mask, logits, -np.inf)

            weights = np.exp(logits - logits.max())
            choice = int(self.policy_rng.choice(N_ACTIONS, p=weights / weights.sum()))
            counts["decisions"] += 1
            counts["overrides"] += int(choice != 0 and mask[choice])
            return options[choice]

    def make_team(side, ids, env):
        agents = team2.make_team(side, ids, env)
        original = agents[ids[0]]
        agents[ids[0]] = MidfieldBot(
            original.adapter, original.rink, policies=original.policies
        )
        return agents

    module.make_team = make_team
    return module


def evaluate(label, model, masked, args):
    rows = []
    for opponent in ("teams.team1", "baselines.scripted_box"):
        counts = {"decisions": 0, "overrides": 0}
        team = "teams.team2" if model is None else experimental_team(model, masked, counts)
        result = series(team, opponent, games=args.games, seed=args.eval_seed)
        row = {"checkpoint": label, "opponent": opponent, **result}
        message = (
            f"{label} vs {opponent}: {result['W']}-{result['D']}-{result['L']} W-D-L, "
            f"{result['GF']}-{result['GA']} goals, strikes={result['strikes']['red']}"
        )
        if model is not None:
            row.update(counts)
            message += f", {counts['overrides']}/{counts['decisions']} actual midfield overrides"
        rows.append(row)
        print(message, flush=True)
    return rows


def run_experiment(args):
    torch.set_num_threads(1)
    masked = args.variant == "masked"
    starts = collect_midfield_starts(args.seed)
    print(f"Collected {len(starts)} midfield starts.", flush=True)

    model_class = MaskablePPO if masked else PPO
    model = model_class(
        "MlpPolicy",
        MidfieldFromMatches(starts),
        device="cpu",
        seed=args.seed,
        n_steps=ROLLOUT_STEPS,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [32], "activation_fn": torch.nn.Tanh},
        verbose=1,
    )

    folder = ROOT / "models" / "midfield_tests" / f"{args.variant}_seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    results = evaluate("current_team2", None, masked, args)

    for steps in args.steps:
        model.learn(total_timesteps=steps - model.num_timesteps, reset_num_timesteps=False)
        model.save(folder / f"steps_{steps}.zip")
        results.extend(evaluate(steps, model, masked, args))
        (folder / "results.json").write_text(
            json.dumps({"settings": vars(args), "matches": results}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("masked", "ppo"), default="masked")
    parser.add_argument("--steps", type=int, nargs="+", default=[16_384, 65_536, 131_072])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1000)
    args = parser.parse_args()

    if args.steps != sorted(set(args.steps)) or any(
        s <= 0 or s % ROLLOUT_STEPS for s in args.steps
    ):
        parser.error("--steps must be increasing multiples of 1024")
    if args.games < 2:
        parser.error("--games must be at least 2 so teams play both sides")

    run_experiment(args)