import argparse
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arena"))

from baselines import scripted_box
from grid2v2 import Grid2v2
from grid2v2.agents import ScriptedAgent
from grid2v2.physics import Ball
from teams.team2 import (
    FINISHING_X,
    MIDFIELD_X,
    N_CHOICES,
    N_INPUTS,
    TEAM_DIR,
    USE_SCRIPT,
    Team2Field,
    features,
    phase_at,
)

START_X = {
    "defense": (2, MIDFIELD_X),
    "midfield": (MIDFIELD_X, FINISHING_X),
    "finishing": (FINISHING_X, 19),
}

TARGET_X = {
    "defense": MIDFIELD_X,
    "midfield": FINISHING_X,
    "finishing": None,
}


class PossessionDrill(gym.Env):
    def __init__(self, phase):
        self.phase = phase
        self.action_space = gym.spaces.Discrete(N_CHOICES)
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(N_INPUTS,), dtype=np.float32
        )
        self.arena = Grid2v2(tiers="box", max_steps=25)
        self.field = Team2Field(self.arena.adapters["red_0"], self.arena.rink)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.arena.reset(seed=int(self.np_random.integers(2**31)))
        rink = self.arena.rink
        x = int(self.np_random.integers(*START_X[self.phase]))
        y = int(self.np_random.integers(rink.height))

        self.arena.pos["red_0"] = (x, y)
        self.arena.pos["red_1"] = (0, rink.height // 2)
        self.arena.pos["blue_0"] = (min(x + 2, rink.width - 2), y)
        self.arena.pos["blue_1"] = (
            rink.width - 1,
            int(self.np_random.choice(rink.goal_rows)),
        )

        # Practice passing with a reachable teammate on some starts.
        if self.phase != "finishing" and self.np_random.random() < 0.5:
            mate_rows = [row for row in (y - 2, y + 2) if 0 <= row < rink.height]
            self.arena.pos["red_1"] = (x + 3, int(self.np_random.choice(mate_rows)))

        self.arena.holder = "red_0"
        self.arena.ball = Ball.at_cell(x, y)
        self.obs = {name: self.arena.observe(name) for name in self.arena.agents}

        self.others = {
            "red_1": ScriptedAgent(self.arena.adapters["red_1"], rink, index=1),
            **scripted_box.make_team("blue", self.arena.members["blue"], self.arena),
        }
        self.field.reset()
        for agent in self.others.values():
            agent.reset()

        return features(self.obs["red_0"], rink), {}

    def step(self, choice):
        field_obs = self.obs["red_0"]
        if phase_at(field_obs["self"][0]) != self.phase:
            choice = USE_SCRIPT

        actions = {"red_0": self.field.choose_action(field_obs, int(choice))}
        actions.update({name: bot.act(self.obs[name]) for name, bot in self.others.items()})
        self.obs, rewards, _, truncated, _ = self.arena.step(actions)

        goal = float(rewards["red_0"])
        holder = self.arena.holder
        target = TARGET_X[self.phase]
        lost_ball = holder in self.arena.members["blue"]
        advanced = (
            target is not None
            and holder in self.arena.members["red"]
            and self.arena.pos[holder][0] >= target
        )

        reward = goal
        if goal == 0.0:
            if lost_ball:
                reward = -1.0
            elif advanced:
                reward = 1.0
            else:
                reward = -0.01

        finished = bool(goal or lost_ball or advanced or truncated["red_0"])
        return features(self.obs["red_0"], self.arena.rink), reward, finished, False, {}


def train_specialist(phase, steps, seed):
    torch.set_num_threads(1)
    model = PPO(
        "MlpPolicy",
        PossessionDrill(phase),
        device="cpu",
        seed=seed,
        n_steps=1024,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        policy_kwargs={"net_arch": [32], "activation_fn": torch.nn.Tanh},
        verbose=1,
    )
    with torch.no_grad():
        model.policy.action_net.bias[USE_SCRIPT] = 1.0
    model.learn(total_timesteps=steps)

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    model.save(models / f"team2_{phase}_seed{seed}.zip")

    hidden = model.policy.mlp_extractor.policy_net[0]
    output = model.policy.action_net
    np.savez(
        TEAM_DIR / f"{phase}.npz",
        w1=hidden.weight.detach().cpu().numpy(),
        b1=hidden.bias.detach().cpu().numpy(),
        w2=output.weight.detach().cpu().numpy(),
        b2=output.bias.detach().cpu().numpy(),
    )


def train_playing_out_of_defense(steps, seed):
    train_specialist("defense", steps, seed)


def train_midfield_buildup(steps, seed):
    train_specialist("midfield", steps, seed)


def train_finishing(steps, seed):
    train_specialist("finishing", steps, seed)


if __name__ == "__main__":
    trainers = {
        "defense": train_playing_out_of_defense,
        "midfield": train_midfield_buildup,
        "finishing": train_finishing,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=trainers)
    parser.add_argument("--steps", type=int, default=32_768)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    trainers[args.phase](args.steps, args.seed)