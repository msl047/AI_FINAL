"""tools/solo.py — single-robot training helpers (the same helpers HW4 ships as hw4util.py, without its test corridor).

    seed_everything(seed)            one call that seeds python, numpy and torch and makes torch deterministic
    make_env(name, **kw)             "cartpole" | "drill" | "solo"  -> a gymnasium env with a flat float observation
    evaluate_greedy(env_fn, act, n, seed)   mean return (and goal / save rates on the drill) of a greedy policy
    linear_schedule(t, start, end, steps)   a linear epsilon schedule
    make_env("solo", shaping=0.1, ...)      potential-based reward shaping on the 1v1 (see ChaseShaping)
    collect_teacher_data(env, teacher_act, n, eps, seed) -> X, A     (obs, teacher action) pairs for behaviour cloning
    clone_policy(policy, X, A, epochs, lr, seed) -> accuracy          cross-entropy fit of a policy's logits

Every environment here follows the gymnasium 1.x API: reset(seed=...) -> (obs, info); step(a) -> (obs, r, terminated,
truncated, info).  Observations are float32 vectors: 4 numbers on CartPole, 23 on the rink (grid2v2.features.flatten).
"""
from __future__ import annotations
import os
import random
from typing import Callable

import numpy as np
import gymnasium as gym

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from grid2v2.features import flatten, N_FEATURES
from grid2v2.single import ShootingDrill, SoloEnv


def seed_everything(seed: int) -> None:
    """Seed every random source this homework touches.  Call it once at the top of a training run."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def linear_schedule(t: int, start: float, end: float, steps: int) -> float:
    """start at t = 0, end at t >= steps, linear in between."""
    if steps <= 0: return end
    return float(max(end, start + (end - start) * min(t, steps) / steps))


class FlatObs(gym.ObservationWrapper):
    """The rink's observation dict -> the 23-float vector of grid2v2.features.flatten."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(N_FEATURES,), dtype=np.float32)
        self.rink = env.rink

    def observation(self, obs):
        return flatten(obs, self.rink.width, self.rink.height)


class ChaseShaping(gym.Wrapper):
    """Potential-based shaping for the 1v1 (Phase 2): r' = r + scale * (gamma * phi(s') - phi(s)) with
    phi = (ball x towards the attacked goal) - 0.5 * (my distance to a ball I do not hold), both scaled to [0, 1].
    Potential-based shaping leaves the optimal policy unchanged (Ng, Harada and Russell 1999); it only adds signal to
    the 600-step episodes whose own reward is +-1 per goal.  `info["raw_reward"]` keeps the unshaped reward."""

    def __init__(self, env, scale: float = 0.1, gamma: float = 0.99):
        super().__init__(env); self.scale, self.gamma, self.rink = scale, gamma, env.rink
        self._phi_prev = 0.0

    def _phi(self, o):
        bx = o["ball"][0] / (self.rink.width - 1)
        mine = o["self"][2] > 0.5
        d = (abs(o["self"][0] - o["ball"][0]) + abs(o["self"][1] - o["ball"][1])) / (self.rink.width + self.rink.height)
        return float(bx - (0.0 if mine else 0.5 * d))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw); self._phi_prev = self._phi(obs); return obs, info

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        phi = self._phi(obs); info["raw_reward"] = r
        r = r + self.scale * ((0.0 if term else self.gamma * phi) - self._phi_prev)
        self._phi_prev = phi
        return obs, r, term, trunc, info


def make_env(name: str, shaping: float = 0.0, **kw) -> gym.Env:
    """cartpole: CartPole-v1.  drill: ShootingDrill with flat observations.  solo: 1v1 vs `opponent` (default scripted),
    optionally with ChaseShaping(scale=shaping)."""
    if name == "cartpole":
        return gym.make("CartPole-v1", **kw)
    if name == "drill":
        return FlatObs(ShootingDrill(**kw))
    if name == "solo":
        env = SoloEnv(**kw)
        if shaping: env = ChaseShaping(env, scale=shaping)
        return FlatObs(env)
    raise ValueError(f"unknown env {name!r}: use cartpole, drill or solo")


def evaluate_greedy(env_fn: Callable[[], gym.Env], act: Callable[[np.ndarray], int], n: int = 20, seed: int = 10_000) -> dict:
    """Run n episodes with `act(obs) -> action` (no exploration) on fresh envs seeded seed, seed+1, ...

    Returns {"return": mean undiscounted return, "goals": fraction of episodes ending +1, "saves": fraction ending -1,
             "length": mean episode length}.  goals/saves are meaningful on the drill; on CartPole look at return."""
    env = env_fn()
    rets, lens, goals, saves = [], [], 0, 0
    for k in range(n):
        obs, _ = env.reset(seed=seed + k)
        done, R, T = False, 0.0, 0
        while not done:
            obs, r, term, trunc, _ = env.step(int(act(obs)))
            R += r; T += 1; done = term or trunc
        rets.append(R); lens.append(T); goals += r > 0; saves += r < 0
    env.close()
    return {"return": float(np.mean(rets)), "goals": goals / n, "saves": saves / n, "length": float(np.mean(lens))}


def collect_teacher_data(env: gym.Env, teacher_act: Callable, n: int = 20_000, eps: float = 0.2, seed: int = 0):
    """Run `teacher_act(raw_obs) -> action` in `env` (a make_env("drill") env: the teacher sees the rink's raw observation
    dict through env.unwrapped.env.observe("red_0"), the data are the flat observations).  With probability eps a random
    action is executed instead, so the data cover states the teacher alone would not visit; the LABEL is always the
    teacher's action.  Returns X (n, 23) float32 and A (n,) int64."""
    rng = np.random.default_rng(seed)
    X, A = [], []
    obs, _ = env.reset(seed=seed)
    while len(X) < n:
        a_t = int(teacher_act(env.unwrapped.env.observe("red_0")))
        X.append(np.asarray(obs, np.float32)); A.append(a_t)
        a = a_t if rng.random() >= eps else int(rng.integers(env.action_space.n))
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
    return np.stack(X), np.asarray(A, np.int64)


def clone_policy(policy, X: np.ndarray, A: np.ndarray, epochs: int = 30, lr: float = 1e-3, batch_size: int = 128, seed: int = 0) -> float:
    """Behaviour cloning: minimise cross-entropy between policy(X)[0] (the logits) and the teacher's actions A, in place.
    Works for any nn.Module whose forward returns (logits, value).  Returns the training accuracy."""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    Xt = torch.as_tensor(np.asarray(X, np.float32)); At = torch.as_tensor(np.asarray(A, np.int64))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(epochs):
        perm = rng.permutation(len(Xt))
        for i in range(0, len(perm), batch_size):
            idx = torch.as_tensor(perm[i:i + batch_size])
            logits, _ = policy(Xt[idx])
            loss = F.cross_entropy(logits, At[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((policy(Xt)[0].argmax(1) == At).float().mean())
