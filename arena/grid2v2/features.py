"""grid2v2/features.py — the observation as one float vector, for function approximation.

This is the *full* feature set.  Choosing a subset (or adding derived features: distance to goal, is the lane to my
teammate clear, ...) is a design decision for your team; `flatten` is the baseline you compare against.
"""
import numpy as np

N_FEATURES = 23


def flatten(obs: dict, width: int = 21, height: int = 11, max_steps: int = 600, max_speed: float = 4.0) -> np.ndarray:
    """[self x y has stunned] [mate ...] [opp0 ...] [opp1 ...] [ball x y vx vy] [score own opp] [t]  -> 23 floats in ~[-1, 1]."""
    def cell(row):
        if row[0] < 0: return [-1.0, -1.0, -1.0, -1.0]
        return [row[0] / (width - 1), row[1] / (height - 1), float(row[2]), min(float(row[3]), 1.0)]
    opps = list(obs["opponents"])
    while len(opps) < 2: opps.append(np.array([-1, -1, -1, -1], np.float32))
    b = obs["ball"]
    v = cell(obs["self"]) + cell(obs["mate"]) + cell(opps[0]) + cell(opps[1]) \
        + [b[0] / (width - 1), b[1] / (height - 1), b[2] / max_speed, b[3] / max_speed] \
        + [obs["score"][0] / 5.0, obs["score"][1] / 5.0] + [obs["t"][0] / max_steps]
    return np.asarray(v, dtype=np.float32)
