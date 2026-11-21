"""grid2v2 — the AI Explorations project arena: a 2v2 ball game on a walled grid (PettingZoo Parallel API)."""
from .env import Grid2v2, TEAMS
from .actions import Intent, DiscreteKick, BoxKick, FineDiscrete, make_adapter, STAY, N, E, S, W
from .physics import Rink, Ball, simulate, predict, kick_range
from .features import flatten, N_FEATURES

__all__ = ["Grid2v2", "TEAMS", "Intent", "DiscreteKick", "BoxKick", "FineDiscrete", "make_adapter",
           "STAY", "N", "E", "S", "W", "Rink", "Ball", "simulate", "predict", "kick_range", "flatten", "N_FEATURES"]
