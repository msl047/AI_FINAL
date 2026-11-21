"""teams/template/agent.py — your agent.  Replace the body of `act`; keep the interface.

Budget: 50 ms per act() on the reference machine (a physics.simulate call costs about 8 us, a 23-128-128-9 MLP forward
pass about 25 us, so there is room for a few hundred look-aheads or a small search).  Over 50 ms, or an exception, is
STAY and a strike; eleven strikes forfeit the match.  Load weights in `load_shared`, not in `act`.
"""

import math

from grid2v2.agents import ScriptedAgent, manhattan
from grid2v2.physics import kick_range

KICK_SPEEDS = (1.5, 2.0, 3.0)
KICK_ANGLES = tuple(math.tau * index / 16 for index in range(16))


class FieldAgent(ScriptedAgent):
    def _kick_candidates(self, me, mate):
        goal_angles = tuple(
            math.atan2(y - me[1], x - me[0]) for x, y in self._goal_targets()
        )
        shot_angles = tuple(dict.fromkeys((*goal_angles, *KICK_ANGLES)))
        goal_distance = self.rink.width - 0.5 - me[0]
        shots = []
        for angle in shot_angles:
            forward = math.cos(angle)
            for speed in KICK_SPEEDS:
                if forward * kick_range(speed) > goal_distance:
                    shots.append((angle, speed))

        shots.sort(key=lambda kick: kick[1] * math.cos(kick[0]), reverse=True)

        if not any(abs(angle) < 1e-6 for angle, _ in shots):
            shots.append((0.0, KICK_SPEEDS[-1]))

        passes = []
        if mate is not None and manhattan(me, mate) >= self.min_pass:
            dx, dy = mate[0] - me[0], mate[1] - me[1]
            pass_angle = math.atan2(dy, dx)
            pass_angles = tuple(dict.fromkeys((pass_angle, *KICK_ANGLES)))
            distance_to_cell = math.hypot(max(abs(dx) - 0.5, 0), max(abs(dy) - 0.5, 0))
            passes = [
                (angle, speed)
                for angle in pass_angles
                for speed in KICK_SPEEDS
                if kick_range(speed) > distance_to_cell
            ]
        return shots, passes