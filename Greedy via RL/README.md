# NOTE
Technically not Greedy fully, only the heuristic is. The heuristic is only used during the action selection phase, but the Q-values are still updated using the standard Q-learning update rule, which is not a greedy approach. I'll need to rework the `train()` function over break.
