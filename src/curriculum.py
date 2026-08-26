from __future__ import annotations

import numpy as np
import torch


class CurriculumScheduler:
    def __init__(self, top_k: int = 1):
        self.top_k = top_k

    @staticmethod
    def rank_by_target(item_difficulties: torch.Tensor, target: float | torch.Tensor) -> torch.Tensor:
        if isinstance(target, torch.Tensor):
            target = float(target.detach().cpu().item())
        return torch.argsort(torch.abs(item_difficulties - target))

    def select(self, item_difficulties: torch.Tensor, target: float | torch.Tensor) -> torch.Tensor:
        return self.rank_by_target(item_difficulties, target)[: self.top_k]


def simulated_proficiency_gain(
    theta: np.ndarray,
    item_difficulties: np.ndarray,
    initial_level: float = 0.4,
    steps: int = 100,
    seed: int = 42,
) -> float:
    """Offline proxy used only to make the AFOA pipeline executable.

    theta = [difficulty_increment, exploration_sigma, transition_concentration,
             learning_rate_scale].

    The article does not publish the exact simulator formula, so this function
    is deliberately marked as an engineering proxy rather than claimed as the
    paper's undisclosed implementation.
    """
    inc, explore, concentration, lr_scale = theta
    rng = np.random.default_rng(seed)
    level = float(initial_level)
    completion = 0.0
    for _ in range(steps):
        target = np.clip(level + inc + rng.normal(0, explore), 0, 1)
        idx = np.argmin(np.abs(item_difficulties - target))
        mismatch = abs(item_difficulties[idx] - level)
        p_success = 1.0 / (1.0 + np.exp(8.0 * (mismatch - 0.25)))
        success = rng.random() < p_success
        if success:
            level = min(1.0, level + 0.01 * lr_scale / max(concentration, 0.1))
            completion += 1.0
        else:
            level = max(0.0, level - 0.002)
    gain = level - initial_level
    rcp = completion / steps
    return float(gain + 0.25 * rcp)
