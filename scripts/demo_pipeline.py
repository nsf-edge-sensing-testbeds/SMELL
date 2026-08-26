from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

"""Fast smoke-test of the non-CNN core: cross-attention, HMM, scheduler, AFOA."""

import numpy as np
import torch

from src.models.cross_modal import BidirectionalCrossAttention
from src.models.proficiency_hmm import ProficiencyHMM
from src.curriculum import CurriculumScheduler, simulated_proficiency_gain
from src.optim.afoa import AFOA, AFOAConfig


def main():
    torch.manual_seed(0)
    cross = BidirectionalCrossAttention(dim=64, heads=8, fusion_dim=32)
    visual = torch.randn(4, 49, 64)
    text = torch.randn(4, 12, 64)
    mask = torch.ones(4, 12, dtype=torch.bool)
    z = cross(visual, text, mask)
    print("fused embedding:", tuple(z.shape))

    hmm = ProficiencyHMM(z_dim=32, num_states=5, state_dim=16)
    belief = hmm.initial_distribution().unsqueeze(0).expand(4, -1)
    obs = torch.tensor([1, 0, 1, 2])
    belief = hmm.filter_step(belief, z, obs)
    target = hmm.target_difficulty(belief, explore=False)
    print("posterior shape:", tuple(belief.shape), "target difficulty:", target.tolist())

    item_d = torch.linspace(0, 1, 20)
    selected = CurriculumScheduler(top_k=3).select(item_d, target[0])
    print("selected item ids:", selected.tolist())

    cfg = AFOAConfig(population_size=8, max_iterations=8, seed=0)
    opt = AFOA(np.array([0, 0, 0.1, 0.5]), np.array([0.2, 0.2, 2.0, 2.0]), cfg)
    d = np.linspace(0, 1, 100)
    result = opt.optimize(lambda th: simulated_proficiency_gain(th, d, seed=0))
    print("AFOA best fitness:", result["best_fitness"])
    print("AFOA best theta:", result["best_position"])


if __name__ == "__main__":
    main()
