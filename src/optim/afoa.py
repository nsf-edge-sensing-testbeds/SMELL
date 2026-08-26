from __future__ import annotations

from dataclasses import dataclass
from math import gamma, pi, sin
from typing import Callable
import numpy as np


@dataclass
class AFOAConfig:
    population_size: int = 30
    max_iterations: int = 200
    w_max: float = 0.9
    w_min: float = 0.4
    c1: float = 2.0
    c2: float = 2.0
    levy_beta: float = 1.5
    levy_alpha: float = 0.05
    diversity_threshold: float = 0.05
    finite_difference_delta: float = 1e-3
    local_step_size: float = 0.05
    seed: int = 42


class AFOA:
    """Advanced Fennec Fox Optimization following the paper's stated update.

    This is a maximizer: larger fitness is better.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray, cfg: AFOAConfig):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _levy(self, shape: tuple[int, ...]) -> np.ndarray:
        beta = self.cfg.levy_beta
        sigma_u = (
            gamma(1 + beta) * sin(pi * beta / 2)
            / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = self.rng.normal(0, sigma_u, size=shape)
        v = self.rng.normal(0, 1, size=shape)
        return u / (np.abs(v) ** (1 / beta) + 1e-12)

    def _finite_difference_gradient(self, x: np.ndarray, fitness: Callable[[np.ndarray], float]) -> np.ndarray:
        delta = self.cfg.finite_difference_delta
        base = fitness(x)
        grad = np.zeros_like(x)
        for j in range(x.size):
            xp = x.copy()
            xp[j] = min(self.upper[j], xp[j] + delta)
            denom = max(xp[j] - x[j], 1e-12)
            grad[j] = (fitness(xp) - base) / denom
        return grad

    def optimize(self, fitness: Callable[[np.ndarray], float]) -> dict:
        c = self.cfg
        n, dim = c.population_size, self.lower.size
        pop = self.rng.uniform(self.lower, self.upper, size=(n, dim))
        scores = np.array([fitness(x) for x in pop])
        pbest = pop.copy()
        pbest_scores = scores.copy()
        g_idx = int(scores.argmax())
        gbest = pop[g_idx].copy()
        gbest_score = float(scores[g_idx])

        history = {"best_fitness": [], "diversity": []}
        for g in range(1, c.max_iterations + 1):
            w = c.w_max - (c.w_max - c.w_min) * (g / c.max_iterations) ** 2
            for i in range(n):
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                levy = c.levy_alpha * self._levy((dim,)) * (gbest - pop[i])
                proposal = (
                    w * pop[i]
                    + c.c1 * r1 * (pbest[i] - pop[i])
                    + c.c2 * r2 * (gbest - pop[i])
                    + levy
                )
                proposal = np.clip(proposal, self.lower, self.upper)
                score = fitness(proposal)
                pop[i] = proposal
                scores[i] = score
                if score > pbest_scores[i]:
                    pbest[i] = proposal
                    pbest_scores[i] = score
                if score > gbest_score:
                    gbest = proposal.copy()
                    gbest_score = float(score)

            center = pop.mean(axis=0)
            diversity = float(np.linalg.norm(pop - center, axis=1).mean())
            if diversity < c.diversity_threshold:
                grad = self._finite_difference_gradient(gbest, fitness)
                norm = np.linalg.norm(grad)
                if norm > 0:
                    proposal = np.clip(gbest + c.local_step_size * grad / norm, self.lower, self.upper)
                    score = fitness(proposal)
                    if score > gbest_score:
                        gbest, gbest_score = proposal, float(score)

            history["best_fitness"].append(gbest_score)
            history["diversity"].append(diversity)

        return {"best_position": gbest, "best_fitness": gbest_score, "history": history}
