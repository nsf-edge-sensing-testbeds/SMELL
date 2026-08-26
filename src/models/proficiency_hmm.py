from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ProficiencyHMM(nn.Module):
    """Discrete 5-state proficiency HMM with neural emissions conditioned on item z.

    Observations: 0=incorrect, 1=correct, 2=skipped.
    """

    def __init__(
        self,
        z_dim: int = 256,
        num_states: int = 5,
        state_dim: int = 32,
        state_difficulties: list[float] | None = None,
        exploration_sigma: float = 0.05,
    ):
        super().__init__()
        self.num_states = num_states
        self.exploration_sigma = exploration_sigma
        self.state_embedding = nn.Embedding(num_states, state_dim)
        self.emission = nn.Sequential(
            nn.Linear(state_dim + z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        # Near-diagonal initialization encourages gradual proficiency changes.
        A = torch.eye(num_states) * 2.0 + torch.ones(num_states, num_states) * 0.2
        self.transition_logits = nn.Parameter(A)
        self.initial_logits = nn.Parameter(torch.zeros(num_states))
        if state_difficulties is None:
            state_difficulties = torch.linspace(0.2, 1.0, num_states).tolist()
        self.register_buffer("mu", torch.tensor(state_difficulties, dtype=torch.float))

    def transition_matrix(self) -> torch.Tensor:
        return F.softmax(self.transition_logits, dim=-1)

    def initial_distribution(self) -> torch.Tensor:
        return F.softmax(self.initial_logits, dim=-1)

    def emission_prob(self, z_t: torch.Tensor) -> torch.Tensor:
        # z_t: [B, Dz] -> [B,K,3]
        b = z_t.shape[0]
        state_ids = torch.arange(self.num_states, device=z_t.device)
        e = self.state_embedding(state_ids).unsqueeze(0).expand(b, -1, -1)
        z = z_t.unsqueeze(1).expand(-1, self.num_states, -1)
        return F.softmax(self.emission(torch.cat([e, z], dim=-1)), dim=-1)

    def filter_step(self, belief: torch.Tensor, z_t: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        # belief [B,K], z_t [B,D], observation [B]
        A = self.transition_matrix()
        pred = belief @ A
        emit = self.emission_prob(z_t)
        obs_prob = emit.gather(2, observation[:, None, None].expand(-1, self.num_states, 1)).squeeze(-1)
        posterior = pred * obs_prob
        posterior = posterior / posterior.sum(-1, keepdim=True).clamp_min(1e-12)
        return posterior

    def target_difficulty(self, belief: torch.Tensor, explore: bool = True) -> torch.Tensor:
        d = belief @ self.mu
        if explore and self.exploration_sigma > 0:
            d = d + torch.randn_like(d) * self.exploration_sigma
        return d.clamp(0.0, 1.0)

    def sequence_nll(self, z: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Differentiable forward-algorithm NLL. z [B,T,D], obs [B,T]."""
        b, t, _ = z.shape
        belief = self.initial_distribution().unsqueeze(0).expand(b, -1)
        loglik = torch.zeros(b, device=z.device)
        A = self.transition_matrix()
        for step in range(t):
            pred = belief @ A
            emit = self.emission_prob(z[:, step])
            obs_prob = emit.gather(2, obs[:, step, None, None].expand(-1, self.num_states, 1)).squeeze(-1)
            joint = pred * obs_prob
            scale = joint.sum(-1).clamp_min(1e-12)
            loglik += torch.log(scale)
            belief = joint / scale.unsqueeze(-1)
        return -loglik.mean()

    @torch.no_grad()
    def baum_welch_transition_update(self, z: torch.Tensor, obs: torch.Tensor, concentration: float = 0.5) -> None:
        """One EM-style transition update using forward-backward expected counts.

        z: [T,D], obs:[T]. Neural emission parameters stay fixed in this step.
        """
        device = z.device
        T = z.size(0)
        A = self.transition_matrix()
        pi0 = self.initial_distribution()
        emit_all = self.emission_prob(z)  # [T,K,3]
        B = emit_all[torch.arange(T, device=device), :, obs]  # [T,K]

        alpha = []
        a = pi0 * B[0]
        a = a / a.sum().clamp_min(1e-12)
        alpha.append(a)
        for t in range(1, T):
            a = (alpha[-1] @ A) * B[t]
            a = a / a.sum().clamp_min(1e-12)
            alpha.append(a)
        alpha = torch.stack(alpha)

        beta = torch.ones(T, self.num_states, device=device)
        for t in range(T - 2, -1, -1):
            beta[t] = A @ (B[t + 1] * beta[t + 1])
            beta[t] = beta[t] / beta[t].sum().clamp_min(1e-12)

        counts = torch.full_like(A, concentration)
        for t in range(T - 1):
            xi = alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :]
            xi = xi / xi.sum().clamp_min(1e-12)
            counts += xi
        A_new = counts / counts.sum(-1, keepdim=True)
        self.transition_logits.copy_(torch.log(A_new.clamp_min(1e-8)))
