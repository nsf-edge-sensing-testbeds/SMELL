from __future__ import annotations

import torch
from torch import nn


class BidirectionalCrossAttention(nn.Module):
    """Region->token and token->region attention followed by pooled fusion."""

    def __init__(self, dim: int = 512, heads: int = 8, fusion_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.text_queries_visual = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.visual_queries_text = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Text queries visual regions: z_{v -> t}
        tv, _ = self.text_queries_visual(text_tokens, visual_tokens, visual_tokens)
        tv = self.norm_t(text_tokens + tv)

        # Visual queries text: z_{t -> v}
        vt, _ = self.visual_queries_text(
            visual_tokens,
            text_tokens,
            text_tokens,
            key_padding_mask=~text_mask.bool(),
        )
        vt = self.norm_v(visual_tokens + vt)

        # Masked mean text pooling and mean visual pooling.
        weights = text_mask.float().unsqueeze(-1)
        text_pool = (tv * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        visual_pool = vt.mean(1)
        return self.fuse(torch.cat([text_pool, visual_pool], dim=-1))
