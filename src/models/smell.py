from __future__ import annotations

import torch
from torch import nn

from .visual_encoder import ResNeXtSEEncoder
from .text_encoder import TextTransformerEncoder
from .cross_modal import BidirectionalCrossAttention


class SMELLModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 5,
        text_dim: int = 512,
        fusion_dim: int = 256,
        text_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        visual_pretrained: bool = False,
    ):
        super().__init__()
        self.visual = ResNeXtSEEncoder(text_dim, pretrained=visual_pretrained)
        self.text = TextTransformerEncoder(
            vocab_size=vocab_size,
            dim=text_dim,
            layers=text_layers,
            heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.cross_modal = BidirectionalCrossAttention(text_dim, num_heads, fusion_dim, dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)
        # delta(z) = ||W_d z||_2, implemented as scalar-vector projection norm.
        self.difficulty_proj = nn.Linear(fusion_dim, 16)

    def encode(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        v = self.visual(image)
        t = self.text(input_ids, attention_mask)
        return self.cross_modal(v, t, attention_mask)

    def difficulty(self, z: torch.Tensor) -> torch.Tensor:
        raw = torch.linalg.vector_norm(self.difficulty_proj(z), ord=2, dim=-1)
        # normalize to a convenient [0,1] range within a batch
        if raw.numel() > 1:
            lo, hi = raw.min().detach(), raw.max().detach()
            raw = (raw - lo) / (hi - lo + 1e-8)
        return raw

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(image, input_ids, attention_mask)
        return {
            "embedding": z,
            "logits": self.classifier(z),
            "difficulty": self.difficulty(z),
        }
