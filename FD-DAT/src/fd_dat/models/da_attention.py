"""A compact, detector-agnostic version of the refined DA attention block.

This module captures the information flow in Eqs. (19)-(23). The actual
Deformable DETR integration should replace ``nn.MultiheadAttention`` in the
cross-attention path with the upstream multi-scale deformable-attention op.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * gradient, None


class GradientReversal(nn.Module):
    def __init__(self, strength: float = 1.0) -> None:
        super().__init__()
        self.strength = strength

    def forward(self, inputs: Tensor) -> Tensor:
        return _GradientReversalFunction.apply(inputs, self.strength)


@dataclass
class DomainAttentionOutput:
    tokens: Tensor
    di_tokens: Tensor
    ds_tokens: Tensor
    domain_logits: Tensor


class RefinedDomainAttention(nn.Module):
    """DI/DS-aware cross-attention with GRL-based domain confusion."""

    def __init__(
        self,
        hidden_dim: int = 256,
        heads: int = 8,
        dropout: float = 0.1,
        grl_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.grl = GradientReversal(grl_strength)
        self.di_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.ds_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.ds_value_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.model_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.di_norm = nn.LayerNorm(hidden_dim)
        self.ds_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query_di: Tensor,
        query_ds: Tensor,
        memory_di: Tensor,
        memory_ds: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
    ) -> DomainAttentionOutput:
        if not (
            query_di.shape == query_ds.shape
            and memory_di.shape == memory_ds.shape
            and query_di.shape[-1] == memory_di.shape[-1]
        ):
            raise ValueError("DI and DS tensors must have matching hidden dimensions")

        di_residual, _ = self.di_attention(
            query_di,
            memory_di,
            memory_di,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        di_tokens = self.di_norm(query_di + di_residual)

        # Eq. (20): DS values are formed from GRL(DI) and DS memory.
        ds_values = self.ds_value_projection(
            torch.cat((self.grl(memory_di), memory_ds), dim=-1)
        )
        # Eq. (21): a learned gate distills domain-specific information while
        # adversarial feedback removes easy domain cues from the DI branch.
        gate = self.model_gate(torch.cat((self.grl(query_di), query_ds), dim=-1))
        gated_ds_query = gate * query_ds + (1.0 - gate) * self.grl(query_di)
        ds_residual, _ = self.ds_attention(
            gated_ds_query,
            ds_values,
            ds_values,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        ds_tokens = self.ds_norm(query_ds + ds_residual)

        tokens = self.output_projection(torch.cat((di_tokens, ds_tokens), dim=-1))
        domain_logits = self.domain_discriminator(self.grl(di_tokens.mean(dim=1))).squeeze(-1)
        return DomainAttentionOutput(tokens, di_tokens, ds_tokens, domain_logits)

