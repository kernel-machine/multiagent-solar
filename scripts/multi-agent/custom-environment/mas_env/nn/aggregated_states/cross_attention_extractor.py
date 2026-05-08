"""
CrossAttentionExtractor
=======================
Custom SB3 BaseFeaturesExtractor that replaces the hand-crafted
(min, avg, max) aggregation of other agents' states with a learned
single-head cross-attention mechanism.

Observation layout expected from CustomEnvironment (use_cross_attention=True):
  [0:3]                      own         : (battery_i, backlog_i, timestep)
  [3 : 3 + 2*(max_agents-1)] others_flat : (batt_j, back_j) for j in 0..max_agents-2
  [3 + 2*(max_agents-1) : ]  mask        : 1.0 = valid agent slot, 0.0 = padding

Output fed to PPO's MLP:
  [own (3) | others_mean (2) | attn_delta (d_model)]  →  dim = 3 + 2 + d_model

Design: residual mean baseline
  The simple mean of others' (battery, backlog) is concatenated directly —
  this provides a useful signal from step 0 (no cold-start problem).
  The attention computes a DELTA on top of it: attn_delta = attn_out - others_mean_proj.
  This means early in training the extractor behaves like a mean aggregation,
  and gradually learns to weight agents selectively.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class CrossAttentionExtractor(BaseFeaturesExtractor):
    """
    Single-head cross-attention feature extractor with mean-baseline residual.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        Flat observation space of size 3 + 3*(max_agents-1).
    max_agents : int
        Maximum number of agents (including self). Slots beyond the actual
        number of agents are zero-padded and masked out.
    d_model : int
        Dimension of the attention embedding (keys, values, output).
        Features returned have dimension 3 + 2 + d_model.
    """

    def __init__(self, observation_space: gym.spaces.Box, max_agents: int = 10, d_model: int = 16, num_heads: int = 1):
        n_others = max_agents - 1
        # own (3) + others_mean (2) + attn_delta (d_model)
        features_dim = 3 + 2 + d_model
        super().__init__(observation_space, features_dim=features_dim)

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.max_agents = max_agents
        self.n_others   = n_others
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads
        self.scale      = self.d_k ** -0.5

        # Query: project own state (3 dims) → d_model
        self.q_proj = nn.Linear(3, d_model)
        # Key / Value: project each other agent's (battery, backlog) → d_model
        self.k_proj = nn.Linear(2, d_model)
        self.v_proj = nn.Linear(2, d_model)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Project mean(others) to d_model for residual subtraction
        self.mean_proj = nn.Linear(2, d_model)

        # Initialize mean_proj to identity-like: attn_delta ≈ 0 early in training
        # so the extractor starts as a pure mean aggregator.
        nn.init.zeros_(self.mean_proj.weight)
        nn.init.zeros_(self.mean_proj.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        observations : (B, 3 + 3*(max_agents-1))

        Returns
        -------
        (B, 3 + 2 + d_model)
        """
        B = observations.shape[0]
        n = self.n_others

        # ── Split raw observation ────────────────────────────────────────────
        own         = observations[:, :3]                   # (B, 3)
        others_flat = observations[:, 3 : 3 + 2 * n]       # (B, 2*n)
        mask        = observations[:, 3 + 2 * n :]          # (B, n)  1=valid, 0=pad

        others = others_flat.view(B, n, 2)                  # (B, n, 2)

        # ── Masked mean of others (valid slots only) ─────────────────────────
        mask_f = mask.unsqueeze(-1)                         # (B, n, 1)
        valid_count = mask_f.sum(dim=1).clamp(min=1.0)     # (B, 1)
        others_sum  = (others * mask_f).sum(dim=1)          # (B, 2)
        others_mean = others_sum / valid_count              # (B, 2)  ← baseline

        # ── Projections ──────────────────────────────────────────────────────
        Q = self.q_proj(own)                # (B, d_model)
        K = self.k_proj(others)             # (B, n, d_model)
        V = self.v_proj(others)             # (B, n, d_model)

        # Reshape for multi-head: (B, h, seq_len, d_k)
        Q = Q.view(B, 1, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, 1, d_k)
        K = K.view(B, n, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, n, d_k)
        V = V.view(B, n, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, n, d_k)

        # ── Scaled dot-product attention ─────────────────────────────────────
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, h, 1, n)

        # Mask padded slots: set their logit to -inf so softmax → 0
        pad_mask = (mask < 0.5).view(B, 1, 1, n)              # (B, 1, 1, n)
        attn_scores = attn_scores.masked_fill(pad_mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)      # (B, h, 1, n)

        # Guard: if ALL slots are padded, softmax → NaN; replace with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_out = (attn_weights @ V)                          # (B, h, 1, d_k)
        
        # Concat heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, self.d_model) # (B, d_model)
        attn_out = self.out_proj(attn_out)                     # (B, d_model)

        # ── Residual: learn delta from the mean baseline ─────────────────────
        # attn_delta = attn_out - mean_proj(others_mean)
        # At init, mean_proj weights = 0  →  attn_delta = attn_out  ≈ small noise
        # but the mean is still passed directly below → cold-start is mitigated.
        attn_delta = attn_out - self.mean_proj(others_mean)    # (B, d)

        # ── Concatenate: own | others_mean | attn_delta ──────────────────────
        return torch.cat([own, others_mean, attn_delta], dim=-1)   # (B, 3+2+d)
