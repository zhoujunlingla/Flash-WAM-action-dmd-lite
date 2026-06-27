"""Action-only DMD-lite helpers for Flash-WAM distillation.

This module intentionally keeps the first distribution-matching extension small:
only the action endpoint distribution is modeled, while video stays under the
original Flash-WAM consistency objective.  The fake score predicts action x0
from a noisy student action endpoint and cheap pooled video-cache statistics.
"""

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeActionScore(nn.Module):
    """Small action fake-score network.

    FastGen/DMD2 usually uses a full fake score model. For WAM action-first
    refinement we start with a lightweight action-only estimator because action
    latents are low dimensional and directly tied to RoboTwin success.

    The fake score is applied element-wise with shared weights instead of
    flattening the whole action chunk. RoboTwin batches can expose different
    temporal lengths, so this keeps the DMD-lite path shape-stable while still
    producing a distribution gradient for every action endpoint element.
    """

    def __init__(self, action_size, hidden_dim=256, depth=3):
        super().__init__()
        del action_size
        hidden_dim = min(int(hidden_dim), 256)
        in_dim = 5  # scalar action + sigma + video mean/std/absmean
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, action_sigma, sigma, video_stats):
        if action_sigma.ndim != 5:
            raise ValueError(f"Expected action_sigma [B,C,F,N,1], got {tuple(action_sigma.shape)}")
        bsz, channels, frames, tokens, one = action_sigma.shape

        sigma = sigma.to(device=action_sigma.device).float()
        if sigma.ndim == 0:
            sigma = sigma.reshape(1, 1).expand(bsz, frames)
        elif sigma.ndim == 1:
            if sigma.numel() == frames:
                sigma = sigma[None].expand(bsz, frames)
            else:
                sigma = sigma.reshape(bsz, 1).expand(bsz, frames)
        elif sigma.shape != (bsz, frames):
            sigma = sigma.reshape(bsz, -1).mean(dim=1, keepdim=True).expand(bsz, frames)

        action_flat = action_sigma.float().permute(0, 2, 1, 3, 4).reshape(-1, 1)
        sigma_flat = sigma[:, :, None, None, None].expand(
            bsz, frames, channels, tokens, one).reshape(-1, 1)
        stats_flat = video_stats.float()[:, None, None, None, :].expand(
            bsz, frames, channels, tokens, 3).reshape(-1, 3)
        cond = torch.cat([action_flat, sigma_flat, stats_flat], dim=1)
        pred = self.net(cond).reshape(bsz, frames, channels, tokens, one)
        return pred.permute(0, 2, 1, 3, 4).contiguous().to(action_sigma.dtype)


class ActionDMDReplayBuffer:
    """Recent on-policy action endpoint replay for fake-score stabilization."""

    def __init__(self, max_size=128):
        self.max_size = int(max_size)
        self.items = deque(maxlen=self.max_size)

    def __len__(self):
        return len(self.items)

    @torch.no_grad()
    def add(self, action_x0, video_stats, mask):
        if self.max_size <= 0:
            return
        for i in range(action_x0.shape[0]):
            self.items.append((
                action_x0[i].detach().float().cpu(),
                video_stats[i].detach().float().cpu(),
                mask[i].detach().float().cpu(),
            ))

    def sample(self, batch_size, device, dtype):
        if len(self.items) == 0:
            return None
        idx = torch.randint(0, len(self.items), (batch_size,))
        actions, stats, masks = zip(*(self.items[int(i)] for i in idx))
        actions = torch.stack(actions).to(device=device, dtype=dtype)
        stats = torch.stack(stats).to(device=device, dtype=torch.float32)
        masks = torch.stack(masks).to(device=device, dtype=dtype)
        return actions, stats, masks


def pooled_video_stats(video_latents):
    """Return cheap conditioning stats for the fake action score."""
    x = video_latents.detach().float().flatten(1)
    return torch.stack(
        [x.mean(dim=1), x.std(dim=1), x.abs().mean(dim=1)],
        dim=1,
    )


def masked_huber(pred, target, mask, huber_c):
    diff = (pred.float() - target.float()) * mask.float()
    return (torch.sqrt(diff ** 2 + huber_c ** 2) - huber_c).sum() / mask.float().sum().clamp(min=1)


def masked_mse(pred, target, mask):
    diff = (pred.float() - target.float()) * mask.float()
    return (diff ** 2).sum() / mask.float().sum().clamp(min=1)


def normalize_dmd_gradient(grad, mask, sigma, min_scale=0.01):
    """Scheduler-aware, low-sigma-safe normalization for action DMD gradients."""
    sigma_5d = sigma[:, None, :, None, None].to(grad.device, grad.dtype)
    scale = torch.clamp(sigma_5d ** 2, min=min_scale, max=1.0)
    grad = grad * scale * mask.to(grad.dtype)
    denom = grad.abs().sum(dim=(1, 2, 3, 4), keepdim=True) / \
        mask.to(grad.dtype).sum(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1)
    return grad / denom.clamp(min=1e-4)
