"""Predictor: maps the last 3 frame embeddings, the pusher position and the pusher action to
the next frame embedding. Training uses an L1 one-step loss plus an L1 2-step rollout loss.

The tokens are the patches of each context frame. Attention is frame-causal: a token sees
every patch of its own frame and of earlier frames, so the output at each frame predicts the
frame after it. Each patch token also gets a small MLP of its own place on the patch grid,
the frame's pusher position and its pusher action (how far the pusher moves over the next
step), so whether the pusher is about to touch a patch can be worked out locally. The
action-blind predictor is the same model without the pusher action.

The predictor outputs the change from the frame it is at, and its output layer starts at
zero, so an untrained predictor is the static baseline. Embeddings, positions and actions
are standardised with statistics of the training episodes, stored with the weights.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pwm.episodes import Episode
from pwm.windows import CONTEXT, make_windows


@dataclass(frozen=True)
class Size:
    width: int
    layers: int
    heads: int
    lr: float  # peak learning rate


SIZES = {
    "tiny": Size(width=32, layers=2, heads=2, lr=3e-3),  # toy runs: every step costs ~0.1 s on a laptop CPU
    "real": Size(width=512, layers=6, heads=8, lr=3e-4),  # about 20M parameters on 16 x 16 x 1024 grids
}


class Predictor(nn.Module):
    def __init__(self, size: str, patches: int, dims: int, use_actions: bool, context: int = CONTEXT):
        super().__init__()
        self.config = dict(size=size, patches=patches, dims=dims, use_actions=use_actions, context=context)
        self.use_actions, self.context = use_actions, context
        s = SIZES[size]
        self.embed_in = nn.Linear(dims, s.width)
        self.patch_pos = nn.Parameter(0.02 * torch.randn(patches, s.width))
        self.frame_pos = nn.Parameter(0.02 * torch.randn(context, 1, s.width))
        self.condition = nn.Sequential(
            nn.Linear(2 + (4 if use_actions else 2), s.width), nn.GELU(), nn.Linear(s.width, s.width)
        )
        layer = nn.TransformerEncoderLayer(
            s.width, s.heads, 4 * s.width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, s.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(s.width)
        self.embed_out = nn.Linear(s.width, dims)
        nn.init.zeros_(self.embed_out.weight)
        nn.init.zeros_(self.embed_out.bias)

        frame = torch.arange(context).repeat_interleave(patches)
        self.register_buffer("blocked", frame[:, None] < frame[None, :], persistent=False)  # no peeking ahead
        self.register_buffer("patch_xy", _grid_coordinates(patches), persistent=False)
        self.register_buffer("emb_mean", torch.zeros(dims))
        self.register_buffer("emb_std", torch.ones(dims))
        self.register_buffer("pos_mean", torch.zeros(2))
        self.register_buffer("pos_std", torch.ones(()))
        self.register_buffer("action_std", torch.ones(()))

    def forward(self, z: torch.Tensor, pusher_pos: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Standardised embeddings z (B, C, P, D), pusher positions (B, C, 2) and the action taken
        at each of those frames (B, C, 2) -> the standardised next embedding after each frame."""
        b, c, p, d = z.shape
        pusher = (pusher_pos - self.pos_mean) / self.pos_std
        if self.use_actions:
            pusher = torch.cat([pusher, actions / self.action_std], dim=-1)
        per_patch = torch.cat([self.patch_xy.expand(b, c, p, 2), pusher[:, :, None].expand(b, c, p, -1)], dim=-1)
        h = self.embed_in(z) + self.patch_pos + self.frame_pos[:c] + self.condition(per_patch)
        h = self.blocks(h.reshape(b, c * p, -1), mask=self.blocked[: c * p, : c * p])
        return z + self.embed_out(self.norm(h)).reshape(b, c, p, d)

    def loss(self, z: torch.Tensor, pusher_pos: torch.Tensor) -> torch.Tensor:
        """One-step loss at every context frame plus the loss after a 2-step rollout.

        z (B, C + 2, P, D) standardised embeddings and pusher_pos (B, C + 2, 2) of a window.
        """
        c = self.context
        actions = pusher_pos.diff(dim=1)
        one = self(z[:, :c], pusher_pos[:, :c], actions[:, :c])
        context = torch.cat([z[:, 1:c], one[:, -1:]], dim=1)  # feed the prediction back
        two = self(context, pusher_pos[:, 1 : c + 1], actions[:, 1 : c + 1])[:, -1]
        return _l1(one, z[:, 1 : c + 1]) + _l1(two, z[:, c + 1])

    @torch.inference_mode()
    def rollout(self, context: np.ndarray, pusher_pos: np.ndarray, actions: np.ndarray, chunk: int = 32) -> np.ndarray:
        """The evaluator's model interface: context (B, C, P, D) real embeddings, pusher_pos
        (B, C, 2) at the context frames and future actions (B, H, 2) -> predictions (B, H, P, D).

        Each prediction is fed back as context. Future pusher positions are integrated from
        the given actions, so they stay consistent with them even when the actions are wrong.
        """
        device = self.emb_mean.device
        out = []
        for i in range(0, len(context), chunk):
            z = (torch.as_tensor(context[i : i + chunk], device=device).float() - self.emb_mean) / self.emb_std
            pos = torch.as_tensor(pusher_pos[i : i + chunk], device=device).float()
            future = torch.as_tensor(actions[i : i + chunk], device=device).float()
            frame_actions = torch.cat([pos.diff(dim=1), future], dim=1)  # the action taken at each frame
            predictions = []
            for k in range(future.shape[1]):
                nxt = self(z, pos, frame_actions[:, k : k + self.context])[:, -1:]
                predictions.append(nxt)
                z = torch.cat([z[:, 1:], nxt], dim=1)
                pos = torch.cat([pos[:, 1:], pos[:, -1:] + future[:, k : k + 1]], dim=1)
            out.append((torch.cat(predictions, dim=1) * self.emb_std + self.emb_mean).cpu().numpy())
        return np.concatenate(out)

    def save(self, path: Path) -> None:
        torch.save({"config": self.config, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "Predictor":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state"])
        return model.to(device).eval()


def train(
    episodes: list[Episode],
    embeddings: list[np.ndarray],
    step: float,
    use_actions: bool = True,
    size: str = "real",
    steps: int = 2000,
    batch_size: int = 16,
    lr: float | None = None,
    device: str = "cpu",
    precision: str = "fp32",
    seed: int = 0,
    log_every: int = 0,
) -> Predictor:
    """Train a predictor on the training episodes; `embeddings[i]` belongs to `episodes[i]`.

    `lr` defaults to the size's. `precision` is "fp32", or "fp16" (CUDA only): fp16
    autocast with a gradient scaler; Turing GPUs (T4, RTX 4000) have no bf16. Batches are
    gathered from `embeddings` at each step, so memory-mapped embeddings are never loaded whole.
    """
    assert precision in ("fp32", "fp16"), precision
    fp16 = precision == "fp16"
    assert not fp16 or device.startswith("cuda"), "fp16 training needs a CUDA device"
    train_ids = [i for i, ep in enumerate(episodes) if ep.split == "train"]
    assert train_ids, "no training episodes"
    train_episodes, train_embeddings = [episodes[i] for i in train_ids], [embeddings[i] for i in train_ids]
    windows = make_windows(train_episodes, step, future=2)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Predictor(size, *train_embeddings[0].shape[1:], use_actions=use_actions)
    _set_statistics(model, train_embeddings, windows)
    model.to(device).train()

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr or SIZES[size].lr, weight_decay=0.01)
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda i: min(1, (i + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    for i in range(steps):
        rows = np.sort(rng.choice(len(windows), batch_size, replace=len(windows) < batch_size))
        z = torch.as_tensor(windows.gather(train_embeddings, rows), device=device).float()
        z = (z - model.emb_mean) / model.emb_std
        pos = torch.as_tensor(windows.pusher_pos[rows], device=device).float()
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            loss = model.loss(z, pos)
        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimiser)
        scaler.update()
        schedule.step()
        if log_every and (i % log_every == 0 or i == steps - 1):
            print(f"step {i:5d}  loss {loss.item():.4f}", flush=True)
    return model.eval()


def _l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss with the kink at zero smoothed: quadratic for errors below 0.1 (in standardised
    units), L1 above. Plain L1 stalls where most entries never change (the toy embeddings):
    its constant pull on those entries keeps the predictor at the static baseline."""
    return nn.functional.smooth_l1_loss(prediction.float(), target.float(), beta=0.1)


def _grid_coordinates(patches: int) -> torch.Tensor:
    """(x, y) of each patch centre on the square patch grid (row-major), scaled to -1..1."""
    side = math.isqrt(patches)
    assert side * side == patches, f"{patches} patches do not form a square grid"
    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    return (torch.stack([x.reshape(-1), y.reshape(-1)], dim=1).float() + 0.5) / side * 2 - 1


def _set_statistics(model: Predictor, embeddings: list[np.ndarray], windows, n_frames: int = 256) -> None:
    """Per-channel mean and std of the embeddings (from up to `n_frames` evenly spread
    frames), and the scale of pusher positions and actions in the training windows."""
    frames = [(e, f) for e, emb in enumerate(embeddings) for f in range(len(emb))]
    picked = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)
    sample = np.stack([embeddings[frames[i][0]][frames[i][1]] for i in picked]).astype(np.float64)
    std = sample.std(axis=(0, 1))
    std = np.maximum(std, 1e-3 * std.mean() + 1e-12)  # channels that never vary stay at zero
    pos, actions = windows.pusher_pos.reshape(-1, 2), windows.actions.reshape(-1, 2)
    with torch.no_grad():
        model.emb_mean.copy_(torch.as_tensor(sample.mean(axis=(0, 1))))
        model.emb_std.copy_(torch.as_tensor(std))
        model.pos_mean.copy_(torch.as_tensor(pos.mean(axis=0)))
        model.pos_std.copy_(torch.as_tensor(max(pos.std(), 1e-6)))
        model.action_std.copy_(torch.as_tensor(max(np.sqrt((actions**2).mean()), 1e-6)))
