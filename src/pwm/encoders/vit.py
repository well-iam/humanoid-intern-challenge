"""The real encoders: frozen ViTs from Hugging Face, each giving a 16 x 16 grid of patch embeddings.

- DINOv2-small at 224 px, patch 14: (T, 256, 384). The class token is dropped.
- V-JEPA 2 ViT-L at 256 px, patch 16, tubelet 2: (T, 256, 1024). Each frame is duplicated
  into its own two-frame tubelet, the model's predictor is skipped, and the output is
  layer-normed over the channels, as in V-JEPA 2-AC.

Both run in fp32 with SDPA attention (Turing GPUs such as the T4 have no bf16). The whole
frame is resized to the model's input size: no centre crop, because the frames are already
the square of table the model should see.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _ViTEncoder:
    name: str
    checkpoint: str
    size: int  # input height and width in pixels
    resample: str  # the resampling of the model's own processor
    batch: int  # frames per forward pass

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(self.checkpoint, dtype=torch.float32, attn_implementation="sdpa")
        self.model.eval().to(self.device)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Frames (T, H, W, 3) uint8, square -> frame embeddings (T, 256, dims) float32."""
        out = []
        for i in range(0, len(frames), self.batch):
            x = torch.from_numpy(frames[i : i + self.batch]).to(self.device).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=(self.size, self.size), mode=self.resample, antialias=True).clamp(0, 1)
            out.append(self._patches((x - self.mean) / self.std).float().cpu().numpy())
        return np.concatenate(out)

    def _patches(self, pixels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DinoV2Small(_ViTEncoder):
    name = "dinov2"
    checkpoint = "facebook/dinov2-small"
    size = 224
    resample = "bicubic"
    batch = 128

    def _patches(self, pixels):
        return self.model(pixel_values=pixels).last_hidden_state[:, 1:]  # drop the class token


class VJepa2Large(_ViTEncoder):
    name = "vjepa2"
    checkpoint = "facebook/vjepa2-vitl-fpc64-256"
    size = 256
    resample = "bilinear"
    batch = 32

    def _patches(self, pixels):
        video = pixels.unsqueeze(1).expand(-1, 2, -1, -1, -1)  # (B, 2, 3, H, W): one tubelet per frame
        tokens = self.model(pixel_values_videos=video, skip_predictor=True).last_hidden_state
        return F.layer_norm(tokens, tokens.shape[-1:])
