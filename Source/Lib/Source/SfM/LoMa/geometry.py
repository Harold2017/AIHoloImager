# Copyright (c) 2026 Minmin Gong
#
# Adapted from https://github.com/davnords/LoMa/blob/main/src/loma/geometry.py
# Only the parts required by the detector are kept, so that OpenCV and einops are not needed.

import torch


def kde(x: torch.Tensor, std: float = 0.1, half: bool = True) -> torch.Tensor:
    if half:
        x = x.half()
    scores = (-(torch.cdist(x, x) ** 2) / (2 * std**2)).exp()
    return scores.sum(dim=-1)


def get_normalized_grid(B: int, H: int, W: int, overload_device: torch.device | None = None) -> torch.Tensor:
    if B == 0:
        return torch.zeros(0, H, W, 2, device=overload_device)
    x1_n = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=overload_device) for n in (B, H, W)],
        indexing="ij",
    )
    return torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
