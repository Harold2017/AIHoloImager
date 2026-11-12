# Copyright (c) 2026 Minmin Gong
#

from pathlib import Path

import torch
import torch.nn.functional as F

from PythonSystem import ComputeDevice, DeviceSync, PurgeTorchCache, TensorToBytes
from LoMa import LoMa, LoMaB, LoMaB128, LoMaL, LoMaG, LoMaR
from LoMa.weights import SetModelDir


class Extractor:
    """Extracts LoMa keypoints and descriptors for a single image.

    LoMa works on RGB images, and its descriptor relies on a global attention window, so the
    image is downscaled to MaxImageSize on its longer side before the inference.
    """

    MODELS = {"b": LoMaB, "b128": LoMaB128, "l": LoMaL, "g": LoMaG, "r": LoMaR}

    def __init__(
        self,
        model_name: str = "b",
        max_image_size: int = 1024,
        num_keypoints: int = 2048,
        model_dir: str | Path | None = None,
    ) -> None:
        self.device = ComputeDevice()

        this_py_dir = Path(__file__).parent.resolve()
        SetModelDir(this_py_dir / "Models/LoMa/" if model_dir is None else Path(model_dir))

        self.max_image_size = max_image_size
        self.num_keypoints = num_keypoints

        self.model = LoMa(self.MODELS[model_name]())
        self.model.eval()
        self.model = self.model.to(self.device)

        # DINOv2 based descriptors need the resolution to be a multiple of the patch size.
        self.align = 14 if self.model.cfg.descriptor == "dedode_g" else 8

    def Destroy(self) -> None:
        del self.model
        PurgeTorchCache()

    @torch.no_grad()
    def Extract(self, image: torch.Tensor) -> dict:
        if image.dim() == 4:
            image = image[0]

        if image.shape[-1] in (3, 4):
            image = image[..., :3].permute(2, 0, 1)

        h, w = image.shape[-2:]

        image = image.to(torch.float32) / 255
        image, padded_size, process_size = self.Resize(image)

        keypoints, descriptors, _, _ = self.model.detect_and_describe(image, self.num_keypoints)

        DeviceSync(self.device)

        return {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "image_size": torch.tensor([w, h], device=keypoints.device, dtype=torch.float32),
            "process_size": torch.tensor(process_size, device=keypoints.device, dtype=torch.float32),
            "padded_size": torch.tensor(padded_size, device=keypoints.device, dtype=torch.float32),
        }

    def Resize(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        """Returns the batched padded image, then the padded and the unpadded sizes, both as (width, height)."""
        if image.dim() == 3:
            image = image[None]

        h, w = image.shape[-2:]

        scale = min(1.0, self.max_image_size / max(h, w))
        process_h = max(self.align, int(round(h * scale / self.align)) * self.align)
        process_w = max(self.align, int(round(w * scale / self.align)) * self.align)

        if (process_h != h) or (process_w != w):
            image = F.interpolate(image, size=(process_h, process_w), mode="bilinear", align_corners=False, antialias=True)

        pad_h = (self.align - process_h % self.align) % self.align
        pad_w = (self.align - process_w % self.align) % self.align
        if (pad_h != 0) or (pad_w != 0):
            image = F.pad(image, (0, pad_w, 0, pad_h))

        return image, (image.shape[-1], image.shape[-2]), (process_w, process_h)

    def ExportFeatures(self, features: dict) -> tuple[bytes, bytes, bytes]:
        keypoints = features["keypoints"][0]
        padded_w, padded_h = features["padded_size"].tolist()
        process_w, process_h = features["process_size"].tolist()
        width, height = features["image_size"].tolist()

        pixel_keypoints = (keypoints + 1) / 2 * torch.tensor([padded_w, padded_h], device=keypoints.device, dtype=keypoints.dtype)
        pixel_keypoints = pixel_keypoints * torch.tensor(
            [width / process_w, height / process_h], device=keypoints.device, dtype=keypoints.dtype
        )

        descriptors = F.normalize(features["descriptors"], p=2, dim=-1)

        return (
            TensorToBytes(descriptors),
            TensorToBytes(pixel_keypoints.contiguous()),
            TensorToBytes(features["image_size"]),
        )
