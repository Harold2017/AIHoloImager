# Copyright (c) 2026 Minmin Gong
#

from pathlib import Path

import torch

from PythonSystem import ComputeDevice, DeviceSync, PurgeTorchCache, TensorToBytes
from LoMa import LoMa, LoMaB, LoMaB128, LoMaL, LoMaG, LoMaR
from LoMa.LoMa import filter_matches
from LoMa.weights import SetModelDir


class Matcher:
    """Matches the features extracted by Extractor, returning putative match index pairs."""

    MODELS = {"b": LoMaB, "b128": LoMaB128, "l": LoMaL, "g": LoMaG, "r": LoMaR}

    def __init__(
        self,
        model_name: str = "b",
        filter_threshold: float = 0.1,
        model_dir: str | Path | None = None,
    ) -> None:
        self.device = ComputeDevice()

        this_py_dir = Path(__file__).parent.resolve()
        SetModelDir(this_py_dir / "Models/LoMa/" if model_dir is None else Path(model_dir))

        self.filter_threshold = filter_threshold

        self.model = LoMa(self.MODELS[model_name]())
        self.model.eval()
        self.model = self.model.to(self.device)

    def Destroy(self) -> None:
        del self.model
        PurgeTorchCache()

    @torch.no_grad()
    def Match(self, features: tuple[int, torch.Tensor], image_idx0: int, image_idx1: int) -> tuple[bytes, int]:
        features0 = features[image_idx0]
        features1 = features[image_idx1]

        scores = self.model(features0["keypoints"], features1["keypoints"], features0["descriptors"], features1["descriptors"])["scores"]
        m0, _, _, _ = filter_matches(scores, self.filter_threshold)

        valid = m0[0] > -1
        pairs = torch.stack([torch.where(valid)[0], m0[0][valid]], dim=-1).to(torch.int32)

        PurgeTorchCache()

        DeviceSync(self.device)
        return (TensorToBytes(pairs.contiguous()), pairs.shape[0])
