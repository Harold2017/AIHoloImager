# Copyright (c) 2026 Minmin Gong
#

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_model_dir: Path | None = None


def SetModelDir(path: str | Path) -> None:
    global _model_dir
    _model_dir = Path(path)


def LoadStateDict(url: str | Path, *, map_location=None, allow_download: bool = True):
    """Load a pretrained state dict, preferring a local copy over a network download.

    The local copy is looked up by the file name of url inside the directory given to
    SetModelDir, so the deployment only needs to ship the weights next to the models.
    """
    path = Path(url)
    if path.is_file():
        return torch.load(path, map_location=map_location, weights_only=True)

    if _model_dir is not None:
        cached_path = _model_dir / path.name
        if cached_path.is_file():
            return torch.load(cached_path, map_location=map_location, weights_only=True)
        if not allow_download:
            logger.info("Skipped loading %s: no local copy available.", path.name)
            return None

    logger.info("Downloading %s ...", path.name)
    return torch.hub.load_state_dict_from_url(str(url), map_location=map_location)
