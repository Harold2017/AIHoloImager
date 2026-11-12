# Copyright (c) 2026 Minmin Gong
#
# Script for evaluating LoMa against the current SuperPoint + LightGlue SfM front end.
# It bypasses the C++ pipeline on purpose: the point is to measure the feature front end alone.
#
# Usage:
#   <venv-python> Source/Lib/Source/SfM/LoMa/eval/eval.py -i <image-dir> [--compare] [--model b|b128|l|g|r]
#
# It does not need the AIHoloImager build, only the Python venv.

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EVAL_DIR = Path(__file__).resolve().parent
SFM_DIR = EVAL_DIR.parent.parent
sys.path.insert(0, str(SFM_DIR.parent / "Python"))
sys.path.insert(0, str(SFM_DIR))

REPO_DIR = next((parent for parent in SFM_DIR.parents if (parent / "External" / "LoMa").is_dir()), None)
if REPO_DIR is None:
    raise RuntimeError("Failed to locate the repository root, please run this from a source checkout.")

DEFAULT_MODEL_DIR = REPO_DIR / "External" / "LoMa" / "LoMaModels"
DEFAULT_LIGHTGLUE_MODEL_DIR = REPO_DIR / "External" / "LightGlue" / "LightGlueModels"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def Sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ResetPeakVram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def PeakVram() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def FindImages(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def LoadImage(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert("RGB"))).to(torch.uint8)


def Report(title, extract_times, match_times, num_keypoints, pair_matches):
    print(f"\n=== {title} ===")
    if extract_times:
        print(
            f"  extract  : {statistics.mean(extract_times):8.1f} ms/image "
            f"(min {min(extract_times):.1f}, max {max(extract_times):.1f}, n={len(extract_times)})"
        )
        print(f"  keypoints: {statistics.mean(num_keypoints):8.1f} /image")
    if match_times:
        print(
            f"  match    : {statistics.mean(match_times):8.1f} ms/pair "
            f"(min {min(match_times):.1f}, max {max(match_times):.1f}, n={len(match_times)})"
        )
        print(f"  matches  : {statistics.mean(pair_matches):8.1f} /pair (min {min(pair_matches)}, max {max(pair_matches)})")
    print(f"  peak VRAM: {PeakVram():8.1f} MB")


def RunLoMa(args, images):
    from LoMaExtractor import Extractor
    from LoMaMatcher import Matcher

    extractor = Extractor(args.model, args.max_image_size, args.num_keypoints, args.model_dir)
    matcher = None
    try:
        warmup_image = LoadImage(images[0])
        for _ in range(args.warmup):
            extractor.Extract(warmup_image)

        features = []
        extract_times = []
        num_keypoints = []
        for path in images:
            start = time.perf_counter()
            features.append(extractor.Extract(LoadImage(path)))
            Sync()
            extract_times.append((time.perf_counter() - start) * 1000)
            num_keypoints.append(features[-1]["keypoints"].shape[1])

        matcher = Matcher(args.model, args.filter_threshold, args.model_dir)
        for _ in range(args.warmup):
            matcher.Match(tuple(features), 0, 1)

        match_times = []
        pair_matches = []
        for i in range(len(features) - 1):
            for j in range(i + 1, len(features)):
                start = time.perf_counter()
                _, count = matcher.Match(tuple(features), i, j)
                Sync()
                match_times.append((time.perf_counter() - start) * 1000)
                pair_matches.append(count)

        Report(f"LoMa-{args.model}", extract_times, match_times, num_keypoints, pair_matches)
        return [(i, j, c) for (i, j), c in zip(((i, j) for i in range(len(features) - 1) for j in range(i + 1, len(features))), pair_matches)]
    finally:
        if matcher is not None:
            matcher.Destroy()
        extractor.Destroy()


def RunSuperPointLightGlue(args, images):
    # Built from the LightGlue package directly (instead of SfM/Extractor.py + SfM/Matcher.py) so that
    # the script can point at the dev-time weight location. The model and its settings are identical.
    from pathlib import Path

    from LightGlue import LightGlue, SuperPoint
    from PythonSystem import ComputeDevice

    model_dir = Path(args.lightglue_model_dir)
    sp = SuperPoint.FromPretrained(model_dir, max_num_keypoints = 4096).eval().to(ComputeDevice())
    lg = LightGlue.FromPretrained(model_dir, features = "superpoint").eval().to(ComputeDevice())

    def Extract(path):
        gray = LoadImage(path).to(torch.float32).mean(dim=-1, keepdim=True)[None]
        return sp.Extract(gray.squeeze(0).permute(2, 0, 1).to(device = ComputeDevice(), dtype = torch.float32) / 255)

    def Match(features0, features1):
        matches = lg({"image0": features0, "image1": features1})
        return matches["matches"][0][..., 0:2].shape[-2]

    try:
        for _ in range(args.warmup):
            Extract(images[0])

        features = []
        extract_times = []
        num_keypoints = []
        for path in images:
            start = time.perf_counter()
            features.append(Extract(path))
            Sync()
            extract_times.append((time.perf_counter() - start) * 1000)
            num_keypoints.append(features[-1]["keypoints"].shape[1])

        for _ in range(args.warmup):
            Match(features[0], features[1])

        match_times = []
        pair_matches = []
        for i in range(len(features) - 1):
            for j in range(i + 1, len(features)):
                start = time.perf_counter()
                count = Match(features[i], features[j])
                Sync()
                match_times.append((time.perf_counter() - start) * 1000)
                pair_matches.append(count)

        Report("SuperPoint + LightGlue (current)", extract_times, match_times, num_keypoints, pair_matches)
        return [(i, j, c) for (i, j), c in zip(((i, j) for i in range(len(features) - 1) for j in range(i + 1, len(features))), pair_matches)]
    finally:
        del lg, sp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    parser = argparse.ArgumentParser(description = "Evaluate the LoMa SfM front end.")
    parser.add_argument("-i", "--input-path", required = True, help = "An image directory or a single image.")
    parser.add_argument("-d", "--device", default = "cuda", choices = ["cuda", "cpu"], help = "The computation device.")
    parser.add_argument("--model", default = "b", choices = ["b", "b128", "l", "g", "r"], help = "LoMa variant.")
    parser.add_argument("--max-image-size", type = int, default = 1024, help = "The longer side of the processed image.")
    parser.add_argument("--num-keypoints", type = int, default = 2048)
    parser.add_argument("--filter-threshold", type = float, default = 0.1)
    parser.add_argument("--model-dir", default = str(DEFAULT_MODEL_DIR), help = "The directory that contains the LoMa weights.")
    parser.add_argument("--lightglue-model-dir", default = str(DEFAULT_LIGHTGLUE_MODEL_DIR), help = "The directory that contains the LightGlue weights.")
    parser.add_argument("--compare", action = "store_true", help = "Also run SuperPoint + LightGlue for an A/B comparison.")
    parser.add_argument("--limit", type = int, default = 0, help = "Only use the first N images.")
    parser.add_argument("--warmup", type = int, default = 2, help = "Warm-up iterations excluded from the timings.")
    args = parser.parse_args()

    # Must happen before anything imports the models, they capture the device at import time.
    import PythonSystem

    PythonSystem.InitPySys(args.device)

    images = FindImages(Path(args.input_path).resolve())
    if args.limit > 0:
        images = images[: args.limit]
    if len(images) < 2:
        print(f"ERROR: Need at least 2 images, got {len(images)} from {args.input_path}")
        return 1

    print(f"# images: {len(images)}, # pairs: {len(images) * (len(images) - 1) // 2}")
    for path in images:
        with Image.open(path) as image:
            print(f"  {path.name}  {image.width}x{image.height}")

    ResetPeakVram()
    loma_matches = RunLoMa(args, images)

    if args.compare:
        ResetPeakVram()
        splg_matches = RunSuperPointLightGlue(args, images)
        total_loma = sum(c for _, _, c in loma_matches)
        total_splg = sum(c for _, _, c in splg_matches)
        print(f"\n=== Summary ===")
        print(f"  {'pair':>10}  {'LoMa':>8}  {'SP+LG':>8}")
        for (i, j, c0), (_, _, c1) in zip(loma_matches, splg_matches):
            print(f"  {f'{i}-{j}':>10}  {c0:>8}  {c1:>8}")
        print(f"  {'total':>10}  {total_loma:>8}  {total_splg:>8}")

    return 0


if __name__ == "__main__":
    sys.exit(main())