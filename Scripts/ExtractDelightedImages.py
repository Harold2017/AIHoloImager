# Copyright (c) 2026 Minmin Gong
#

"""Extract the delighted images with the Exif metadata of the original photos.

When AIHoloImager is built with AIHI_KEEP_INTERMEDIATES, the delighted images are written to
<output-dir>/Tmp/Delighter/Delighted_<name>.jpg. This script matches each of them with its original
photo, puts the original Exif metadata back into the JPEG, and writes the results into a separate
directory.
"""

import argparse
import sys
from pathlib import Path

JPEG_SOI = b"\xff\xd8"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXIF_HEADER = b"Exif\x00\x00"
DELIGHTED_PREFIX = "Delighted_"
MAX_APP1_SIZE = 0xFFFF
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp")


def _jpeg_header_segments(data: bytes) -> list[tuple[int, int, int]]:
    """Returns the (marker, start, end) of every segment in front of the first SOS marker."""
    segments: list[tuple[int, int, int]] = []
    offset = len(JPEG_SOI)
    while (offset + 4) <= len(data):
        if data[offset] != 0xFF:
            break

        marker = data[offset + 1]
        if marker == 0xFF:
            offset += 1
            continue
        if (marker == 0x01) or (0xD0 <= marker <= 0xD9):
            offset += 2
            continue
        if marker == 0xDA:
            break

        size = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if size < 2:
            break

        segments.append((marker, offset, offset + 2 + size))
        offset += 2 + size

    return segments


def _is_exif_segment(data: bytes, marker: int, start: int, end: int) -> bool:
    return (marker == 0xE1) and data[start + 4 : end].startswith(EXIF_HEADER)


def _read_jpeg_exif(data: bytes) -> bytes | None:
    for marker, start, end in _jpeg_header_segments(data):
        if _is_exif_segment(data, marker, start, end):
            return data[start + 4 + len(EXIF_HEADER) : end]
    return None


def _read_png_exif(data: bytes) -> bytes | None:
    offset = len(PNG_SIGNATURE)
    while (offset + 8) <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        if chunk_type == b"eXIf":
            return data[offset + 8 : offset + 8 + size]
        if chunk_type in (b"IDAT", b"IEND"):
            return None
        offset += 12 + size
    return None


def read_exif(path: Path) -> bytes | None:
    """Returns the TIFF data of the Exif metadata inside path, or None if there is none."""
    try:
        data = path.read_bytes()
    except OSError as ex:
        print(f"WARNING: Unable to read {path}: {ex}", file=sys.stderr)
        return None

    if data.startswith(JPEG_SOI):
        return _read_jpeg_exif(data)
    if data.startswith(PNG_SIGNATURE):
        return _read_png_exif(data)
    return None


def insert_exif(data: bytes, exif: bytes) -> bytes | None:
    """Replaces the Exif metadata of a JPEG, or returns None if it is too large."""
    payload = EXIF_HEADER + exif
    if (len(payload) + 2) > MAX_APP1_SIZE:
        return None

    segments = _jpeg_header_segments(data)
    app1 = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    kept = b"".join(
        data[start:end]
        for marker, start, end in segments
        if not _is_exif_segment(data, marker, start, end)
    )
    tail = data[segments[-1][2] :] if segments else data[len(JPEG_SOI) :]

    return data[: len(JPEG_SOI)] + app1 + kept + tail


def process(delighted_path: Path, original_path: Path, output_path: Path) -> bool:
    delighted = delighted_path.read_bytes()
    if not delighted.startswith(JPEG_SOI):
        print(f"WARNING: {delighted_path.name} is not a JPEG, skipped.", file=sys.stderr)
        return False

    exif = read_exif(original_path)
    if exif is None:
        print(f"WARNING: {original_path.name} has no Exif metadata.", file=sys.stderr)
        output_path.write_bytes(delighted)
        return True

    delighted = insert_exif(delighted, exif)
    if delighted is None:
        print(
            f"WARNING: The Exif metadata of {original_path.name} is too large, skipped.",
            file=sys.stderr,
        )
        return False

    output_path.write_bytes(delighted)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy the Exif metadata of the original photos into the delighted images."
    )
    parser.add_argument("input_dir", type=Path, help="The directory of the original photos.")
    parser.add_argument("delighted_dir", type=Path, help="The directory of the delighted images.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Where to write the images. <delighted-dir>/Exif by default.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    delighted_dir: Path = args.delighted_dir
    output_dir: Path = args.output_dir if args.output_dir is not None else delighted_dir / "Exif"

    if not input_dir.is_dir():
        print(f"ERROR: {input_dir} is not a directory.", file=sys.stderr)
        return 1
    if not delighted_dir.is_dir():
        print(f"ERROR: {delighted_dir} is not a directory.", file=sys.stderr)
        return 1

    originals: dict[str, Path] = {}
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and (path.suffix.lower() in IMAGE_SUFFIXES):
            originals.setdefault(path.stem.lower(), path)

    delighted_paths = sorted(
        path
        for path in delighted_dir.iterdir()
        if path.is_file()
        and path.name.startswith(DELIGHTED_PREFIX)
        and (path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    )
    if not delighted_paths:
        print(f"ERROR: No delighted images found in {delighted_dir}.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    num_written = 0
    num_missing = 0
    for delighted_path in delighted_paths:
        original_path = originals.get(delighted_path.stem[len(DELIGHTED_PREFIX) :].lower())
        if original_path is None:
            print(f"WARNING: No original photo for {delighted_path.name}.", file=sys.stderr)
            num_missing += 1
            continue

        if process(delighted_path, original_path, output_dir / f"{original_path.stem}.jpg"):
            num_written += 1

    message = f"Wrote {num_written} image(s) to {output_dir}."
    if num_missing > 0:
        message += f" {num_missing} delighted image(s) have no original photo."
    print(message)

    return 0 if (num_written > 0) else 1


if __name__ == "__main__":
    sys.exit(main())
