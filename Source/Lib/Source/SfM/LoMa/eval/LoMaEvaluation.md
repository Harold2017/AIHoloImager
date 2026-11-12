# LoMa as the SfM Feature Front End: Evaluation

Comparison of [LoMa](https://github.com/davnords/LoMa) against the current SuperPoint + LightGlue
front end used by the **Structure from Motion** module.

- **Branch**: `feature/sfm-loma`
- **Date**: 2026-11-12
- **Status**: Wired into the pipeline behind a runtime switch (`--matcher lightglue|loma`).

## Summary

LoMa-B wins on every axis on a real handheld capture, and it does so *because* it runs at a lower
resolution instead of in spite of it:

| | LoMa-B @1024 | SuperPoint + LightGlue (native) | Change |
|---|---|---|---|
| Feature extraction | **388 ms/image** | 2254 ms/image | **5.8x faster** |
| Pair matching | **26.4 ms/pair** | 53.7 ms/pair | **2.0x faster** |
| Keypoints per image | 2048 | 4096 | |
| Putative matches (28 pairs) | **7688** (275/pair) | 1250 (44.6/pair) | **6.2x more** |
| Epipolar inliers @2px (28 pairs) | **1572** (56.1/pair) | 459 (16.4/pair) | **3.4x more** |
| Pairs with fewer than 20 inliers | **4 / 28** | 23 / 28 | |
| Peak VRAM | **2200 MB** | 6279 MB | **2.9x less** |

The headline is not the 3.4x inlier gain, it is the last row: SuperPoint + LightGlue fails to
produce usable correspondences on 23 of the 28 pairs for this capture, returning as few as
1 to 11 matches. Two-view geometry cannot be estimated from those pairs at all.

### End-to-end reconstruction

Both front ends were then run through the real `StructureFromMotion` pipeline on the same 8 photos
(sequential SfM), which is the measurement that actually matters.

| | LoMa | SuperPoint + LightGlue |
|---|---|---|
| Registered views | **8 / 8** | 5 / 8 (views 3, 4 and 7 dropped) |
| Reconstructed 3D points | **1053** | 399 |
| Feature extraction | 17.6 s | 19.5 s |
| Pair matching | 0.85 s | 1.67 s |
| Geometric filter | 15.5 ms | 8.0 ms |
| Point cloud reconstruction | 0.88 s | 0.18 s |
| End to end | 26 s | 27 s |

Every view registers instead of 5 of 8, with 2.6x more 3D points, at the same wall clock. The lower
epipolar inlier *ratio* measured above turns out to be irrelevant in practice: openMVG's own robust
estimation discards the outliers, and what remains is more good correspondences than the current
front end ever produced.

The in-app timings hide the front-end speed difference. Extraction costs about 2 s per image for
both matchers, because the Python/GPU bridge dominates it (0.26 s of real LoMa work against 2.25 s
of SuperPoint work). LoMa's 5.8x extraction advantage would only become visible if that per-image
bridge cost were reduced.

## Test Data

`Input/`: 8 handheld photos of a small plastic object on a turntable.

- Resolution: 3072x4096, RGB JPEG.
- 8 images give 28 image pairs.

The directory is a 3DGS dataset-prep working tree (COLMAP `features/`, `sparse/`, `dense/`,
`matches.bin`, plus `normals/` renders). Its own logs record that this capture is hard:

| Log | Result |
|---|---|
| `logs/unreg_20261112104116.log` | **unregistered 8/8** |
| `logs/unreg_20261112104809.log` | unregistered 10/32, i.e. 2 of the 8 photos (`IMG_...140621.jpg`, `IMG_...140625.jpg`) |
| `logs/unreg_20261112113108.log` | same as above |

So an independent SfM implementation also struggles here. Any front-end comparison on this data is
therefore measuring behaviour on a genuinely difficult capture, which is exactly where the choice
of matcher matters.

## Environment

| | |
|---|---|
| OS | Windows |
| Python | 3.14.7 (free-threading build), from the project venv |
| torch / torchvision | 2.13.0+cu132 / 0.28.0+cu132 |
| GPU compute capability | 7.5, therefore LoMa autocast uses **float16**, not bfloat16 |
| LoMa settings | `LoMaB()`, `num_keypoints=2048`, `filter_threshold=0.1`, `max_image_size=1024` |
| LightGlue settings | `max_num_keypoints=4096`, native image resolution, `filter_threshold=0.1` |

The two front ends are configured exactly as the pipeline configures them today, so the comparison
reflects the real behaviour of each, not a normalized one. In particular LightGlue runs at the full
12 MP while LoMa runs at 1024 px, because LoMa's detector and descriptor are trained around that scale.

## Results

### Per-pair putative matches

| pair | LoMa | SP+LG | | pair | LoMa | SP+LG |
|---|---|---|---|---|---|---|
| 0-1 | 670 | 248 | | 3-4 | 459 | 65 |
| 0-2 | 259 | **11** | | 3-5 | 115 | **4** |
| 0-3 | 68 | **3** | | 3-6 | 71 | **2** |
| 0-4 | 95 | 12 | | 3-7 | 188 | 27 |
| 0-5 | 382 | **11** | | 4-5 | 189 | **6** |
| 0-6 | 472 | 106 | | 4-6 | 157 | **5** |
| 0-7 | 56 | **2** | | 4-7 | 883 | 148 |
| 1-2 | 470 | 45 | | 5-6 | 825 | 404 |
| 1-3 | 105 | 19 | | 5-7 | 131 | **11** |
| 1-4 | 94 | 19 | | 6-7 | 130 | **11** |
| 1-5 | 319 | **4** | | | | |
| 1-6 | 336 | **3** | | | | |
| 1-7 | 55 | **9** | | | | |
| 2-3 | 474 | 59 | | | | |
| 2-4 | 201 | **5** | | | | |
| 2-5 | 217 | **1** | | | | |
| 2-6 | 156 | **1** | | | | |
| 2-7 | 111 | 9 | | | | |

LoMa never drops below 55 matches on any pair. SuperPoint + LightGlue drops to 11 matches or fewer
on 16 of the 28 pairs (bold).

### Epipolar inlier analysis

A turntable object is not planar, so quality is measured with the fundamental matrix rather than a
homography: normalized 8-point estimation inside RANSAC (1500 iterations), inliers by Sampson
distance at the given pixel threshold. Measurements are at full 3072x4096 resolution.

| epipolar threshold | LoMa-B @1024 | SuperPoint + LightGlue |
|---|---|---|
| 1 px | **1168** (41.7/pair), ratio 0.169, 8/28 pairs <20 | 389 (13.9/pair), ratio 0.249, 23/28 pairs <20 |
| 2 px | **1572** (56.1/pair), ratio 0.215, 4/28 pairs <20 | 459 (16.4/pair), ratio 0.268, 23/28 pairs <20 |
| 4 px | **2049** (73.2/pair), ratio 0.263, 1/28 pairs <20 | 558 (19.9/pair), ratio 0.302, 23/28 pairs <20 |
| 8 px | **2688** (96.0/pair), ratio 0.339, 1/28 pairs <20 | 639 (22.8/pair), ratio 0.320, 23/28 pairs <20 |

Two observations:

1. **The ranking is stable at every threshold.** LoMa leads by 3.0x to 4.2x in absolute inlier
   count regardless of how strict the test is, so the result is not an artifact of threshold choice.
2. **LoMa's inlier *ratio* is lower** (0.17 to 0.34 against 0.25 to 0.32), yet its absolute count is
   far higher because it produces 6.2x more putative matches. For a RANSAC-based geometric filter
   in openMVG, the absolute inlier count is what decides whether a pair is usable.

The comparison is deliberately biased **against** LoMa: its keypoints are detected at 1024 px and
scaled 4x back to full resolution, so its localization is roughly 4x coarser than SuperPoint's
native-resolution keypoints, yet it still wins at the strictest 1 px threshold.

### Input resolution sweep

Higher input resolution does not help LoMa-B, and costs a lot of time. All 28 pairs:

| LoMa `max_image_size` | Extraction | Putative matches | Epipolar inliers @2px | Pairs <20 inliers |
|---|---|---|---|---|
| **1024** | 442 ms/image | **7688** (274.6/pair) | **1572** (56.1/pair) | **4/28** |
| 1536 | 3548 ms/image | 6260 (223.6/pair) | 1426 (50.9/pair) | 9/28 |
| 2048 | 3753 ms/image | 3857 (137.8/pair) | 1033 (36.9/pair) | 15/28 |
| SuperPoint + LightGlue, native | 2163 ms/image | 1250 (44.6/pair) | 459 (16.4/pair) | 23/28 |

LoMa-B's detector and descriptor are trained around the ~1024 scale, so 1024 is both the fastest and
the best setting. There is no accuracy reason to feed it more pixels; the cost grows about 8x from
1024 to 1536 for a worse result.

## Correctness Validation

Before trusting any of the numbers above, the vendored integration was checked in three ways.

1. **Export geometry.** `LoMaExtractor.ExportFeatures` maps keypoints from LoMa's normalized
   `[-1, 1]` space through the padded/processed size back to original pixel coordinates. Verified
   that all exported keypoints land inside the image bounds and that descriptors are L2 normalized.
   This check caught a real width/height ordering bug during development.
2. **Behavioural equivalence with upstream.** Running upstream's own `LoMa.match()` on the same
   images reproduces the same match counts and the same degenerate pairs as the vendored path, at
   `filter_threshold` 0.1, 0.05 and 0.02. The integration is faithful rather than merely plausible.
3. **Planar sanity case.** On a synthetic set built from one photo, where the true relation between
   views is a homography, LoMa reached a 0.69 to 0.79 homography inlier ratio against 0.90 to 0.94
   for SuperPoint + LightGlue, on 1.6x to 2.6x more matches. That test is what motivated measuring
   epipolar inliers on the real, non-planar capture instead.

## Caveats

- **One scene, one failure mode.** This capture is a small, fairly low-texture plastic object on a
  turntable, which is close to the worst case for SuperPoint at 12 MP: 4096 keypoints spread over
  12.6 MP, many of them outside the object. A texture-rich scene may narrow the gap, so a second
  capture type should be evaluated before making LoMa the default.
- **Timing is only valid on an idle GPU.** All numbers above were taken with the GPU to itself. A
  first attempt while another workload shared the card reported 196 s of feature extraction for
  3 images instead of 18 s, which is contention, not a regression. Wall-clock numbers from this
  document are not comparable to a shared-GPU run.
- **float16 autocast.** The test GPU is compute capability 7.5, so LoMa runs mixed precision in
  float16 rather than bfloat16. A bf16-capable GPU may behave slightly differently.
- **Keypoint precision.** Detecting at 1024 px and reporting at 3072x4096 gives a localization
  granularity around 4 px at full resolution. The 1 px epipolar result and the full reconstruction
  both suggest this is tolerable, but it remains a structural difference from the current front end.
- **The default is still LightGlue.** The switch is opt-in so that nothing changes silently;
  flipping the default is a one-line change once a second scene type has been checked.

## Reproduction

The harness needs only the project venv, no C++ build.

```powershell
External\Python\Venv\Venv\Scripts\python.exe Source\Lib\Source\SfM\LoMa\eval\eval.py `
    -i <image-dir> --compare
```

Options: `--model b|b128|l|g|r`, `--max-image-size`, `--num-keypoints`, `--filter-threshold`,
`--limit`, `--warmup`, `--device cuda|cpu`.

The end-to-end numbers come from the application itself, which prints the registered view and point
counts during a normal run:

```powershell
build\bin\Release\AIHoloImager.exe -I <image-dir> -O <out-dir>\Mesh.glb --matcher loma
build\bin\Release\AIHoloImager.exe -I <image-dir> -O <out-dir>\Mesh.glb --matcher lightglue
```

Note that the SfM input directory must contain only the photos; it does not filter by extension, so
a working directory with other files in it will try to load them as images.

## Implementation Notes

- The vendored package is `Source/Lib/Source/SfM/LoMa/`. It keeps LoMa's inference code and drops
  the training-only modules, so **OpenCV and einops are not required**; neither is in
  `External/Python/Venv/requirements.txt`.
- `Source/Lib/Source/SfM/LoMa/weights.py` resolves pretrained weights from the deployed
  `Models/LoMa/` directory before ever touching the network. **`loma_B.pt` already contains the
  frozen DINOv2 and DaD detector weights**, so neither the 1.1 GB standalone DINOv2 checkpoint nor
  the separate `dad.pth` has to be shipped. Only `loma_B.pt` (723 MB) is deployed.
- `Source/Lib/Source/SfM/LoMaExtractor.py` and `LoMaMatcher.py` mirror the existing
  `Extractor.py` / `Matcher.py` contract (`Extract`, `ExportFeatures`, `Match`) so the C++ side
  needs no openMVG changes. The visible differences are that LoMa consumes RGB instead of grayscale,
  and that it requires the resolution to be a multiple of the DINOv2 patch size (14 for LoMa-B),
  which the extractor handles by padding and by mapping keypoints back through the processed size.
- `AIHoloImager::FeatureMatcher` selects the front end at runtime; the Python objects are created
  under the matching module name (`LoMaExtractor`/`LoMaMatcher` or `Extractor`/`Matcher`) during the
  existing async init, and for LoMa the grayscale pass is skipped entirely and the original RGB
  texture is handed to the extractor. The `USES_SP_LG` compile switch is left in place, so the
  SIFT + Cascade Hashing fallback still compiles as a separate configuration.

## Conclusion

On this capture LoMa-B is not a speed/accuracy trade-off, it is strictly better: it registers all 8
views instead of 5, reconstructs 2.6x more 3D points, and does so in the same wall clock while using
a third of the memory. The front-end advantage survives conversion into a full reconstruction, which
was the open question.
