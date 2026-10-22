# Front-facing driving reconstruction results

Measured on 2026-10-22: two complete KITTI routes, 7,422 original RGB frames, and original-speed replay. See [README](README.md#front-facing-driving-camera) for reproduction and [aggregate benchmark data](docs/benchmark-summary.json) for machine-readable measurements.

The results support research into local road visualization. RGB-only input produces recognizable road, building, vegetation, and vehicle point clouds; the five-kilometer run kept up with native cadence. Global drift remains large, and the two routes did not both meet the latency target.

## Scope of the public release

This source repository includes code, aggregate measurements, and reproduction instructions. KITTI images, ground-truth files, maps, raw logs, and recordings remain local experiment artifacts and are not distributed here.

The original recordings were 4 min 55 s for KITTI 05 and 8 min 07 s for KITTI 02, at 960×600. They preserve wall-clock timing, including initial waiting and backlog, without speed changes. These recordings are not hosted by this repository. The browser harness can record a new run using locally obtained data.

## Full-replay measurements

These figures describe **streaming output before offline loop correction**. Both replays ran sequentially on the same available GPU reported as an RTX 4090, bound by UUID. Browser rendering used an NVIDIA RTX PRO 1000.

| Metric | KITTI 05 | KITTI 02 |
|---|---:|---:|
| Ground-truth route length | 2.206 km | 5.068 km |
| Input / processed frames | 2,761 / 2,761 | 4,661 / 4,661 |
| Source duration | 287.537 s | 483.276 s |
| Processing wall time | 287.779 s | 483.608 s |
| Native input cadence | 9.599 Hz | 9.643 Hz |
| Mean model time | 54.12 ms | 53.31 ms |
| Browser latency P95 | **1.268 s: missed target** | **0.310 s: passed at native cadence** |
| Maximum browser latency | 2.241 s | 0.993 s |
| Maximum backlog | 19.53 frames | 7.59 frames |
| Peak inference GPU allocation | 7,437.65 MiB | 7,443.67 MiB |
| Maximum CPU RSS | 1,391.13 MiB | 1,404.55 MiB |
| Global Sim(3)-aligned ATE RMSE | **13.53 m** | **70.40 m** |
| Aligned 100–800 m segment translation error | 3.50% | 7.14% |
| Aligned final-frame position error | 36.09 m | 53.69 m |
| Archived detailed map chunks | 28 | 47 |

Neither run establishes strict 10 Hz performance: native timestamps are approximately 9.6 Hz. All frames were retained without retiming or extra dropping to hide backlog. Reports distinguish `native_cadence_realtime_pass` from strict `realtime_pass`; the latter is false for both runs.

Latency spans scheduled arrival through loading, preprocessing, inference, mapping, archival, transmission, browser rendering, and the returned image. It includes return-trip time and render-acknowledgment overhead. Window, render requests, and recording all used 960×600.

Model loading and independent kernel warm-up preceded replay; state was reset afterward. Percentiles exclude the first 12 replay frames; logs retain every frame. KITTI 05 accumulated and recovered from backlog around seconds 16–50, causing its P95 failure. KITTI 02 also had a spike close to one second.

## Offline loop correction

| Metric | KITTI 05 | KITTI 02 |
|---|---:|---:|
| Detected and inserted loop edges | 11 | 8 |
| ATE RMSE: streaming → loop | 13.53 → **5.99 m** | 70.40 → **38.97 m** |
| Segment translation error: streaming → loop | 3.50% → **2.28%** | 7.14% → **6.30%** |
| Upstream loop-module time | 57.75 s | 81.31 s |

Module time includes retrieval, reinference, and pose-graph optimization, excluding subsequent map rebuilding/export. Each trajectory receives a separate global Sim(3) alignment.

Both runs reduced error, but the final inner PCG solves terminated at `max_iterations` without convergence. KITTI 02's final relative residual was approximately 0.158. This default offline backend did not establish global consistency and is excluded from realtime acceptance.

### Remaining error

**38.97 m is whole-trajectory ATE RMSE, not the gap between loop endpoints.** Analysis of detected revisit pairs found:

| Metric | KITTI 05 | KITTI 02 |
|---|---:|---:|
| Median detected-pair relative translation error: raw → loop | 11.75 → **3.01 m** | 45.88 → **5.32 m** |
| Maximum detected-pair translation error after loop | 5.25 m | 9.93 m |
| Maximum whole-trajectory position error after loop | 11.98 m | **85.09 m** |

Pair error compares predicted and ground-truth relative poses, without assuming the revisiting cameras occupy identical positions. Saved global alignments are reused without per-segment fitting or ground-truth-based changes to reconstruction.

KITTI 02's largest residual position error occurs at source time **274.3 s**, after **2.805 km** of travel. All eight loop edges have later endpoints at **436–471 s**, connected to earlier stretches at approximately **97–106 s** and **188–204 s**. Constraints are concentrated; this analysis does not measure retrieval recall or count missed true revisits.

These observations motivate examining loop-constraint quality and global pose/scale optimization. They do not isolate causes through an ablation or prove that more iterations will eliminate drift.

With complete local saved runs, reproduce the diagnosis using:

```bash
.venv/bin/python scripts/diagnose-driving-loop.py artifacts/driving/kitti05-final artifacts/driving/kitti02-final
```

This CPU-only script writes PNG/PDF plots and JSON without rerunning inference or modifying reconstruction.

## Data, map, and evaluation boundaries

Input was rectified front-left color (`image_2`) from [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). Approximately 5.8 GiB of selected official archive members were downloaded through byte-range requests; all 7,422 RGB files passed ZIP CRC32 checks.

Only RGB entered the model. Calibration converted camera-0 ground truth to camera-2 ground truth offline. Ground truth never selected reconstruction scale or entered inference. Whole-trajectory Sim(3) alignment means start-point error need not be zero.

Segment metrics use 100–800 m segments starting every ten frames, after global scale alignment. They are **not official metric-scale leaderboard scores**. No LiDAR geometry accuracy was measured.

Roads and buildings were recognizable, while vehicles, vegetation, and occlusion boundaries had fragments and holes. Visual inspection alone does not establish geometry accuracy. There is no dynamic-object removal, and scale remains uncalibrated. Maximum ground-truth speeds were approximately 42 and 53 km/h. Highway driving, night, rain, strong dashcam distortion, and in-vehicle compute were not tested.

The pinned model used 504×280, BF16, paged attention, and cuRoPE. KITTI frames were aspect-preserved and vertically padded; padding was excluded from mapping. Confidence was at least 0.5, pixel stride was 4, and each chunk had a 200,000-point budget over 100 frames. Initial voxel size used predicted first-frame depth, without ground-truth scale.

The 75 archived chunks contain approximately 9.47 million samples, potentially including repeated observations across chunks. Chunk resolutions are independently managed. Global maps are bounded overviews; the saved viewer shows at most three detailed chunks, capped at 600,000 points.

The live chase preview replaces its displayed chunk every 100 frames. Background geometry can disappear at boundaries while remaining archived. Missing preview background alone does not establish a model failure. Steady-state GPU allocation was approximately 5,473–5,489 MiB without continuous growth over these routes.

## Earlier trials and verification

| Trial | Conditions | Browser P95 | Outcome |
|---|---|---:|---|
| `kitti05-smoke` | 200 frames, 1280×800, stride 3, 500k points | 1.183 s | Missed target |
| `kitti05-smoke-stride4` | 200 frames, 1280×800, stride 4, 200k points | 0.899 s | Short trial only |
| `kitti05-smoke-960` | 300 frames, 960×600, stride 4, 200k points | 0.221 s | Short trial only |
| `kitti05` | First long replay during another dataset transfer | 28.533 s | Stopped; 517 frames completed all stages |

The interrupted run was not counted as complete. Shared-host load and parallel transfer were confounding conditions, not established sole causes. Final runs started after all input transfers and checks completed.

Twelve CPU tests passed. A real-weight sixteen-frame comparison against upstream had zero maximum pose/point difference and approximately 1.19×10⁻⁷ confidence difference. Reset, empty input, partial archival, camera extrinsics, timestamps, and padding checks passed.

All 75 chunks had continuous frame coverage, matching manifest counts, and finite coordinates. Browser tests exercised routes, chunk selection, neighbors, ground-truth overlay, and loop switching without page errors. MP4 and original WEBM durations matched.

## Interpretation

Local road visualization is a reasonable research target. Reliable long-range maps require further work on pose/scale consistency, loop constraints, and dynamic objects. Streaming accumulation plus default offline correction does not establish that capability.

All experiments used the revisions in [README](README.md). Models, acceleration, baseline, and datasets retain independent terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
