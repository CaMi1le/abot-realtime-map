# Indoor reconstruction results

Measured on 2026-10-22. See [driving results](DRIVING_RESULTS.md) for subsequent front-facing camera experiments and [README](README.md) for reproduction.

The prototype sustained original-speed 10 Hz replay on the tested room sequence with a 500,000-point preview and P95 browser latency below one second. This supports further non-commercial research. Dynamic geometry, long-term drift, and effective loop correction remain unresolved.

This public report contains historical measurements. Original RGB/depth, maps, recordings, and raw logs are not included in Git. [Aggregate benchmark data](docs/benchmark-summary.json) records measurements and local artifact identifiers; the README commands generate new results.

## Realtime measurement

Inference used one GPU reported as an RTX 4090 with approximately 48 GiB total memory, 504×280 input, BF16, cuRoPE, and FlashInfer paged attention. The reported capacity is the server configuration, not a claim about standard retail hardware. Browser rendering used an NVIDIA RTX PRO 1000 through an SSH tunnel.

Timing spans source-timestamp-derived scheduled arrival through decoding, preprocessing, inference, mapping, dense archival, and a rendered image returned by Viser `get_render()`. The final measurement used 1280×800, matching the recording. It includes return-network and image-encoding time, not physical screen scanout latency. The first 12 replay frames are excluded from percentiles and retained in logs.

| Room configuration | Input / completed | Browser P95 | Maximum | Outcome |
|---|---:|---:|---:|---|
| 1,000,000 points; 160×90 acknowledgment | 454 / 454 | 0.439 s | 0.678 s | Diagnostic only |
| 1,000,000 points; 1280×800 acknowledgment | 454 / 454 | 1.078 s | 1.132 s | Missed P95 < 1 s |
| **500,000 points; 1280×800 acknowledgment** | **454 / 454** | **0.758 s** | **0.822 s** | **Passed for this sequence** |

The final profile averaged 53.6 ms model time and 96.2 ms processing per frame. It completed 454 frames in 45.53 seconds, with peak inference allocation of 7,435.6 MiB. All 442 post-warm-up frames received browser acknowledgments, without acknowledgment errors or extra dropped frames. Maximum backlog was about 6.23 frame periods.

The preview coarsened to 225,107 points while dense observations remained on disk. Its default budget is now 500,000 points. The latency run is identified locally as `artifacts/room-preview500k`; the quality/comparison run is `artifacts/room`.

Desk and walking replay completed at sampled 10 Hz, with server-side P95 latencies of 79 ms and 77 ms. They did not measure browser latency and do not count as end-to-end passes. An early SDPA short run averaged about 106 ms total processing and accumulated backlog. cuRoPE, paged attention, and the array-based map cache changed together; this is not a controlled attention-kernel ablation.

## Geometry and trajectory quality

The model received RGB only. TUM depth and ground-truth poses were used exclusively offline. Each trajectory received one global Sim(3) fit, without per-frame scale fitting or ICP. Geometry sampled every tenth selected frame and every fourth pixel, retaining valid depth and confidence of at least 0.5.

| Sequence | Completed frames | Aligned ATE RMSE | Observed-surface mean Chamfer | Depth RMSE |
|---|---:|---:|---:|---:|
| `freiburg1_desk` | 205 | 3.41 cm | 3.63 cm | 7.46 cm |
| `freiburg1_room` | 454 | 10.09 cm | 5.42 cm | 9.00 cm |
| `freiburg3_walking_xyz` | 289; 288 matched to GT | 4.23 cm | 17.41 cm | 52.94 cm |

Walls, desks, cabinets, and the camera path were recognizable. Turns and revisits showed doubled edges and missing surfaces. Walking had relatively low trajectory error but degraded geometry around moving people. Dynamic-object removal is not implemented.

These scores describe the confidence-filtered subset with valid depth. They do not measure whole-room completeness or establish metric scale during inference.

## MASt3R-SLAM comparison

The baseline used the same 454 RGB frames and timestamps, no camera calibration, the upstream headless single-thread configuration, and native 512-pixel preprocessing. On **50 shared keyframe timestamps**, global Sim(3)-aligned ATE was **9.86 cm for ABot** and **12.02 cm for MASt3R-SLAM**.

MASt3R-SLAM took 43.00 seconds, approximately 10.56 FPS. This was unpaced headless throughput including backend startup and export, not browser latency. One room sequence does not establish general superiority. Setup and licenses are in [the comparison guide](docs/comparison.md).

## Offline loop correction

The upstream module ran after replay and took 9.66 seconds. It produced one candidate and one edge, from frame 449 to 410. This did not effectively correct the complete path: ATE changed from 10.0903 cm to 10.0904 cm.

The final optimizer iteration reported `invalid_recurrence`. This is not evidence of improved loop quality. The saved viewer supports before/after inspection; realtime acceptance uses the raw trajectory.

## Memory and correctness

A ten-minute repeated-input test completed 5,999 frames in 600.03 seconds. After frame 100, active GPU allocation stayed between 5,478.6 and 5,489.1 MiB. The last 100 frames averaged 0.012 MiB less than the first 100 in that interval. CPU RSS peaked at 1,414.0 MiB; preview occupancy stayed below 999,719 points under a one-million-point budget.

This test disabled the browser and dense archival. Its repeated input has discontinuous jumps and tests memory behavior only, not long-route quality.

Historical verification included:

- Eight CPU tests for transforms, voxel capacity/color association, Sim(3), timestamp selection/association, PLY, and partial saving after inference failure. Driving work expanded the suite to twelve.
- Sixteen-frame official-interface comparisons for both SDPA and paged attention: maximum pose/point differences were zero; confidence differences were approximately 1.2×10⁻⁷. Reset and empty-input checks passed.
- Browser interaction with three saved scenes and before/after loop views, without JavaScript errors.
- Recalculation of the pinned checkpoint SHA256 on the inference machine.

## Interpretation

ABot-Recon is a useful streaming geometry backbone for further experiments. This short indoor run and synthetic memory test do not establish reliable long-route mapping, phone livestream support, online loop closure, or navigation-grade metric maps.

Revisions are listed in [README](README.md#setup). Project code, model weights, optional acceleration, baseline, and datasets have distinct terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
