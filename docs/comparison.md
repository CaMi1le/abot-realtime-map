# Optional MASt3R-SLAM comparison

This baseline is separate from the ABot-Recon inference environment. MASt3R-SLAM and the included `scripts/prepare-comparison.py` adaptation are **CC BY-NC-SA 4.0**. MASt3R checkpoints have additional independent terms. Read [third-party notices](../THIRD_PARTY_NOTICES.md) before setup or downloading assets.

## Tested scope

The historical experiment used Python 3.12, CUDA 12.1, GCC/G++ 12, an SM 8.9 GPU, and upstream commit `e6f4e3d474fad0e11f561482012be864ba8c3f17`. The helpers are specific to that environment. They create a separate `.venv-mast3r` that reuses the main environment's Python 3.12 packages through a `.pth` file.

The headless adaptation removes eager visualization imports, adds timing/export instrumentation, and restricts the CUDA build to SM 8.9. Optional Gradio/Pyglet UI dependencies are omitted. Tracking logic is unchanged. Modified upstream files remain under their original license.

## Reproduce

First complete the main [setup](../README.md#setup), obtain the TUM room sequence, and produce `artifacts/room` if comparing against a new ABot run. Then, from the repository root:

```bash
bash scripts/setup-comparison.sh
mkdir -p models/mast3r
.venv/bin/python scripts/download-mast3r-ranges.py
.venv/bin/python scripts/prepare-comparison-data.py
```

The input preparation copies the ABot-selected timestamps and creates links to the same RGB files. It uses `artifacts/room/input.json` when present, otherwise the same 10 Hz TUM sampling. The path must include `data/tum/` so the upstream loader preserves TUM timestamps.

The upstream runner also needs the retrieval checkpoint and codebook. After reviewing the checkpoint terms, obtain those assets from the original publisher and run from the upstream directory:

```bash
mkdir -p vendor/MASt3R-SLAM/checkpoints
ln -s ../../../models/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth vendor/MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
curl -fL https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -o vendor/MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
curl -fL https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -o vendor/MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
cd vendor/MASt3R-SLAM
../../.venv-mast3r/bin/python main.py --dataset ../../data/tum/mast3r-freiburg1_room --config config/realtime-map.yaml --save-as realtime-map-room --no-viz
cd ../..
```

The generated configuration uses `use_calib: False`, `single_thread: True`, and no further subsampling. These commands require the tested CUDA/compiler environment; the baseline is optional.

After exporting its TUM-format trajectory:

```bash
.venv/bin/python scripts/evaluate-comparison.py artifacts/room vendor/MASt3R-SLAM/logs/realtime-map-room/mast3r-freiburg1_room.txt
```

The evaluator associates common keyframe timestamps, fits a separate global Sim(3) for each trajectory, and reports aligned ATE. Ground truth is used only for evaluation. Headless unpaced throughput must be reported separately from browser end-to-end latency. The historical single-room result is in [RESULTS.md](../RESULTS.md#mast3r-slam-comparison).
