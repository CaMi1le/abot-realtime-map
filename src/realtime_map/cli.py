# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--dataset-format", choices=["tum", "kitti", "waymo"], default="tum")
    p.add_argument("--waymo-context", help="One official Waymo context; state never crosses contexts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--occ-checkpoint", help="Trained Waymo OCC predictor checkpoint")
    p.add_argument("--occ-adapter", help="Trained OCC-to-ABot adapter checkpoint")
    p.add_argument("--occ-mode", choices=["predicted", "zero", "shuffle"], default="predicted")
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default="models/abot_recon.safetensors")
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--attention", choices=["sdpa", "paged", "auto"], default="sdpa")
    p.add_argument("--confidence", type=float, default=.5)
    p.add_argument("--point-stride", type=int, default=3)
    p.add_argument("--voxel", type=float, default=0, help="Model units; 0 selects first-frame median depth / 150")
    p.add_argument("--max-points", type=int, default=500_000)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--measure-browser", action="store_true")
    p.add_argument("--render-width",type=int,default=1280)
    p.add_argument("--render-height",type=int,default=800)
    p.add_argument("--port", type=int, default=8890)
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--chunk-frames", type=int, default=0, help="Archive independent map chunks; 0 disables")
    p.add_argument("--follow-camera", action="store_true", help="Show the current map chunk and follow the camera")
    p.add_argument("--keep-open", action="store_true")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--stress-seconds", type=float)
    a = p.parse_args()
    if bool(a.occ_checkpoint) != bool(a.occ_adapter):
        p.error("OCC prediction and adapter checkpoints must be supplied together")
    if a.occ_checkpoint and (a.dataset_format != 'waymo' or a.loop or a.stress_seconds):
        p.error("OCC conditioning requires Waymo and a causal stream without offline loop or synthetic repetition")
    if not a.occ_checkpoint and a.occ_mode != 'predicted':
        p.error("OCC ablations require both OCC checkpoints")
    if a.dataset_format == 'waymo' and (a.loop or a.stress_seconds):
        p.error("Waymo runs must preserve independent contexts and original sensor timing")
    if a.fps <= 0 or a.point_stride <= 0 or a.voxel < 0 or a.max_points <= 0:
        p.error("fps, point stride and point budget must be positive; voxel must be nonnegative")
    if a.max_points > 1_000_000: p.error("Preview point budget cannot exceed one million")
    if not 0 <= a.confidence <= 1: p.error("confidence must be in [0, 1]")
    if a.max_frames is not None and a.max_frames <= 0: p.error("max frames must be positive")
    if a.stress_seconds is not None and a.stress_seconds <= 0: p.error("stress duration must be positive")
    if a.measure_browser and (not a.viewer or not a.realtime):
        p.error("--measure-browser requires --viewer --realtime")
    if min(a.render_width,a.render_height)<64: p.error("Render dimensions must be at least 64")
    if a.no_dense and a.loop: p.error("Loop correction needs archived local points")
    if a.chunk_frames < 0: p.error("chunk frames must be nonnegative")
    if a.follow_camera and not a.chunk_frames: p.error("--follow-camera requires --chunk-frames")
    if a.stress_seconds and a.loop: p.error("Synthetic repeated input is for memory testing only")
    from .runner import run
    run(a)


if __name__ == "__main__": main()
