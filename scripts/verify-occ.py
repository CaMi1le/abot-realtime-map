# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint baseline/zero/predicted OCC integration smoke verification."""

import argparse
import json
from pathlib import Path

import numpy as np

from realtime_map.backend import ABotStream
from realtime_map.occ_fusion import EncoderConditioner
from realtime_map.occ_model import OccRuntime
from realtime_map.waymo import read_waymo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--checkpoint", default="models/abot_recon.safetensors")
    p.add_argument("--occ-checkpoint", required=True)
    p.add_argument("--occ-adapter", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--attention", default="sdpa")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    frame = read_waymo(args.input, max_frames=1, context=args.context)[0]
    backend = ABotStream(args.checkpoint, args.attention, device=args.device)
    try:
        baseline = backend.step(frame)
        backend.reset()
        runtime = OccRuntime.from_checkpoints(
            args.occ_checkpoint, args.occ_adapter, args.device, "zero"
        )
        runtime.validate_abot_checkpoint(args.checkpoint)
        backend.occ_runtime = runtime
        backend.conditioner = EncoderConditioner(
            backend.network.encoder, runtime.adapter
        )
        zero = backend.step(frame)
        np.testing.assert_array_equal(baseline.camera_pose, zero.camera_pose)
        np.testing.assert_array_equal(baseline.local_points, zero.local_points)
        backend.reset()
        runtime.mode = "predicted"
        predicted = backend.step(frame)
        differences = {
            "pose_max_abs": float(
                np.max(np.abs(predicted.camera_pose - baseline.camera_pose))
            ),
            "point_max_abs": float(
                np.max(np.abs(predicted.local_points - baseline.local_points))
            ),
        }
        if not np.isfinite(predicted.local_points).all():
            raise AssertionError("Non-finite conditioned geometry")
        result = {
            "passed": True,
            "released_abot_checkpoint": str(args.checkpoint),
            "source": "real Waymo FRONT RGB + online predicted OCC; no GT inference",
            "zero_prior_exact_pose_and_point_parity": True,
            "conditioned_encoder_calls": backend.conditioner.calls,
            "conditioned_output_differences": differences,
            "device": args.device,
            "attention": args.attention,
            "occ_cameras": list(runtime.source.cameras),
            "projection": runtime.projector.config() if hasattr(runtime.projector, "config") else {"kind": "hard"},
            "occ_predict_ms": predicted.occ_predict_ms,
            "occ_project_ms": predicted.occ_project_ms,
            "model_ms": predicted.model_ms,
            "realtime_evidence": False,
            "quality_improvement_evidence": False,
            "limitations": "One frame only; checkpoint training quality is not established by this test",
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
