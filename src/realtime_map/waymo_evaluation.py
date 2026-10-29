# SPDX-License-Identifier: Apache-2.0
"""Offline-only Waymo trajectory/depth evaluation. GT never enters inference."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .evaluation import sim3
from .occ_training import depth_samples
from .waymo import WAYMO_FROM_CV, read_waymo


def evaluate_waymo(run, scene):
    run, scene = Path(run), Path(scene)
    if not (scene / "source-manifest.json").exists():
        scene = scene / "scene"
    inputs = json.loads((run / "input.json").read_text())
    contexts = {r.get("context") for r in inputs}
    if len(contexts) != 1 or None in contexts:
        raise ValueError(
            "Waymo evaluation requires one explicitly identified official context"
        )
    frames = read_waymo(scene, context=next(iter(contexts)))
    by_time = {f.timestamp_us: f for f in frames}
    rows = {
        int(r["timestamp_micros"]): r
        for r in json.loads((scene / "source-manifest.json").read_text())["frames"]
    }
    poses = np.load(run / "poses.npy")
    matched = [by_time[round(r["timestamp"] * 1e6)] for r in inputs[: len(poses)]]
    gt = np.array(
        [
            np.asarray(
                rows[f.timestamp_us]["sensors"]["FRONT"]["source_world_from_camera"]
            ).reshape(4, 4)
            @ WAYMO_FROM_CV
            for f in matched
        ]
    )
    result = {
        "context": next(iter(contexts)),
        "frames": len(poses),
        "alignment": "one global trajectory Sim(3); no per-frame scale/ICP",
        "metric_scale_observed_at_inference": False,
        "tracking": {
            "available": False,
            "reason": "ABot does not produce instance IDs; evaluate fixed external candidates separately",
        },
    }
    try:
        if len(poses) < 3:
            raise ValueError("At least three matched poses are needed")
        scale, rotation, translation = sim3(poses[:, :3, 3], gt[:, :3, 3])
    except ValueError as exc:
        result.update(
            trajectory_available=False, reason=str(exc), depth_available=False
        )
        return result
    error = np.linalg.norm(
        scale * poses[:, :3, 3] @ rotation.T + translation - gt[:, :3, 3], axis=1
    )
    result.update(
        trajectory_available=True,
        alignment_scale=float(scale),
        ate_rmse_m=float(np.sqrt(np.mean(error**2))),
    )
    errors, rel = [], []
    for i, frame in enumerate(matched):
        path = run / "frames" / f"{i:06d}.npz"
        if not path.is_file():
            continue
        with np.load(path) as prediction:
            local, confidence = prediction["points"], prediction["confidence"]
        height, width = local.shape[:2]
        camera = frame.cameras[0]
        resized = round(camera.height * width / camera.width)
        transform = SimpleNamespace(
            source_height=camera.height,
            source_width=camera.width,
            target_height=height,
            target_width=width,
            resized_height=resized,
            crop_top=round((resized - height) / 2) if resized > height else 0,
            pad_top=(height - resized) // 2 if resized < height else 0,
            pad_bottom=(height - resized + 1) // 2 if resized < height else 0,
        )
        ids, target = depth_samples(
            scene, rows[frame.timestamp_us], frame, transform, "cpu"
        )
        ids, target = ids.numpy(), target.numpy()
        depth = scale * local[..., 2].reshape(-1)[ids]
        good = np.isfinite(depth) & (depth > 0) & (confidence.reshape(-1)[ids] >= 0.5)
        errors.append(depth[good] - target[good])
        rel.append(np.abs(depth[good] - target[good]) / target[good])
    count = sum(len(x) for x in errors)
    result.update(
        depth_available=bool(count),
        depth_samples=count,
        depth_rmse_m=float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
        if count
        else None,
        depth_abs_rel=float(np.mean(np.concatenate(rel))) if count else None,
        geometry_protocol="Observed nearest-return camera-z only; sparse LiDAR, not dense occupancy truth",
    )
    return result


def association_scores(predictions, truth):
    """ID continuity for already matched GT/predicted object pairs.

    Inputs map frame_id to pairs (gt_id,pred_id). Matching/candidates must be held
    fixed across experiments and produced independently of the GT IDs. This is
    an association-only diagnostic, not an official tracking benchmark score.
    """
    previous = {}
    switches = matches = 0
    for frame in sorted(predictions):
        pairs = predictions[frame]
        if len({g for g, p in pairs}) != len(pairs) or len(
            {p for g, p in pairs}
        ) != len(pairs):
            raise ValueError("Tracking matches must be one-to-one within each frame")
        for gt, pred in pairs:
            if gt not in truth.get(frame, []):
                raise ValueError("Matched GT ID is absent from this frame")
            if gt in previous and previous[gt] != pred:
                switches += 1
            previous[gt] = pred
            matches += 1
    total = sum(len(v) for v in truth.values())
    return {
        "association_only": True,
        "matched_observations": matches,
        "gt_observations": total,
        "matched_fraction": matches / total if total else None,
        "id_switches": switches,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("scene")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    value = evaluate_waymo(args.run, args.scene)
    Path(args.output).write_text(json.dumps(value, indent=2))
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
