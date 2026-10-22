# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Offline-only ground-truth evaluation. No ground truth enters inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .mapping import VoxelMap, transform_points, write_ply, write_trajectory


def sim3(source, target):
    """Least-squares Umeyama alignment, without reflection."""
    source, target = np.asarray(source), np.asarray(target)
    x, y = source-source.mean(0), target-target.mean(0)
    variance = np.mean(np.sum(x*x, 1))
    if variance < 1e-12: raise ValueError("Degenerate trajectory; cannot estimate scale")
    u, d, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3); sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = np.dot(d, sign) / variance
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    return scale, rotation, translation


def associate(times, reference_times, tolerance=.03):
    times, reference_times = np.asarray(times), np.asarray(reference_times)
    right = np.searchsorted(reference_times, times).clip(0, len(reference_times)-1)
    left = (right-1).clip(0)
    idx = np.where(abs(reference_times[left]-times) < abs(reference_times[right]-times), left, right)
    return idx, abs(reference_times[idx]-times) <= tolerance


def evaluate(run, dataset, suffix=""):
    run, dataset = Path(run), Path(dataset)
    poses = np.load(run / f"poses{suffix}.npy")
    times = np.atleast_1d(np.loadtxt(run / "timestamps.txt"))
    gt = np.loadtxt(dataset / "groundtruth.txt")
    ids, valid = associate(times, gt[:, 0])
    if valid.sum() < 3: raise ValueError("Fewer than three associated GT poses")
    scale, rotation, translation = sim3(poses[valid, :3, 3], gt[ids[valid], 1:4])
    aligned = scale * poses[:, :3, 3] @ rotation.T + translation
    errors = np.linalg.norm(aligned[valid] - gt[ids[valid], 1:4], axis=1)
    report = {"trajectory_alignment": "one global Sim(3), fitted to associated trajectory",
              "associated_frames": int(valid.sum()), "total_frames": len(poses),
              "alignment_scale": float(scale), "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
              "ate_p95_m": float(np.percentile(errors, 95)), "ate_max_m": float(errors.max()),
              "metric_scale_observed_at_inference": False}
    np.savez(run / f"alignment{suffix}.npz", scale=scale, rotation=rotation, translation=translation)
    depth_entries = [line.split() for line in (dataset / "depth.txt").read_text().splitlines()
                     if line.strip() and not line.startswith("#")]
    dt = np.array([float(x[0]) for x in depth_entries])
    dids, dvalid = associate(times, dt, .035)
    # TUM recommends ROS default intrinsics without re-undistortion for its
    # pre-registered depth maps (official rgbd-dataset/file_formats).
    intrinsics = (525.0, 525.0, 319.5, 239.5)
    report['depth_intrinsics_fx_fy_cx_cy'] = list(intrinsics)
    report['depth_calibration_note'] = 'TUM recommended ROS default; no re-undistortion of pre-registered depth'
    fx, fy, cx, cy = intrinsics
    predicted, actual, zerrors = [], [], []
    for i in range(0, len(poses), 10):
        file = run / "frames" / f"{i:06d}.npz"
        if not valid[i] or not dvalid[i] or not file.is_file(): continue
        with np.load(file) as f:
            lp, conf = f["points"], f["confidence"]
        depth = np.asarray(Image.open(dataset / depth_entries[dids[i]][1]), np.float32) / 5000
        source_h, source_w = depth.shape
        h, w = lp.shape[:2]
        resized_h = round(source_h * w / source_w)
        crop_top = round((resized_h-h)/2)
        v, u = np.mgrid[0:h:4, 0:w:4]
        # Inverse pixel-centre resize convention, including centre crop.
        ox = (u+.5)*source_w/w-.5
        oy = (v+crop_top+.5)*source_h/resized_h-.5
        sx = np.round(ox).astype(int).clip(0, source_w-1)
        sy = np.round(oy).astype(int).clip(0, source_h-1)
        z = depth[sy, sx]
        sample = lp[v, u]
        good = (z > .1) & (z < 10) & np.isfinite(sample).all(-1) & (sample[..., 2] > 0) & (conf[v,u] >= .5)
        local_gt = np.stack([(ox-cx)/fx*z, (oy-cy)/fy*z, z], -1)[good]
        gtpose = np.eye(4)
        gtpose[:3,:3] = Rotation.from_quat(gt[ids[i],4:8]).as_matrix()
        gtpose[:3,3] = gt[ids[i],1:4]
        actual.append(transform_points(local_gt, gtpose))
        world = transform_points(sample[good], poses[i])
        predicted.append(scale * world @ rotation.T + translation)
        zerrors.append(scale * sample[good,2] - z[good])
    if predicted and sum(map(len,predicted)):
        p, g = np.concatenate(predicted), np.concatenate(actual)
        dp = cKDTree(g).query(p, workers=4)[0]
        dg = cKDTree(p).query(g, workers=4)[0]
        precision, recall = float(np.mean(dp < .25)), float(np.mean(dg < .25))
        ze = np.concatenate(zerrors)
        report.update({"geometry_protocol": "Every 10th inference frame, pixel stride 4; same global trajectory Sim(3); matched observed valid depth, confidence >= .5; no ICP. Observed-surface subset, not full-room completeness.",
                       "evaluated_points": len(p), "chamfer_mean_m": float((dp.mean()+dg.mean())/2),
                       "precision_025m": precision, "recall_025m": recall,
                       "f1_025m": 2*precision*recall/(precision+recall+1e-12),
                       "depth_rmse_m": float(np.sqrt(np.mean(ze**2))),
                       "depth_median_abs_m": float(np.median(abs(ze)))})
    (run / f"evaluation{suffix}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def refine(backend, frames, poses, out, args):
    import torch
    from abot_recon.loop_closure import refine_trajectory
    config = backend.api.config.override(loop_salad_checkpoint=Path("models/loop/dino_salad.ckpt"),
        loop_dino_checkpoint=Path("models/loop/dinov2_vitb14_pretrain.pth"), loop_output_dir=out / "loop")
    corrected = refine_trajectory([f.path for f in frames], torch.from_numpy(poses), backend.wrapper, config).cpu().numpy()
    np.save(out / "poses-loop.npy", corrected)
    write_trajectory(out / "trajectory-loop.txt", [f.timestamp for f in frames], corrected)
    voxel = args.voxel or json.loads((out / "report.json").read_text())["final_voxel_model_units"]
    cloud = VoxelMap(voxel, args.max_points)
    for i, pose in enumerate(corrected):
        with np.load(out / "frames" / f"{i:06d}.npz") as f:
            rows = slice(*f['valid_rows']) if 'valid_rows' in f else slice(None)
            p = f["points"][rows][::args.point_stride, ::args.point_stride].reshape(-1,3)
            c = f["colors"][rows][::args.point_stride, ::args.point_stride].reshape(-1,3)
            conf = f["confidence"][rows][::args.point_stride, ::args.point_stride].reshape(-1)
        valid = np.isfinite(p).all(1) & (p[:,2] > 0) & (conf >= args.confidence)
        cloud.add(transform_points(p[valid], pose), c[valid])
    xyz, rgb = cloud.arrays()
    write_ply(out / "map-loop.ply", xyz, rgb)
    np.savez(out / "map-loop.npz", points=xyz, colors=rgb)
    return corrected, (xyz, rgb)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("dataset"); p.add_argument("--suffix", default="")
    a = p.parse_args(); evaluate(a.run, a.dataset, a.suffix)
