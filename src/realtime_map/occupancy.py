# SPDX-License-Identifier: Apache-2.0
"""External voxel predictions and calibrated, bounded-cost OCC-to-patch projection."""

from dataclasses import dataclass
import math

import numpy as np
import torch

from .waymo import patch_rays


@dataclass(frozen=True)
class GridSpec:
    minimum: tuple = (-10.0, -10.0, -0.5)
    voxel: tuple = (0.2, 0.2, 0.2)
    shape: tuple = (100, 100, 10)

    def __post_init__(self):
        for name in ("minimum", "voxel", "shape"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            len(self.minimum) != 3
            or len(self.voxel) != 3
            or len(self.shape) != 3
            or not np.isfinite(self.minimum).all()
            or not np.isfinite(self.voxel).all()
            or any(v <= 0 for v in self.voxel)
            or any(int(n) != n or n <= 0 for n in self.shape)
        ):
            raise ValueError("Invalid occupancy grid")
        if math.prod(self.shape) > 2_000_000:
            raise ValueError("Occupancy grid exceeds the bounded experiment budget")

    @property
    def maximum(self):
        return np.asarray(self.minimum) + np.asarray(self.voxel) * self.shape

    def centres(self, device, feature_layers=None):
        shape = (
            self.shape if feature_layers is None else (*self.shape[:2], feature_layers)
        )
        lo = torch.tensor(self.minimum, device=device)
        step = torch.tensor(
            (self.maximum - self.minimum) / np.asarray(shape),
            dtype=torch.float32,
            device=device,
        )
        ijk = torch.stack(
            torch.meshgrid(
                *(torch.arange(n, device=device) for n in shape), indexing="ij"
            ),
            -1,
        )
        return (ijk + 0.5) * step + lo


@dataclass
class VoxelPrediction:
    occupied_probability: torch.Tensor  # [X,Y,Z], external prediction, not GT
    valid: torch.Tensor  # spatial support; outside this mask remains unknown
    grid: GridSpec
    timestamp: float
    cameras: tuple[str, ...]

    def validate(self, timestamp):
        p = self.occupied_probability
        if (
            p.shape != self.grid.shape
            or self.valid.shape != p.shape
            or self.valid.dtype != torch.bool
        ):
            raise ValueError(
                "OCC probability/support shape or mask dtype disagrees with grid"
            )
        if not math.isfinite(self.timestamp) or abs(self.timestamp - timestamp) > 0.001:
            raise ValueError(
                "OCC prior must belong to the current frame, never a future/stale frame"
            )
        if not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("OCC probabilities must be finite and in [0,1]")


def prediction_from_logits(logits, valid, grid, timestamp, cameras, free_class=0):
    """Normalize an external OCC head, including parking free/occupied/wall.

    Its non-free classes are combined as occupied. Coordinate system, visible
    support and sensor calibration must already match the current ego frame;
    this utility does not convert fisheye weights to a Waymo model.
    """
    if logits.ndim == 5 and logits.shape[0] == 1:
        logits = logits[0]
    if (
        logits.ndim != 4
        or tuple(logits.shape[:3]) != grid.shape
        or not 0 <= free_class < logits.shape[-1]
    ):
        raise ValueError("Expected external OCC logits [X,Y,Z,C] or [1,X,Y,Z,C]")
    prediction = VoxelPrediction(
        1 - logits.float().softmax(-1)[..., free_class],
        valid,
        grid,
        timestamp,
        tuple(cameras),
    )
    prediction.validate(timestamp)
    return prediction


class VoxelProjector:
    """Ray sample at <= half a voxel; output four features per image patch.

    Features: log(1+first-hit camera-z metres), first-hit occupancy probability,
    mean visible free probability, known fraction. No hit is distinct from unknown.
    Cache holds only one camera geometry; dense historical priors are never retained.
    """

    def __init__(self, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("OCC threshold must be in (0,1)")
        self.threshold = threshold
        self._key = self._rays = None

    def __call__(self, prediction, camera, transform, timestamp):
        prediction.validate(timestamp)
        device = prediction.occupied_probability.device
        key = (
            transform,
            camera.intrinsic.tobytes(),
            camera.distortion.tobytes(),
            camera.ego_from_camera.tobytes(),
            str(device),
        )
        if key != self._key:
            rays, valid = patch_rays(camera, transform)
            self._rays = (
                torch.as_tensor(rays, device=device),
                torch.as_tensor(valid, device=device),
            )
            self._key = key
        rays, valid_patch = self._rays
        pose = torch.as_tensor(
            camera.ego_from_camera, dtype=torch.float32, device=device
        )
        direction = rays @ pose[:3, :3].T
        origin = pose[:3, 3]
        grid = prediction.grid
        lo = torch.tensor(grid.minimum, dtype=torch.float32, device=device)
        hi = torch.tensor(grid.maximum, dtype=torch.float32, device=device)
        parallel = direction.abs() < 1e-8
        inverse = torch.where(
            parallel, torch.ones_like(direction), direction
        ).reciprocal()
        t0, t1 = (lo - origin) * inverse, (hi - origin) * inverse
        near_axis, far_axis = torch.minimum(t0, t1), torch.maximum(t0, t1)
        near_axis = torch.where(parallel, -torch.inf, near_axis)
        far_axis = torch.where(parallel, torch.inf, far_axis)
        missed_parallel = (parallel & ((origin < lo) | (origin >= hi))).any(-1)
        near = near_axis.max(-1).values.clamp_min(0)
        far = far_axis.min(-1).values
        ray_valid = valid_patch & ~missed_parallel & (far > near)
        step = min(grid.voxel) / 2
        count = math.ceil(float(np.linalg.norm(grid.maximum - grid.minimum)) / step) + 1
        if count > 4096:
            raise ValueError("OCC ray budget exceeds 4096 samples per patch")
        distance = near[:, None] + (torch.arange(count, device=device) + 0.5) * step
        xyz = origin + distance[..., None] * direction[:, None]
        index = torch.floor((xyz - lo) / torch.tensor(grid.voxel, device=device)).long()
        shape = torch.tensor(grid.shape, device=device)
        inside = (
            ((index >= 0) & (index < shape)).all(-1)
            & (distance < far[:, None])
            & ray_valid[:, None]
        )
        index = torch.minimum(torch.maximum(index, torch.zeros_like(index)), shape - 1)
        ix, iy, iz = index.unbind(-1)
        known = prediction.valid[ix, iy, iz] & inside
        prob = prediction.occupied_probability[ix, iy, iz]
        hit = known & (prob >= self.threshold)
        has_hit = hit.any(-1)
        first = hit.int().argmax(-1)
        before = (
            torch.arange(count, device=device)[None]
            <= torch.where(has_hit, first, count - 1)[:, None]
        )
        visible = before & inside
        known = known & before
        weight = known.sum(-1).clamp_min(1)
        row = torch.arange(len(rays), device=device)
        depth = torch.where(has_hit, distance[row, first] * rays[:, 2], 0)
        hit_prob = torch.where(has_hit, prob[row, first], 0)
        free = ((1 - prob) * known).sum(-1) / weight
        fraction = known.sum(-1) / visible.sum(-1).clamp_min(1)
        return torch.stack([depth.log1p(), hit_prob, free, fraction], -1).float()[None]


def lidar_supervision(origins, returns, grid):
    """Observed ray free space + return endpoints only; unknown=-1, hits win.

    Offline training utility. Points beyond the ROI still carve their traversed
    in-ROI free cells; nothing behind a measured return is labelled free.
    """
    origins, returns = np.asarray(origins), np.asarray(returns)
    if origins.shape != returns.shape or origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("LiDAR origins and endpoints must be matching Nx3 arrays")
    labels = np.full(grid.shape, -1, np.int64)
    lo, size = np.asarray(grid.minimum), np.asarray(grid.voxel)
    shape = np.asarray(grid.shape)
    good = np.isfinite(origins).all(1) & np.isfinite(returns).all(1)
    origins, returns = origins[good], returns[good]
    # Clip measured rays to the ROI before sampling. Temporary RAM and sample
    # count depend on the grid diagonal, not the sensor's maximum range.
    step = size.min() / 2
    offsets = (
        np.arange(math.ceil(np.linalg.norm(grid.maximum - lo) / step) + 1) + 0.5
    ) * step
    for start in range(0, len(returns), 512):
        a, b = origins[start : start + 512], returns[start : start + 512]
        length = np.linalg.norm(b - a, axis=1)
        direction = (b - a) / np.maximum(length[:, None], 1e-12)
        parallel = np.abs(direction) < 1e-8
        inverse = 1 / np.where(parallel, 1.0, direction)
        t0, t1 = (lo - a) * inverse, (grid.maximum - a) * inverse
        near = np.where(parallel, -np.inf, np.minimum(t0, t1)).max(1).clip(0)
        far = np.minimum(np.where(parallel, np.inf, np.maximum(t0, t1)).min(1), length)
        valid_ray = (far > near) & ~(parallel & ((a < lo) | (a >= grid.maximum))).any(1)
        near = np.where(valid_ray, near, 0)
        distance = near[:, None] + offsets
        xyz = a[:, None] + direction[:, None] * distance[..., None]
        ids = np.floor((xyz - lo) / size).astype(int)
        end = np.floor((b - lo) / size).astype(int)
        valid = (
            ((ids >= 0) & (ids < shape)).all(-1)
            & valid_ray[:, None]
            & (distance < far[:, None])
            & (ids != end[:, None]).any(-1)
        )
        labels[tuple(ids[valid].T)] = 0
    end = np.floor((returns - lo) / size).astype(int)
    valid = ((end >= 0) & (end < shape)).all(1)
    labels[tuple(end[valid].T)] = 1
    return labels
