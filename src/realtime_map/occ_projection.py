# SPDX-License-Identifier: Apache-2.0
"""Differentiable first-hit projection of occupancy into ABot patch hints."""

import numpy as np
import torch

from .waymo import patch_rays


class SoftVoxelProjector:
    """Project probabilities with exact voxel traversal and soft transmittance.

    Features are log1p(conditional expected first-hit camera-z metres), first-hit
    probability mass, visibility-weighted mean known free probability, and
    visibility-weighted known fraction. Unknown cells have zero opacity; an
    entirely unknown or out-of-grid ray has four zero features. A known empty
    ray instead has features [0, 0, 1, 1].

    Geometry is fixed calibration, not a differentiable camera-pose estimate.
    Each intersected voxel contributes once, irrespective of its ray length.
    Only one geometry is cached, and it never retains prediction tensors or
    training graphs. Budgets bound both cache and temporary tensor sizes.
    """

    def __init__(
        self, max_rays=4096, max_samples_per_ray=4096, max_ray_voxels=2_000_000
    ):
        limits = (max_rays, max_samples_per_ray, max_ray_voxels)
        if any(int(n) != n or n <= 0 for n in limits):
            raise ValueError("Projection budgets must be positive integers")
        self.max_rays, self.max_samples_per_ray, self.max_ray_voxels = limits
        self._key = self._geometry = None

    def config(self):
        """Serializable geometry/feature contract for training and inference."""
        return {
            "kind": "soft",
            "version": 1,
            "max_rays": self.max_rays,
            "max_samples_per_ray": self.max_samples_per_ray,
            "max_ray_voxels": self.max_ray_voxels,
        }

    def _traversal(self, grid, camera, transform, device):
        count = (transform.target_height // 14) * (transform.target_width // 14)
        # Internal axis planes plus the entry and exit points. Sorting them
        # divides each ray into intervals contained in exactly one voxel.
        samples = sum(grid.shape) - 2
        if (
            count > self.max_rays
            or samples > self.max_samples_per_ray
            or count * (samples + 1) > self.max_ray_voxels
        ):
            raise ValueError("OCC soft projection exceeds the ray/voxel budget")
        if count == 0:
            raise ValueError("Projection requires at least one image patch")
        rays, valid_patch = patch_rays(camera, transform)
        rays = rays.astype(np.float64)
        pose = np.asarray(camera.ego_from_camera, dtype=np.float64)
        direction, origin = rays @ pose[:3, :3].T, pose[:3, 3]
        lo, hi = np.asarray(grid.minimum), grid.maximum
        size, shape = np.asarray(grid.voxel), np.asarray(grid.shape)
        parallel = np.abs(direction) < 1e-12
        inverse = 1 / np.where(parallel, 1.0, direction)
        t0, t1 = (lo - origin) * inverse, (hi - origin) * inverse
        near = np.maximum(np.where(parallel, -np.inf, np.minimum(t0, t1)).max(-1), 0)
        far = np.where(parallel, np.inf, np.maximum(t0, t1)).min(-1)
        missed = (parallel & ((origin < lo) | (origin >= hi))).any(-1)
        good_ray = valid_patch & ~missed & (far > near)
        near, far = np.where(good_ray, near, 0), np.where(good_ray, far, 0)
        crossings = [near[:, None], far[:, None]]
        for axis, n in enumerate(shape):
            planes = lo[axis] + np.arange(1, n) * size[axis]
            distance = (planes[None] - origin[axis]) * inverse[:, axis, None]
            distance = np.where(parallel[:, axis, None], near[:, None], distance)
            crossings.append(np.clip(distance, near[:, None], far[:, None]))
        edges = np.sort(np.concatenate(crossings, axis=1), axis=1)
        lengths = np.diff(edges, axis=1)
        distance = edges[:, :-1] + lengths / 2
        xyz = origin + distance[..., None] * direction[:, None]
        index = np.floor((xyz - lo) / size).astype(np.int64)
        inside = (
            good_ray[:, None]
            & (lengths > 1e-10 * np.maximum(1, far[:, None]))
            & ((index >= 0) & (index < shape)).all(-1)
        )
        # Empty intervals occur at coincident axis crossings and outside the
        # intersection. Safe indices make gathering valid for all padded rows.
        index = np.clip(index, 0, shape - 1)
        flat = (index[..., 0] * shape[1] + index[..., 1]) * shape[2] + index[..., 2]
        camera_z = distance * rays[:, 2, None]
        return (
            torch.as_tensor(flat, dtype=torch.long, device=device),
            torch.as_tensor(camera_z, dtype=torch.float32, device=device),
            torch.as_tensor(inside, dtype=torch.bool, device=device),
        )

    def __call__(self, prediction, camera, transform, timestamp):
        prediction.validate(timestamp)
        device = prediction.occupied_probability.device
        if prediction.valid.device != device:
            raise ValueError("OCC probability and support must share a device")
        key = (
            prediction.grid,
            transform,
            camera.width,
            camera.height,
            camera.intrinsic.tobytes(),
            camera.distortion.tobytes(),
            camera.ego_from_camera.tobytes(),
            str(device),
        )
        if key != self._key:
            self._geometry = self._traversal(prediction.grid, camera, transform, device)
            self._key = key
        index, camera_z, inside = self._geometry
        # Accumulate in float32 even when the OCC encoder uses mixed precision.
        probability = prediction.occupied_probability.float().reshape(-1)[index]
        known = prediction.valid.reshape(-1)[index] & inside
        alpha = probability * known
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha], dim=-1), dim=-1
        )[:, :-1]
        hit = transmittance * alpha
        mass = hit.sum(-1)
        has_hit = mass > 0
        safe_mass = torch.where(has_hit, mass, torch.ones_like(mass))
        depth = torch.where(has_hit, (hit * camera_z).sum(-1) / safe_mass, 0)
        known_weight = transmittance * known
        known_mass = known_weight.sum(-1)
        # The first known/inside cell has transmittance 1, so a nonempty
        # denominator here is at least 1. Clamping safely handles unknown rays.
        free = (known_weight * (1 - probability)).sum(-1) / known_mass.clamp_min(1)
        fraction = known_mass / (transmittance * inside).sum(-1).clamp_min(1)
        return torch.stack([depth.log1p(), mass, free, fraction], dim=-1)[None]
