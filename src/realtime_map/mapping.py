# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
from pathlib import Path
import json
import numpy as np


def transform_points(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def filtered_points(result, threshold=.5, stride=3):
    points = result.local_points[result.valid_rows][::stride, ::stride].reshape(-1, 3)
    colors = result.colors[result.valid_rows][::stride, ::stride].reshape(-1, 3)
    conf = result.confidence[result.valid_rows][::stride, ::stride].reshape(-1)
    mask = np.isfinite(points).all(1) & np.isfinite(conf) & (points[:, 2] > 0) & (conf >= threshold)
    return transform_points(points[mask], result.camera_pose), colors[mask]


class VoxelMap:
    """Bounded preview, progressively coarsened rather than evicting old rooms.

    Voxel size is in uncalibrated model units. Each cell retains one sample;
    full dense observations are separately archived by the runner.
    """
    def __init__(self, voxel=.04, max_points=1_000_000):
        if voxel <= 0 or max_points <= 0:
            raise ValueError("voxel and max_points must be positive")
        self.voxel, self.max_points = voxel, max_points
        self.cells = set()
        self.points = np.empty((max_points, 3), np.float32)
        self.colors = np.empty((max_points, 3), np.uint8)
        self.rebuilds = 0

    def _keys(self, xyz):
        integer = np.ascontiguousarray(np.floor(xyz / self.voxel), dtype=np.int64)
        # Fixed-width bytes preserve all three signed coordinates without hash collisions.
        return integer.view("V24").reshape(-1)

    def add(self, xyz, rgb):
        if not len(xyz): return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8), False
        keys, first = np.unique(self._keys(xyz), return_index=True)
        byte_keys = keys.tolist()
        unseen = np.fromiter((k not in self.cells for k in byte_keys), bool, len(keys))
        selected = first[unseen]
        added_xyz, added_rgb = xyz[selected], rgb[selected]
        n = len(self.cells)
        if n + len(selected) <= self.max_points:
            self.points[n:n+len(selected)] = added_xyz
            self.colors[n:n+len(selected)] = added_rgb
            self.cells.update(k for k, yes in zip(byte_keys, unseen) if yes)
            return added_xyz, added_rgb, False
        points = np.concatenate([self.points[:n], added_xyz])
        colors = np.concatenate([self.colors[:n], added_rgb])
        while len(points) > self.max_points:
            self.voxel *= 2
            keys, first = np.unique(self._keys(points), return_index=True)
            points, colors = points[first], colors[first]
            self.rebuilds += 1
        self.cells = set(keys.tolist())
        self.points[:len(points)] = points
        self.colors[:len(points)] = colors
        return added_xyz, added_rgb, True

    def arrays(self):
        n = len(self.cells)
        return self.points[:n].copy(), self.colors[:n].copy()


class MapChunks:
    """Archive local portions at independent resolution with bounded RAM.

    A long route cannot repeatedly coarsen earlier chunks. Each chunk retains
    its own voxel size and frame interval; the global map is only an overview.
    """
    def __init__(self, directory, voxel, max_points, frames_per_chunk):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.voxel, self.max_points = voxel, max_points
        self.frames_per_chunk = frames_per_chunk
        self.current = VoxelMap(voxel, max_points)
        self.entries = []; self.first = None; self.last = None

    def add(self, frame_id, xyz, rgb):
        boundary = self.first is not None and frame_id - self.first >= self.frames_per_chunk
        if boundary:
            self.flush()
            self.current = VoxelMap(self.voxel, self.max_points)
        if self.first is None: self.first = frame_id
        self.last = frame_id
        added_xyz, added_rgb, rebuilt = self.current.add(xyz, rgb)
        return added_xyz, added_rgb, rebuilt or boundary

    def flush(self):
        if self.first is None: return
        xyz, rgb = self.current.arrays()
        name = f"{len(self.entries):05d}.npz"
        np.savez(self.directory / name, points=xyz, colors=rgb)
        self.entries.append({"file": name, "first_frame": self.first,
                             "last_frame": self.last, "points": len(xyz),
                             "voxel_model_units": self.current.voxel})
        temporary = self.directory / "manifest.json.partial"
        temporary.write_text(json.dumps({"scale": "uncalibrated", "chunks": self.entries}, indent=2))
        temporary.replace(self.directory / "manifest.json")
        self.first = self.last = None


def write_ply(path, xyz, rgb):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, name in enumerate(["x", "y", "z"]): data[name] = xyz[:, i]
    for i, name in enumerate(["red", "green", "blue"]): data[name] = rgb[:, i]
    with path.open("wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\ncomment scale uncalibrated\n"
                 f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        data.tofile(f)


def write_trajectory(path, times, poses):
    from scipy.spatial.transform import Rotation
    if not len(poses):
        Path(path).write_text("# timestamp tx ty tz qx qy qz qw; scale uncalibrated\n")
        return
    poses = np.asarray(poses)
    quaternion = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    np.savetxt(path, np.column_stack([times, poses[:, :3, 3], quaternion]), fmt="%.9f",
               header="timestamp tx ty tz qx qy qz qw; scale uncalibrated")
