# SPDX-License-Identifier: Apache-2.0
"""Small pinhole multi-camera OCC branch, inspired by the parking BEV pipeline.

This is a new trainable baseline, not a converted MagModel checkpoint. Both one
and three-camera variants must be trained on disjoint Waymo contexts before use.
"""

from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F

from .occupancy import GridSpec, VoxelPrediction, VoxelProjector
from .occ_fusion import OccTokenAdapter
from .waymo import CAMERAS, camera_batch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def project_ego(points, intrinsic, distortion, ego_from_camera):
    """Broadcast [P,3] to [B,V,P,2] with Waymo's Brown-Conrady distortion."""
    inverse = torch.linalg.inv(ego_from_camera.float())
    camera = torch.einsum(
        "bvij,pj->bvpi", inverse[..., :3, :3], points.float()
    ) + inverse[..., :3, 3].unsqueeze(-2)
    z = camera[..., 2]
    x, y = (camera[..., :2] / z.clamp_min(1e-6)[..., None]).unbind(-1)
    k1, k2, p1, p2, k3 = distortion.float().unbind(-1)
    k1, k2, p1, p2, k3 = [v[..., None] for v in (k1, k2, p1, p2, k3)]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = intrinsic[..., 0, 0, None] * xd + intrinsic[..., 0, 2, None]
    v = intrinsic[..., 1, 1, None] * yd + intrinsic[..., 1, 2, None]
    return torch.stack((u, v), -1), z > 1e-4


class CameraOccupancyNet(nn.Module):
    def __init__(self, grid=None, channels=32, feature_layers=4):
        super().__init__()
        self.grid = grid or GridSpec()
        self.channels, self.feature_layers = channels, feature_layers
        layers = []
        input_channels = 3
        for _ in range(3):
            layers.extend(
                [
                    nn.Conv2d(input_channels, channels, 3, stride=2, padding=1),
                    nn.GroupNorm(4, channels),
                    nn.ReLU(),
                ]
            )
            input_channels = channels
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(channels * feature_layers, channels * 2, 3, padding=1),
            nn.GroupNorm(4, channels * 2),
            nn.ReLU(),
            nn.Conv2d(channels * 2, self.grid.shape[2] * 2, 1),
        )
        self.register_buffer(
            "feature_centres",
            self.grid.centres("cpu", feature_layers).reshape(-1, 3).float(),
            persistent=False,
        )
        self.register_buffer(
            "output_centres",
            self.grid.centres("cpu").reshape(-1, 3).float(),
            persistent=False,
        )

    def forward(self, images, intrinsic, distortion, ego_from_camera):
        b, v, _, height, width = images.shape
        features = self.backbone((images.reshape(b * v, 3, height, width) - 0.5) / 0.5)
        uv, valid = project_ego(
            self.feature_centres, intrinsic, distortion, ego_from_camera
        )
        valid &= (
            (uv[..., 0] >= 0)
            & (uv[..., 0] < width)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] < height)
        )
        # align_corners=False uses image pixel-centre coordinates at every scale.
        grid = 2 * (uv + 0.5) / uv.new_tensor([width, height]) - 1
        grid = torch.where(valid[..., None], grid, 2.0).reshape(b * v, 1, -1, 2)
        sampled = F.grid_sample(
            features, grid.to(features.dtype), align_corners=False
        ).squeeze(2)
        sampled = sampled.reshape(b, v, self.channels, -1) * valid[:, :, None]
        pooled = sampled.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        nx, ny, nz = self.grid.shape
        volume = pooled.reshape(b, self.channels, nx, ny, self.feature_layers)
        bev = volume.permute(0, 1, 4, 2, 3).reshape(b, -1, nx, ny)
        logits = self.head(bev).reshape(b, nz, 2, nx, ny).permute(0, 3, 4, 1, 2)
        uv, support = project_ego(
            self.output_centres, intrinsic, distortion, ego_from_camera
        )
        support &= (
            (uv[..., 0] >= 0)
            & (uv[..., 0] < width)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] < height)
        )
        return logits, support.any(1).reshape(b, nx, ny, nz)

    def config(self):
        return {
            "grid": asdict(self.grid),
            "channels": self.channels,
            "feature_layers": self.feature_layers,
        }


def load_occ_checkpoint(path, device="cpu"):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        value.get("format") != "realtime-map-occ-v1"
        or value.get("training", {}).get("steps", 0) <= 0
    ):
        raise ValueError(
            "Expected a trained Waymo OCC checkpoint, not parking weights or an untrained model"
        )
    config = dict(value["model_config"])
    config["grid"] = GridSpec(**config["grid"])
    model = CameraOccupancyNet(**config)
    model.load_state_dict(value["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    cameras = tuple(value["cameras"])
    if cameras not in ((CAMERAS[0],), CAMERAS):
        raise ValueError(
            "Checkpoint cameras must be FRONT or FRONT/FRONT_LEFT/FRONT_RIGHT"
        )
    return model, cameras, value


class PredictedOccSource:
    def __init__(self, checkpoint, device="cpu", cameras=None):
        self.model, self.cameras, self.metadata = load_occ_checkpoint(
            checkpoint, device
        )
        self.device = device
        if cameras is not None and tuple(cameras) != self.cameras:
            raise ValueError(
                "Inference camera set must match the trained OCC checkpoint"
            )
        self.image_size = tuple(self.metadata["image_size"])

    @torch.inference_mode()
    def predict(self, frame):
        batch = camera_batch(frame, self.cameras, self.image_size, self.device)
        logits, valid = self.model(*batch)
        return VoxelPrediction(
            logits.float().softmax(-1)[0, ..., 1],
            valid[0],
            self.model.grid,
            frame.timestamp,
            self.cameras,
        )


class OccRuntime:
    def __init__(self, source, adapter, mode="predicted", seed=42, projector=None):
        if mode not in ("predicted", "zero", "shuffle"):
            raise ValueError("Unknown OCC ablation")
        self.source, self.adapter, self.mode, self.seed = source, adapter, mode, seed
        self.projector = projector or VoxelProjector()
        self.abot_checkpoint_sha256 = None
        self._permutation = None

    @classmethod
    def from_checkpoints(
        cls, occ_checkpoint, adapter_checkpoint, device, mode="predicted"
    ):
        source = PredictedOccSource(occ_checkpoint, device)
        state = torch.load(adapter_checkpoint, map_location="cpu", weights_only=True)
        if (
            state.get("format")
            not in ("realtime-map-occ-adapter-v1", "realtime-map-occ-adapter-v2")
            or state.get("training", {}).get("steps", 0) <= 0
        ):
            raise ValueError("Expected a trained OCC adapter checkpoint")
        if tuple(state.get("cameras", ())) != source.cameras:
            raise ValueError("Adapter/OCC checkpoint camera sets differ")
        expected = state.get("occ_checkpoint_sha256")
        if expected != file_sha256(occ_checkpoint):
            raise ValueError("Adapter was trained against a different OCC checkpoint")
        projector = None
        if state["format"] == "realtime-map-occ-adapter-v2":
            from .occ_projection import SoftVoxelProjector

            projection = dict(state.get("projection", {}))
            if (
                projection.pop("kind", None) != "soft"
                or projection.pop("version", None) != 1
            ):
                raise ValueError("Unsupported joint OCC projection configuration")
            projector = SoftVoxelProjector(**projection)
            if not state.get("abot_checkpoint_sha256"):
                raise ValueError(
                    "Joint adapter is missing its frozen ABot checkpoint binding"
                )
        adapter = OccTokenAdapter(**state["adapter_config"])
        adapter.load_state_dict(state["state_dict"], strict=True)
        adapter.to(device).eval()
        runtime = cls(source, adapter, mode, projector=projector)
        runtime.abot_checkpoint_sha256 = state.get("abot_checkpoint_sha256")
        return runtime

    def validate_abot_checkpoint(self, checkpoint):
        if (
            self.abot_checkpoint_sha256
            and file_sha256(checkpoint) != self.abot_checkpoint_sha256
        ):
            raise ValueError(
                "Joint adapter was trained against a different ABot checkpoint"
            )

    def _sync(self):
        if str(self.source.device).startswith("cuda"):
            torch.cuda.synchronize(self.source.device)

    def prepare(self, frame, transform):
        self._sync()
        started = time.perf_counter()
        prior = self.source.predict(frame)
        self._sync()
        predicted = time.perf_counter()
        front = next(c for c in frame.cameras if c.name == "FRONT")
        hints = self.projector(prior, front, transform, frame.timestamp)
        if self.mode == "zero":
            hints = torch.zeros_like(hints)
        elif self.mode == "shuffle":
            # Same-frame spatial shuffle: no future frame or GT leakage.
            if self._permutation is None or len(self._permutation) != hints.shape[1]:
                generator = torch.Generator(device=hints.device).manual_seed(self.seed)
                self._permutation = torch.randperm(
                    hints.shape[1], generator=generator, device=hints.device
                )
            hints = hints[:, self._permutation]
        self._sync()
        finished = time.perf_counter()
        return hints, {
            "occ_predict_ms": (predicted - started) * 1000,
            "occ_project_ms": (finished - predicted) * 1000,
        }
