# SPDX-License-Identifier: Apache-2.0
"""Read the existing Waymo export without exposing labels to inference."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .data import FramePacket


CAMERAS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT")
# Waymo camera axes: forward, left, up. OpenCV: right, down, forward.
WAYMO_FROM_CV = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@dataclass(frozen=True)
class CameraView:
    name: str
    path: Path
    intrinsic: np.ndarray
    distortion: np.ndarray
    ego_from_camera: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class WaymoFrame(FramePacket):
    context: str
    timestamp_us: int
    cameras: tuple[CameraView, ...]


def read_waymo(root, max_frames=None, context=None):
    root = Path(root)
    manifest = root / "source-manifest.json"
    if not manifest.is_file() and (root / "scene/source-manifest.json").is_file():
        root = root / "scene"
        manifest = root / "source-manifest.json"
    rows = json.loads(manifest.read_text())["frames"]
    contexts = sorted({r["context"] for r in rows})
    if context is None:
        if len(contexts) != 1:
            raise ValueError(
                "Select --waymo-context; each official context is an independent stream: "
                + ", ".join(contexts)
            )
        context = contexts[0]
    rows = [r for r in rows if r["context"] == context]
    if max_frames is not None:
        rows = rows[:max_frames]
    if not rows:
        raise ValueError(f"No Waymo frames for context {context}")
    result = []
    previous = None
    for i, row in enumerate(rows):
        stamp = int(row["timestamp_micros"])
        if previous is not None and stamp <= previous:
            raise ValueError("Waymo source timestamps must increase strictly")
        previous = stamp
        cameras = []
        for name in CAMERAS:
            if name not in row["sensors"]:
                continue
            s = row["sensors"][name]
            k = s["image_intrinsic"]
            intrinsic = np.array(
                [[k["f_u"], 0, k["c_u"]], [0, k["f_v"], k["c_v"]], [0, 0, 1.0]]
            )
            distortion = np.array([k[n] for n in ("k1", "k2", "p1", "p2", "k3")])
            # Use fixed sensor calibration, never GT vehicle/camera world poses.
            calibration = root / "extrinsics" / f"{s['camera_id']}.txt"
            if calibration.is_file():
                ego_from_waymo = np.loadtxt(calibration).reshape(4, 4)
            else:
                value = s.get("source_calibration", {}).get(
                    "[CameraCalibrationComponent].extrinsic.transform"
                )
                if value is None:
                    raise ValueError(f"Missing fixed camera extrinsic: {calibration}")
                ego_from_waymo = np.asarray(value).reshape(4, 4)
            path = root / s["image"]
            if not path.is_file():
                raise FileNotFoundError(path)
            cameras.append(
                CameraView(
                    name,
                    path,
                    intrinsic,
                    distortion,
                    ego_from_waymo @ WAYMO_FROM_CV,
                    s["width"],
                    s["height"],
                )
            )
        if not cameras or cameras[0].name != "FRONT":
            raise ValueError("Waymo inference requires FRONT")
        result.append(
            WaymoFrame(i, stamp / 1e6, cameras[0].path, context, stamp, tuple(cameras))
        )
    return result


def camera_batch(frame, camera_names, size=(192, 320), device="cpu"):
    """Images and fixed calibration only; source distortion is retained."""
    import torch

    by_name = {c.name: c for c in frame.cameras}
    images, intrinsics, distortions, extrinsics = [], [], [], []
    height, width = size
    for name in camera_names:
        camera = by_name[name]
        with Image.open(camera.path) as im:
            if im.size != (camera.width, camera.height):
                raise ValueError("Waymo image dimensions disagree with calibration")
            image = (
                np.asarray(im.convert("RGB").resize((width, height)), dtype=np.float32)
                / 255.0
            )
        images.append(image.transpose(2, 0, 1).copy())
        k = camera.intrinsic.copy()
        k[0] *= width / camera.width
        k[1] *= height / camera.height
        # PIL uses pixel-centre resize coordinates.
        k[0, 2] += 0.5 * (width / camera.width - 1)
        k[1, 2] += 0.5 * (height / camera.height - 1)
        intrinsics.append(k)
        distortions.append(camera.distortion)
        extrinsics.append(camera.ego_from_camera)
    return tuple(
        torch.as_tensor(np.asarray(x)[None], dtype=torch.float32, device=device)
        for x in (images, intrinsics, distortions, extrinsics)
    )


def patch_rays(camera, transform, patch_size=14):
    """Invert ABot resize/crop/pad at patch centres, then invert lens distortion."""
    h, w = transform.target_height, transform.target_width
    yy, xx = np.mgrid[: h // patch_size, : w // patch_size]
    u = (xx + 0.5) * patch_size - 0.5
    v = (yy + 0.5) * patch_size - 0.5
    sx = transform.target_width / transform.source_width
    sy = transform.resized_height / transform.source_height
    source_u = (u + 0.5) / sx - 0.5
    source_v = (v + transform.crop_top - transform.pad_top + 0.5) / sy - 0.5
    valid = (
        (source_u >= 0)
        & (source_u < camera.width)
        & (source_v >= 0)
        & (source_v < camera.height)
        & (v >= transform.pad_top)
        & (v < h - transform.pad_bottom)
    )
    pixels = np.stack([source_u, source_v], -1).reshape(-1, 1, 2)
    xy = cv2.undistortPoints(pixels, camera.intrinsic, camera.distortion).reshape(-1, 2)
    rays = np.c_[xy, np.ones(len(xy))]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays.astype(np.float32), valid.reshape(-1)
