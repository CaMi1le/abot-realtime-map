# SPDX-License-Identifier: Apache-2.0
"""Offline training entrypoints. Labels are used here, never by OccRuntime.

Two stages: train the Waymo camera OCC branch, then freeze OCC and ABot and
train only the zero-initialized token adapter through causal SDPA forwards.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .occ_fusion import EncoderConditioner, OccTokenAdapter
from .occ_model import CameraOccupancyNet, OccRuntime, PredictedOccSource, project_ego
from .occupancy import GridSpec, lidar_supervision
from .waymo import CAMERAS, camera_batch, read_waymo


def context_split(roots, validation_contexts):
    train, val = [], []
    seen = set()
    for root in roots:
        root = Path(root)
        if not (root / "source-manifest.json").exists():
            root = root / "scene"
        manifest = json.loads((root / "source-manifest.json").read_text())
        rows = {
            (r["context"], int(r["timestamp_micros"])): r for r in manifest["frames"]
        }
        for context in sorted({r["context"] for r in manifest["frames"]}):
            if context in seen:
                raise ValueError("An official context occurs in multiple input roots")
            seen.add(context)
            group = [
                (f, root, rows[(f.context, f.timestamp_us)])
                for f in read_waymo(root, context=context)
            ]
            (val if context in validation_contexts else train).extend(group)
    if not train or not val or not set(validation_contexts) <= seen:
        raise ValueError(
            "Training and validation require nonempty, disjoint official contexts"
        )
    return train, val


def lidar_target(root, row, grid):
    cloud = np.fromfile(root / row["lidar"]["file"], np.float32).reshape(-1, 14)
    return lidar_supervision(cloud[:, :3], cloud[:, 3:6], grid)


def checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def train_occupancy(args, train, val):
    grid = GridSpec()
    model = CameraOccupancyNet(grid).to(args.device)
    cameras = (CAMERAS[0],) if args.cameras == 1 else CAMERAS
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Cache only discrete supervision, on CPU. Raw LiDAR never enters prediction.
    targets = {
        (f.context, f.timestamp_us): lidar_target(root, row, grid)
        for f, root, row in train + val
    }
    completed = 0
    for step in range(args.steps):
        f, root, row = train[step % len(train)]
        batch = camera_batch(f, cameras, (192, 320), args.device)
        logits, support = model(*batch)
        target = torch.as_tensor(
            targets[(f.context, f.timestamp_us)], device=args.device
        )
        mask = support[0] & (target >= 0)
        if not mask.any():
            raise ValueError("No observed in-FOV OCC supervision in a training frame")
        loss = F.cross_entropy(logits[0][mask], target[mask])
        if not torch.isfinite(loss):
            raise ValueError("Non-finite occupancy loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        completed += 1
        print(
            json.dumps(
                {"stage": "occupancy", "step": completed, "loss": float(loss.detach())}
            ),
            flush=True,
        )
    model.eval()
    intersection = union = observed = 0
    with torch.inference_mode():
        for f, root, row in val:
            logits, support = model(*camera_batch(f, cameras, (192, 320), args.device))
            target = torch.as_tensor(
                targets[(f.context, f.timestamp_us)], device=args.device
            )
            mask = support[0] & (target >= 0)
            pred = logits[0].argmax(-1) == 1
            truth = target == 1
            intersection += int((pred & truth & mask).sum())
            union += int(((pred | truth) & mask).sum())
            observed += int(mask.sum())
    value = {
        "format": "realtime-map-occ-v1",
        "model_config": model.config(),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "cameras": list(cameras),
        "image_size": [192, 320],
        "training": {
            "steps": completed,
            "contexts": sorted({f.context for f, _, _ in train}),
            "validation_contexts": sorted({f.context for f, _, _ in val}),
            "dataset": "waymo",
            "target": "measured ray free/occupied; unobserved ignored",
        },
        "validation": {
            "occupied_iou": intersection / union if union else None,
            "observed_voxels": observed,
        },
    }
    checkpoint(args.output, value)
    return value["validation"]


def depth_samples(root, row, frame, transform, device):
    """Sparse measured surface camera-z, with nearest-return pixel visibility."""
    cloud = np.fromfile(root / row["lidar"]["file"], np.float32).reshape(-1, 14)
    points = torch.as_tensor(cloud[:, 3:6], device=device)
    camera = frame.cameras[0]
    intrinsic = torch.as_tensor(camera.intrinsic, dtype=torch.float32, device=device)[
        None, None
    ]
    distortion = torch.as_tensor(camera.distortion, dtype=torch.float32, device=device)[
        None, None
    ]
    extrinsic = torch.as_tensor(
        camera.ego_from_camera, dtype=torch.float32, device=device
    )[None, None]
    uv, valid = project_ego(points, intrinsic, distortion, extrinsic)
    uv, valid = uv[0, 0], valid[0, 0]
    inverse = torch.linalg.inv(extrinsic[0, 0])
    depth = (points @ inverse[:3, :3].T + inverse[:3, 3])[:, 2]
    sx = transform.target_width / transform.source_width
    sy = transform.resized_height / transform.source_height
    u = ((uv[:, 0] + 0.5) * sx - 0.5).round().long()
    v = (
        ((uv[:, 1] + 0.5) * sy - 0.5 - transform.crop_top + transform.pad_top)
        .round()
        .long()
    )
    good = (
        valid
        & torch.isfinite(depth)
        & (depth > 0.5)
        & (depth < 80)
        & (u >= 0)
        & (u < transform.target_width)
        & (v >= transform.pad_top)
        & (v < transform.target_height - transform.pad_bottom)
    )
    ids = v[good] * transform.target_width + u[good]
    zbuffer = torch.full(
        (transform.target_height * transform.target_width,), torch.inf, device=device
    )
    zbuffer.scatter_reduce_(0, ids, depth[good], reduce="amin", include_self=True)
    ids = torch.nonzero(torch.isfinite(zbuffer)).flatten()
    return ids, zbuffer[ids]


def fusion_frame_loss(
    backend, runtime, conditioner, item, carry=None, ref_hidden=None, camera_state=None
):
    from abot_recon.preprocessing import preprocess_image
    from abot_recon.modeling.streaming.core.cache_utils import detach_carry

    frame, root, row = item
    with Image.open(frame.path) as image:
        rgb, transform = preprocess_image(image.convert("RGB"))
    hints, _ = runtime.prepare(frame, transform)
    tensor = rgb[None, None].to(backend.device, dtype=backend.wrapper.compute_dtype)
    with (
        conditioner.condition(hints),
        torch.autocast(
            "cuda" if backend.device.startswith("cuda") else "cpu",
            dtype=backend.wrapper.compute_dtype,
            enabled=backend.device.startswith("cuda"),
        ),
    ):
        pred = backend.network(
            tensor,
            causal_global_attn=True,
            long_sequence_parallel=True,
            streaming_inference=True,
            past_key_values=carry,
            ref_hidden=ref_hidden,
            camera_state=camera_state,
        )
    ids, target = depth_samples(root, row, frame, transform, backend.device)
    if len(ids) < 16:
        raise ValueError("Insufficient visible LiDAR depth for fusion training")
    depth = pred["local_points"][0, 0, ..., 2].float().reshape(-1)[ids].clamp_min(1e-5)
    residual = depth.log() - target.log()
    # Preserve the monocular scale gauge; no GT-fitted scale is used at inference.
    loss = ((residual - residual.mean()) ** 2).mean()
    next_carry = backend.network._postprocess_stream_carry(
        detach_carry(pred.get("past_key_values")), spatial_hw=tuple(tensor.shape[-2:])
    )

    def detached(value):
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, dict):
            return {k: detached(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(detached(v) for v in value)
        return value

    next_ref_hidden = pred.get("ref_hidden")
    if next_ref_hidden is None:
        next_ref_hidden = ref_hidden
    return (
        loss,
        next_carry,
        detached(next_ref_hidden),
        detached(pred.get("camera_state", camera_state)),
    )


def train_fusion(args, train, val):
    from .backend import ABotStream

    backend = ABotStream(args.abot_checkpoint, "sdpa", device=args.device)
    backend.network.eval().requires_grad_(False)
    source = PredictedOccSource(args.occ_checkpoint, args.device)
    validation_contexts = {f.context for f, _, _ in val}
    if validation_contexts & set(source.metadata["training"].get("contexts", [])):
        raise ValueError(
            "Fusion validation contexts were used to train the OCC predictor"
        )
    if len(source.cameras) != args.cameras:
        raise ValueError("--cameras must match the OCC checkpoint")
    token_dim = backend.network.encoder.embed_dim
    adapter = OccTokenAdapter(token_dim).to(args.device)
    conditioner = EncoderConditioner(backend.network.encoder, adapter)
    runtime = OccRuntime(source, adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    previous = None
    carry = ref_hidden = camera_state = None
    try:
        for step in range(args.steps):
            item = train[step % len(train)]
            frame = item[0]
            if (
                previous is None
                or previous.context != frame.context
                or frame.timestamp <= previous.timestamp
            ):
                backend.reset()
                carry = ref_hidden = camera_state = None
            loss, carry, ref_hidden, camera_state = fusion_frame_loss(
                backend, runtime, conditioner, item,
                carry=carry,
                ref_hidden=ref_hidden,
                camera_state=camera_state,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            previous = frame
            print(
                json.dumps(
                    {
                        "stage": "fusion",
                        "step": step + 1,
                        "scale_invariant_log_depth_loss": float(loss.detach()),
                    }
                ),
                flush=True,
            )
        adapter.eval()
        scores = []
        previous, carry, ref_hidden, camera_state = None, None, None, None
        with torch.no_grad():
            for item in val:
                frame = item[0]
                if (
                    previous is None
                    or previous.context != frame.context
                    or frame.timestamp <= previous.timestamp
                ):
                    backend.reset()
                    carry = ref_hidden = camera_state = None
                loss, carry, ref_hidden, camera_state = fusion_frame_loss(
                    backend, runtime, conditioner, item,
                    carry=carry,
                    ref_hidden=ref_hidden,
                    camera_state=camera_state,
                )
                scores.append(float(loss))
                previous = frame
        digest = hashlib.sha256(Path(args.occ_checkpoint).read_bytes()).hexdigest()
        value = {
            "format": "realtime-map-occ-adapter-v1",
            "adapter_config": {
                "token_dim": token_dim,
                "hidden_dim": adapter.hidden_dim,
            },
            "state_dict": {k: v.cpu() for k, v in adapter.state_dict().items()},
            "cameras": list(source.cameras),
            "occ_checkpoint_sha256": digest,
            "training": {
                "steps": args.steps,
                "contexts": sorted({f.context for f, _, _ in train}),
                "validation_contexts": sorted({f.context for f, _, _ in val}),
                "frozen": ["ABot", "OCC"],
                "loss": "scale-invariant observed log camera-z depth",
            },
            "validation": {"scale_invariant_log_depth_loss": float(np.mean(scores))},
        }
        checkpoint(args.output, value)
        return value["validation"]
    finally:
        conditioner.close()
        backend.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["occupancy", "fusion"], required=True)
    p.add_argument(
        "--input", nargs="+", required=True, help="Prepared Waymo scene roots"
    )
    p.add_argument("--validation-context", action="append", required=True)
    p.add_argument("--cameras", type=int, choices=[1, 3], default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--occ-checkpoint")
    p.add_argument("--abot-checkpoint", default="models/abot_recon.safetensors")
    args = p.parse_args()
    if args.steps <= 0 or args.lr <= 0:
        p.error("steps and learning rate must be positive")
    if args.stage == "fusion" and not args.occ_checkpoint:
        p.error("Fusion training requires a trained OCC checkpoint")
    train, val = context_split(args.input, set(args.validation_context))
    report = (train_occupancy if args.stage == "occupancy" else train_fusion)(
        args, train, val
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
