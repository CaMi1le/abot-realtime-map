# SPDX-License-Identifier: Apache-2.0
"""Joint image-OCC and token-adapter training through frozen, causal ABot.

Each optimizer step processes one short clip. History is detached after each
frame and reset between clips; checkpoints therefore resume at exact clip
boundaries without serializing historical ABot activations/KV buffers.
"""

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .occ_fusion import EncoderConditioner, OccTokenAdapter
from .occ_model import CameraOccupancyNet, file_sha256, load_occ_checkpoint
from .occ_projection import SoftVoxelProjector
from .occ_training import checkpoint, context_split, depth_samples, lidar_target
from .occupancy import VoxelPrediction
from .waymo import CAMERAS, camera_batch


def detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_tree(v) for v in value)
    return value


@dataclass
class JointState:
    carry: object = None
    ref_hidden: object = None
    camera_state: object = None
    context: str | None = None
    timestamp: float | None = None
    geometry: object = None


class JointFrameRunner:
    def __init__(
        self,
        backend,
        occupancy,
        adapter,
        cameras=CAMERAS,
        image_size=(192, 320),
        projector=None,
    ):
        self.backend, self.occupancy, self.adapter = backend, occupancy, adapter
        self.cameras, self.image_size = tuple(cameras), tuple(image_size)
        if self.cameras not in ((CAMERAS[0],), CAMERAS):
            raise ValueError("Use FRONT or the calibrated front-three camera set")
        backend.network.eval().requires_grad_(False)
        self.conditioner = EncoderConditioner(backend.network.encoder, adapter)
        self.projector = projector or SoftVoxelProjector()
        self.permutation = None

    def forward(
        self, item, state=None, *, occ_target=None, mode="predicted", audit_images=False
    ):
        from abot_recon.preprocessing import preprocess_image
        from abot_recon.modeling.streaming.core.cache_utils import detach_carry

        if mode not in ("predicted", "zero", "shuffle"):
            raise ValueError("Unknown OCC ablation")
        frame, root, row = item
        reset = (
            state is None
            or state.context != frame.context
            or state.timestamp is None
            or frame.timestamp <= state.timestamp
        )
        if reset:
            self.backend.reset()
            state = JointState()
        with Image.open(frame.path) as image:
            rgb, transform = preprocess_image(image.convert("RGB"))
        if state.geometry is not None and state.geometry != transform:
            raise ValueError("ABot geometry changed inside a streaming clip")
        batch = camera_batch(frame, self.cameras, self.image_size, self.backend.device)
        images = batch[0]
        if audit_images:
            images.requires_grad_(True)
        # This path deliberately keeps autograd enabled through OCC and projection.
        logits, support = self.occupancy(*batch)
        prior = VoxelPrediction(
            logits.float().softmax(-1)[0, ..., 1],
            support[0],
            self.occupancy.grid,
            frame.timestamp,
            self.cameras,
        )
        hints = self.projector(prior, frame.cameras[0], transform, frame.timestamp)
        if mode == "zero":
            hints = torch.zeros_like(hints)
        elif mode == "shuffle":
            if self.permutation is None or self.permutation.numel() != hints.shape[1]:
                rng = torch.Generator(device=hints.device).manual_seed(42)
                self.permutation = torch.randperm(
                    hints.shape[1], generator=rng, device=hints.device
                )
            hints = hints[:, self.permutation]
        tensor = rgb[None, None].to(
            self.backend.device, dtype=self.backend.wrapper.compute_dtype
        )
        with (
            self.conditioner.condition(hints),
            torch.autocast(
                "cuda" if self.backend.device.startswith("cuda") else "cpu",
                dtype=self.backend.wrapper.compute_dtype,
                enabled=self.backend.device.startswith("cuda"),
            ),
        ):
            prediction = self.backend.network(
                tensor,
                causal_global_attn=True,
                long_sequence_parallel=True,
                streaming_inference=True,
                past_key_values=state.carry,
                ref_hidden=state.ref_hidden,
                camera_state=state.camera_state,
            )
        target = torch.as_tensor(
            lidar_target(root, row, self.occupancy.grid)
            if occ_target is None
            else occ_target,
            dtype=torch.long,
            device=logits.device,
        )
        mask = support[0] & (target >= 0)
        if not mask.any():
            raise ValueError("No observed camera-supported occupancy supervision")
        loss_occ = F.cross_entropy(logits[0][mask], target[mask])
        ids, depth_target = depth_samples(
            root, row, frame, transform, self.backend.device
        )
        if len(ids) < 16:
            raise ValueError("Insufficient visible LiDAR depth for joint training")
        depth = (
            prediction["local_points"][0, 0, ..., 2]
            .float()
            .reshape(-1)[ids]
            .clamp_min(1e-5)
        )
        residual = depth.log() - depth_target.log()
        loss_depth = (residual - residual.mean()).square().mean()
        if not torch.isfinite(loss_occ) or not torch.isfinite(loss_depth):
            raise ValueError("Non-finite joint training loss")
        carry = self.backend.network._postprocess_stream_carry(
            detach_carry(prediction.get("past_key_values")),
            spatial_hw=tuple(tensor.shape[-2:]),
        )
        next_state = JointState(
            carry,
            detach_tree(prediction.get("ref_hidden", state.ref_hidden)),
            detach_tree(prediction.get("camera_state")),
            frame.context,
            frame.timestamp,
            transform,
        )
        occupied = logits[0].argmax(-1) == 1
        truth = target == 1
        return {
            "loss_occ": loss_occ,
            "loss_depth": loss_depth,
            "logits": logits,
            "images": images,
            "hints": hints,
            "state": next_state,
            "state_reset": reset,
            "observed_voxels": int(mask.sum()),
            "depth_samples": len(ids),
            "intersection": int((occupied & truth & mask).sum()),
            "union": int(((occupied | truth) & mask).sum()),
        }

    def close(self):
        self.conditioner.close()
        self.backend.close()


def weighted_objective(result, occ_weight, depth_weight):
    return occ_weight * result["loss_occ"] + depth_weight * result["loss_depth"]


def make_clips(items, clip_frames):
    if clip_frames <= 0:
        raise ValueError("clip_frames must be positive")
    clips, current = [], []
    for item in items:
        if current and (
            len(current) == clip_frames
            or item[0].context != current[-1][0].context
            or item[0].timestamp <= current[-1][0].timestamp
        ):
            clips.append(current)
            current = []
        current.append(item)
    if current:
        clips.append(current)
    return clips


class TargetCache:
    """Bounded CPU label cache; online prediction never receives these labels."""

    def __init__(self, grid, capacity=4):
        self.grid, self.capacity, self.entries = grid, capacity, OrderedDict()

    def get(self, item):
        frame, root, row = item
        key = (str(root), frame.context, frame.timestamp_us)
        if key not in self.entries:
            self.entries[key] = lidar_target(root, row, self.grid)
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return self.entries[key]


def dataset_identity(roots, train, val):
    manifests = []
    for root in roots:
        path = Path(root) / "source-manifest.json"
        if not path.is_file():
            path = Path(root) / "scene/source-manifest.json"
        manifests.append(file_sha256(path))
    return {
        "manifest_sha256": manifests,
        "train": [[f.context, f.timestamp_us] for f, _, _ in train],
        "validation": [[f.context, f.timestamp_us] for f, _, _ in val],
    }


def validate_resume(saved, identity, config, abot_sha256):
    if saved.get("format") != "realtime-map-occ-joint-v1":
        raise ValueError("Expected a joint training state checkpoint")
    if saved.get("dataset") != identity:
        raise ValueError("Resume data or context split differs from the checkpoint")
    if saved.get("config") != config:
        raise ValueError("Resume training configuration differs from the checkpoint")
    if saved.get("abot_checkpoint_sha256") != abot_sha256:
        raise ValueError("Resume requires the same frozen ABot checkpoint")
    if saved.get("stream_boundary") != "reset_between_optimizer_clips":
        raise ValueError("Resume checkpoint is not at a supported stream boundary")


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def export_inference(output, state, validation):
    training = {
        "steps": state["steps"],
        "dataset": "waymo",
        "contexts": state["training_contexts"],
        "validation_contexts": sorted({x[0] for x in state["dataset"]["validation"]}),
        "joint": True,
        "frozen": ["ABot"],
        "loss": "observed OCC cross entropy + scale-invariant observed camera-z depth",
    }
    occ = {
        "format": "realtime-map-occ-v1",
        "model_config": state["model_config"],
        "state_dict": state["occ_state_dict"],
        "cameras": state["config"]["cameras"],
        "image_size": state["config"]["image_size"],
        "training": training,
        "validation": validation.get("predicted", {}),
    }
    checkpoint(output / "occ.pt", occ)
    adapter = {
        "format": "realtime-map-occ-adapter-v2",
        "adapter_config": state["adapter_config"],
        "state_dict": state["adapter_state_dict"],
        "cameras": occ["cameras"],
        "projection": state["projection"],
        "occ_checkpoint_sha256": file_sha256(output / "occ.pt"),
        "abot_checkpoint_sha256": state["abot_checkpoint_sha256"],
        "training": training,
        "validation": validation,
    }
    checkpoint(output / "adapter.pt", adapter)


def evaluate(runner, items, targets):
    runner.occupancy.eval()
    runner.adapter.eval()
    report = {}
    with torch.no_grad():
        for mode in ("predicted", "zero", "shuffle"):
            state, losses, intersection, union = None, [], 0, 0
            for item in items:
                result = runner.forward(
                    item, state, occ_target=targets.get(item), mode=mode
                )
                state = result["state"]
                losses.append([float(result["loss_occ"]), float(result["loss_depth"])])
                intersection += result["intersection"]
                union += result["union"]
            report[mode] = {
                "frames": len(losses),
                "occupancy_cross_entropy": float(np.mean(losses, axis=0)[0]),
                "scale_invariant_log_depth_loss": float(np.mean(losses, axis=0)[1]),
                "occupied_iou": intersection / union if union else None,
            }
    return report


def train(args):
    from .backend import ABotStream

    train_items, val = context_split(args.input, set(args.validation_context))
    cameras = CAMERAS if args.cameras == 3 else (CAMERAS[0],)
    for frame, root, row in train_items + val:
        if not set(cameras) <= {c.name for c in frame.cameras}:
            raise ValueError("A training/validation frame is missing a required camera")
        if "lidar" not in row or not (root / row["lidar"]["file"]).is_file():
            raise ValueError(
                "Joint training requires prepared measured LiDAR supervision"
            )
    identity = dataset_identity(args.input, train_items, val)
    config = {
        "cameras": list(cameras),
        "image_size": [192, 320],
        "clip_frames": args.clip_frames,
        "lr_occ": args.lr_occ,
        "lr_adapter": args.lr_adapter,
        "occ_weight": args.occ_weight,
        "depth_weight": args.depth_weight,
        "seed": args.seed,
        "device_type": torch.device(args.device).type,
        "gradient_audit": args.gradient_audit,
    }
    abot_sha = file_sha256(args.abot_checkpoint)
    saved = (
        torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.resume
        else None
    )
    if saved:
        validate_resume(saved, identity, config, abot_sha)
        if args.steps <= saved["steps"]:
            raise ValueError(
                "--steps is the total target and must exceed resumed completed steps"
            )
    output = Path(args.output)
    reuse = bool(
        saved and Path(args.resume).resolve() == (output / "state.pt").resolve()
    )
    output.mkdir(parents=True, exist_ok=reuse)
    torch.manual_seed(args.seed)
    inherited_contexts = set()
    if saved:
        from .occupancy import GridSpec

        model_config = dict(saved["model_config"])
        model_config["grid"] = GridSpec(**model_config["grid"])
        occupancy = CameraOccupancyNet(**model_config).to(args.device)
        occupancy.load_state_dict(saved["occ_state_dict"], strict=True)
        inherited_contexts.update(saved["training_contexts"])
    elif args.occ_checkpoint:
        occupancy, source_cameras, metadata = load_occ_checkpoint(
            args.occ_checkpoint, args.device
        )
        if source_cameras != cameras:
            raise ValueError("Initial OCC checkpoint camera set differs from --cameras")
        inherited_contexts.update(metadata["training"].get("contexts", []))
    else:
        occupancy = CameraOccupancyNet().to(args.device)
    if inherited_contexts & {f.context for f, _, _ in val}:
        raise ValueError("Validation contexts were used to train the initial OCC model")
    occupancy.requires_grad_(True)
    backend = ABotStream(args.abot_checkpoint, "sdpa", device=args.device)
    adapter = OccTokenAdapter(backend.network.encoder.embed_dim).to(args.device)
    if saved:
        if saved["adapter_config"] != {
            "token_dim": adapter.token_dim,
            "hidden_dim": adapter.hidden_dim,
        }:
            raise ValueError("Resume adapter geometry differs from ABot encoder")
        adapter.load_state_dict(saved["adapter_state_dict"], strict=True)
    runner = JointFrameRunner(backend, occupancy, adapter, cameras)
    optimizer = torch.optim.AdamW(
        [
            {"params": occupancy.parameters(), "lr": args.lr_occ},
            {"params": adapter.parameters(), "lr": args.lr_adapter},
        ]
    )
    if saved:
        if saved["projection"] != runner.projector.config():
            raise ValueError("Resume projection configuration differs")
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_cpu"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
    clips = make_clips(train_items, args.clip_frames)
    targets = TargetCache(occupancy.grid)
    completed = saved["steps"] if saved else 0
    counts = {
        "OCC": sum(p.numel() for p in occupancy.parameters()),
        "adapter": sum(p.numel() for p in adapter.parameters()),
        "ABot": 0,
    }
    print(
        json.dumps(
            {
                "trainable_parameters": counts,
                "clips": len(clips),
                "resume_step": completed,
                "cameras": cameras,
            }
        ),
        flush=True,
    )
    (output / "config.json").write_text(
        json.dumps({**vars(args), "training": config}, indent=2, default=str)
    )
    (output / "report.json").write_text(
        json.dumps(
            {
                "completed": False,
                "optimizer_steps": completed,
                "requested_optimizer_steps": args.steps,
                "status": "training_or_validation_in_progress",
            },
            indent=2,
        )
    )

    def training_state():
        return {
            "format": "realtime-map-occ-joint-v1",
            "steps": completed,
            "dataset": identity,
            "config": config,
            "abot_checkpoint_sha256": abot_sha,
            "training_contexts": sorted(
                inherited_contexts | {f.context for f, _, _ in train_items}
            ),
            "model_config": occupancy.config(),
            "occ_state_dict": cpu_tree(occupancy.state_dict()),
            "adapter_config": {
                "token_dim": adapter.token_dim,
                "hidden_dim": adapter.hidden_dim,
            },
            "adapter_state_dict": cpu_tree(adapter.state_dict()),
            "optimizer": cpu_tree(optimizer.state_dict()),
            "projection": runner.projector.config(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all()
            if args.device.startswith("cuda")
            else [],
            "stream_boundary": "reset_between_optimizer_clips",
        }

    try:
        with (output / "training.jsonl").open(
            "a" if reuse else "w", buffering=1
        ) as log:
            for step in range(completed, args.steps):
                started = time.perf_counter()
                occupancy.train()
                adapter.train()
                optimizer.zero_grad(set_to_none=True)
                clip, state, rows = clips[step % len(clips)], None, []
                for item in clip:
                    result = runner.forward(
                        item,
                        state,
                        occ_target=targets.get(item),
                        audit_images=args.gradient_audit,
                    )
                    state = result["state"]
                    audit = reconstruction_view_grads = None
                    if args.gradient_audit:
                        grad, image_grad = torch.autograd.grad(
                            result["loss_depth"],
                            (result["logits"], result["images"]),
                            retain_graph=True,
                        )
                        audit = float(grad.detach().norm())
                        reconstruction_view_grads = (
                            image_grad.flatten(2).norm(dim=-1)[0].tolist()
                        )
                    loss = weighted_objective(
                        result, args.occ_weight, args.depth_weight
                    ) / len(clip)
                    loss.backward()
                    view_grads = None
                    if args.gradient_audit:
                        view_grads = (
                            result["images"].grad.flatten(2).norm(dim=-1)[0].tolist()
                        )
                    rows.append(
                        {
                            "context": item[0].context,
                            "timestamp_us": item[0].timestamp_us,
                            "occ_loss": float(result["loss_occ"].detach()),
                            "depth_loss": float(result["loss_depth"].detach()),
                            "reconstruction_to_occ_logits_grad_norm": audit,
                            "reconstruction_image_grad_norm_by_view": reconstruction_view_grads,
                            "joint_loss_image_grad_norm_by_view": view_grads,
                            "observed_voxels": result["observed_voxels"],
                            "depth_samples": result["depth_samples"],
                            "state_reset": result["state_reset"],
                        }
                    )
                    del result, loss
                if any(p.grad is not None for p in backend.network.parameters()):
                    raise RuntimeError(
                        "Frozen ABot unexpectedly received parameter gradients"
                    )
                norm = torch.nn.utils.clip_grad_norm_(
                    list(occupancy.parameters()) + list(adapter.parameters()),
                    1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                completed = step + 1
                state = None
                record = {
                    "step": completed,
                    "frames": rows,
                    "gradient_norm_before_clip": float(norm),
                    "wall_seconds": time.perf_counter() - started,
                    "frozen_abot_gradients_absent": True,
                }
                log.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                if completed % args.save_every == 0 or completed == args.steps:
                    checkpoint(output / "state.pt", training_state())
        validation = evaluate(runner, val, targets)
        state = training_state()
        export_inference(output, state, validation)
        report = {
            "completed": True,
            "optimizer_steps": completed,
            "trainable_parameters": counts,
            "train_frames_available": len(train_items),
            "validation_frames": len(val),
            "validation": validation,
            "cameras": list(cameras),
            "abot_camera": "FRONT",
            "projection": runner.projector.config(),
            "frozen": ["ABot"],
            "stream_policy": "causal within clips; detached history each frame; reset between optimizer clips",
            "resume_policy": "model + optimizer + RNG + deterministic next clip; no historical KV serialized",
            "quality_improvement_established": False,
            "realtime_established": False,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        runner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--validation-context", action="append", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="New directory for training state, inference checkpoints and reports",
    )
    parser.add_argument("--cameras", type=int, choices=[1, 3], default=3)
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Total optimizer clip updates, including resumed steps",
    )
    parser.add_argument("--clip-frames", type=int, default=4)
    parser.add_argument("--lr-occ", type=float, default=1e-4)
    parser.add_argument("--lr-adapter", type=float, default=1e-4)
    parser.add_argument("--occ-weight", type=float, default=1.0)
    parser.add_argument("--depth-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--gradient-audit",
        action="store_true",
        help="Extra backward audit; disable for performance measurements",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--abot-checkpoint", default="models/abot_recon.safetensors")
    start = parser.add_mutually_exclusive_group()
    start.add_argument(
        "--occ-checkpoint", help="Optional compatible trained OCC initialization"
    )
    start.add_argument(
        "--resume", help="state.pt from a completed optimizer clip boundary"
    )
    args = parser.parse_args()
    if (
        min(args.steps, args.clip_frames, args.save_every, args.lr_occ, args.lr_adapter)
        <= 0
    ):
        parser.error(
            "steps, clip size, save interval and learning rates must be positive"
        )
    if not np.isfinite(
        [args.lr_occ, args.lr_adapter, args.occ_weight, args.depth_weight]
    ).all():
        parser.error("learning rates and loss weights must be finite")
    if args.occ_weight <= 0 or args.depth_weight <= 0:
        parser.error("Both OCC and reconstruction supervision need positive weights")
    train(args)


if __name__ == "__main__":
    main()
