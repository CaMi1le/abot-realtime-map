# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Frame-at-a-time adapter for pinned ABot-Recon, without output accumulation.

State transitions follow upstream network.py:1334-1445. Upstream is unmodified.
Adaptation: explicit per-frame state and CPU outputs instead of accumulated
sequence outputs. See NOTICE for upstream attribution and the pinned revision.
"""
from dataclasses import dataclass
from contextlib import nullcontext
import time
import numpy as np
from PIL import Image


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    camera_pose: np.ndarray  # camera to world, column-vector convention
    local_points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    valid_rows: slice
    model_ms: float
    preprocess_ms: float
    occ_predict_ms: float = 0.0
    occ_project_ms: float = 0.0


class ABotStream:
    def __init__(self, checkpoint, attention="sdpa", device="cuda", max_frames=22000,
                 occ_runtime=None):
        if occ_runtime is not None:
            occ_runtime.validate_abot_checkpoint(checkpoint)
        import torch
        from abot_recon import ABotRecon
        torch.set_num_threads(4)
        self.torch = torch
        self.api = ABotRecon.from_pretrained(checkpoint, device=device,
            attention_backend=attention, loop_closure=False, max_frames=max_frames,
            amp_dtype='bf16' if device.startswith('cuda') else 'fp32')
        self.wrapper = self.api.model
        self.network = self.wrapper.network
        self.device = device
        self.occ_runtime = occ_runtime
        self.conditioner = None
        if occ_runtime is not None:
            from .occ_fusion import EncoderConditioner
            self.conditioner = EncoderConditioner(self.network.encoder, occ_runtime.adapter)
        self.reset()

    def reset(self):
        self.wrapper.reset()
        self.carry = self.ref_h = self.camera_state = self.geometry = None
        self.index = 0
        self.paged = None
        if self.conditioner is not None:
            self.conditioner.pending = None

    def close(self):
        self.reset()
        if self.conditioner is not None:
            self.conditioner.close()
            self.conditioner = None

    def infer_stream(self, frames):
        self.reset()
        count = 0
        try:
            for frame in frames:
                yield self.step(frame)
                count += 1
            if count == 0:
                raise ValueError("infer_stream requires at least one frame")
        finally:
            self.reset()

    def step(self, frame):
        torch = self.torch
        with torch.inference_mode():
            return self._step(frame)

    def _step(self, frame):
        from abot_recon.preprocessing import preprocess_image
        from abot_recon.modeling.streaming.core.cache_utils import detach_carry
        torch = self.torch
        start = time.perf_counter()
        with Image.open(frame.path) as image:
            rgb, transform = preprocess_image(image.convert("RGB"))
        if self.geometry is not None and self.geometry != transform:
            raise ValueError("Frame geometry changed; start a new stream")
        self.geometry = transform
        colors = (rgb.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        tensor = rgb[None, None].to(self.device, dtype=self.wrapper.compute_dtype)
        if self.paged is None:
            self.paged = self.network._can_use_paged_kv(tensor)
            if self.paged:
                self.network._ensure_paged_manager(H=tensor.shape[-2], W=tensor.shape[-1],
                                                  dtype=tensor.dtype, device=tensor.device)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        pre_ms = (time.perf_counter() - start) * 1000
        hints = None
        occ_times = {"occ_predict_ms": 0., "occ_project_ms": 0.}
        if self.occ_runtime is not None:
            hints, occ_times = self.occ_runtime.prepare(frame, transform)
        start = time.perf_counter()
        scope = self.conditioner.condition(hints) if self.conditioner else nullcontext()
        with scope, torch.autocast("cuda" if self.device.startswith("cuda") else "cpu",
                            dtype=self.wrapper.compute_dtype,
                            enabled=self.device.startswith("cuda")):
            if self.paged:
                pred = self.network._forward_frame_paged(tensor, frame_idx=self.index,
                    ref_hidden=self.ref_h, camera_state=self.camera_state)
            else:
                pred = self.network.forward(tensor,
                    causal_global_attn=self.network.causal_global_attn,
                    long_sequence_parallel=True, streaming_inference=True,
                    past_key_values=self.carry, ref_hidden=self.ref_h,
                    camera_state=self.camera_state)
                self.carry = self.network._postprocess_stream_carry(
                    detach_carry(pred.get("past_key_values")),
                    spatial_hw=(tensor.shape[-2], tensor.shape[-1]))
        if pred.get("ref_hidden") is not None:
            self.ref_h = pred["ref_hidden"]
        self.camera_state = pred.get("camera_state", self.camera_state)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        model_ms = (time.perf_counter() - start) * 1000
        pose = pred["camera_poses"][0, 0].float().cpu().numpy()
        points = pred["local_points"][0, 0].float().cpu().numpy()
        conf = pred["conf"][0, 0].float().sigmoid().cpu().numpy().squeeze(-1)
        self.index += 1
        # Returning only CPU arrays releases dense CUDA outputs each frame.
        return FrameResult(frame.frame_id, frame.timestamp, pose, points, colors,
                           conf, transform.valid_rows, model_ms, pre_ms, **occ_times)
