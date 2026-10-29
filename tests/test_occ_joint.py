# SPDX-License-Identifier: Apache-2.0
"""Joint-learning regressions without loading the released 4 GB ABot model."""

from collections import namedtuple
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
import torch.nn.functional as F

from realtime_map.occ_fusion import EncoderConditioner, OccTokenAdapter
from realtime_map.occ_model import CameraOccupancyNet
from realtime_map.occ_projection import SoftVoxelProjector
from realtime_map.occupancy import GridSpec, VoxelPrediction
from realtime_map.waymo import CAMERAS, CameraView, WaymoFrame


class FrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8)

    def forward(self, tokens):
        return {"x_norm_patchtokens": self.projection(tokens)}


class FrozenReconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FrozenEncoder()
        self.decoder = nn.Sequential(nn.Linear(8, 12), nn.GELU(), nn.Linear(12, 3))

    def forward(self, tokens):
        return self.decoder(self.encoder(tokens)["x_norm_patchtokens"])


def one_patch_geometry():
    camera = CameraView(
        "FRONT",
        None,
        np.array([[14.0, 0, 6.5], [0, 14.0, 6.5], [0, 0, 1.0]]),
        np.zeros(5),
        np.eye(4),
        14,
        14,
    )
    values = dict(
        target_height=14,
        target_width=14,
        source_height=14,
        source_width=14,
        resized_height=14,
        crop_top=0,
        pad_top=0,
        pad_bottom=0,
    )
    return camera, namedtuple("Transform", values)(**values)


def test_joint_reconstruction_gradient_reaches_three_cameras_after_zero_init():
    """CE alone is not evidence that the reconstruction loss can train OCC."""
    torch.manual_seed(19)
    grid = GridSpec((-1.0, -1.0, 1.0), (0.5, 0.5, 0.5), (4, 4, 4))
    occupancy = CameraOccupancyNet(grid, channels=8, feature_layers=2)
    reconstruction = FrozenReconstruction().eval().requires_grad_(False)
    adapter = OccTokenAdapter(token_dim=8, hidden_dim=4)
    conditioner = EncoderConditioner(reconstruction.encoder, adapter)
    projector = SoftVoxelProjector()
    camera, transform = one_patch_geometry()
    images = torch.rand(1, 3, 3, 32, 48, requires_grad=True)
    intrinsic = torch.tensor([[30.0, 0, 23.5], [0, 30.0, 15.5], [0, 0, 1.0]])
    intrinsic = intrinsic[None, None].repeat(1, 3, 1, 1)
    distortion = torch.zeros(1, 3, 5)
    extrinsic = torch.eye(4)[None, None].repeat(1, 3, 1, 1)
    tokens = torch.randn(1, 1, 8)
    target = torch.tensor([[[0.4, 1.0, 2.0]]])
    labels = torch.zeros((1, *grid.shape), dtype=torch.long)
    labels[..., -1] = 1
    optimizer = torch.optim.AdamW(
        [*occupancy.parameters(), *adapter.parameters()], lr=0.01
    )

    def forward():
        logits, support = occupancy(images, intrinsic, distortion, extrinsic)
        prior = VoxelPrediction(
            logits.softmax(-1)[0, ..., 1], support[0], grid, 0.0, CAMERAS
        )
        hints = projector(prior, camera, transform, 0.0)
        with conditioner.condition(hints):
            result = reconstruction(tokens)
        return F.mse_loss(result, target), F.cross_entropy(logits[support], labels[support])

    try:
        first_depth, first_occ = forward()
        gradients = torch.autograd.grad(
            first_depth, tuple(occupancy.parameters()), retain_graph=True
        )
        assert sum(float(g.abs().sum()) for g in gradients) == 0.0
        (first_depth + first_occ).backward()
        assert adapter.projection[-1].weight.grad.abs().sum() > 0
        assert occupancy.backbone[0].weight.grad.abs().sum() > 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        second_depth, _ = forward()
        source_grad, camera_grad = torch.autograd.grad(
            second_depth,
            (occupancy.backbone[0].weight, images),
            retain_graph=True,
        )
        assert torch.isfinite(source_grad).all() and source_grad.abs().sum() > 0
        assert torch.isfinite(camera_grad).all()
        assert (camera_grad.abs().sum(dim=(0, 2, 3, 4)) > 0).all()
        second_depth.backward()
        assert adapter.projection[0].weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in reconstruction.parameters())
        assert conditioner.calls == 2
    finally:
        conditioner.close()


def test_unknown_soft_prior_cannot_update_frozen_reconstruction_or_source():
    """An unobserved volume must not become an arbitrary residual input."""
    torch.manual_seed(31)
    grid = GridSpec((-1.0, -1.0, 1.0), (0.5, 0.5, 0.5), (4, 4, 4))
    probability = torch.rand(grid.shape, requires_grad=True)
    prior = VoxelPrediction(
        probability, torch.zeros(grid.shape, dtype=torch.bool), grid, 0.0, CAMERAS
    )
    camera, transform = one_patch_geometry()
    hints = SoftVoxelProjector()(prior, camera, transform, 0.0)
    adapter = OccTokenAdapter(8, 4)
    with torch.no_grad():
        adapter.projection[-1].weight.fill_(0.2)
        adapter.projection[-1].bias.fill_(0.2)
    tokens = torch.randn(1, 1, 8)
    result = adapter(tokens, hints)
    torch.testing.assert_close(result, tokens, rtol=0, atol=0)
    result.sum().backward()
    assert torch.isfinite(probability.grad).all()
    assert probability.grad.abs().sum() == 0
    assert all(p.grad.abs().sum() == 0 for p in adapter.parameters())


@pytest.fixture
def calibrated_three_view_frame(tmp_path):
    from PIL import Image

    front, _ = one_patch_geometry()
    cameras = []
    for index, name in enumerate(CAMERAS):
        image = tmp_path / f"{name}.png"
        rng = np.random.default_rng(index)
        Image.fromarray(rng.integers(0, 256, (14, 14, 3), dtype=np.uint8)).save(image)
        yaw = (0.0, 0.15, -0.15)[index]
        pose = np.eye(4)
        pose[:3, :3] = [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                         [-np.sin(yaw), 0, np.cos(yaw)]]
        pose[0, 3] = (0.0, -0.1, 0.1)[index]
        cameras.append(replace(front, name=name, path=image, ego_from_camera=pose))
    return WaymoFrame(0, 1.0, cameras[0].path, "train-a", 1_000_000, tuple(cameras))


class ImagePatchEncoder(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, self.embed_dim)

    def forward(self, images):
        tokens = self.linear(images.mean(dim=(-2, -1)))[:, None]
        return {"x_norm_patchtokens": tokens}


class TinyCausalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ImagePatchEncoder()
        self.decoder = nn.Linear(8, 14 * 14)
        self.arguments = []

    def forward(self, images, **kwargs):
        self.arguments.append(kwargs)
        assert kwargs["causal_global_attn"]
        assert kwargs["long_sequence_parallel"] and kwargs["streaming_inference"]
        tokens = self.encoder(images[:, 0])["x_norm_patchtokens"]
        if kwargs["past_key_values"] is not None:
            tokens = tokens + 0.1 * kwargs["past_key_values"][0][0].reshape_as(tokens)
        depth = self.decoder(tokens).exp().reshape(1, 1, 14, 14)
        return {
            "local_points": torch.stack((torch.zeros_like(depth), torch.zeros_like(depth), depth), -1),
            "past_key_values": [(tokens[:, None], tokens[:, None] * 2)],
            "ref_hidden": tokens if kwargs["ref_hidden"] is None else kwargs["ref_hidden"],
            "camera_state": {"previous_pose": tokens.sum(-1),
                             "nested": (tokens * 2, [tokens * 3])},
        }

    def _postprocess_stream_carry(self, carry, spatial_hw):
        assert spatial_hw == (14, 14)
        assert all(not t.requires_grad for pair in carry for t in pair)
        return carry


class TinyBackend:
    def __init__(self):
        self.device = "cpu"
        self.network = TinyCausalNetwork()
        self.wrapper = SimpleNamespace(compute_dtype=torch.float32)
        self.resets = 0
        self.closed = False

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


def assert_detached(value):
    if torch.is_tensor(value):
        assert not value.requires_grad and value.grad_fn is None
    elif isinstance(value, dict):
        for child in value.values():
            assert_detached(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            assert_detached(child)


def test_joint_runner_detaches_causal_history_and_resets_contexts(
    monkeypatch, calibrated_three_view_frame, tmp_path
):
    import realtime_map.occ_joint_training as joint

    preprocessing = pytest.importorskip("abot_recon.preprocessing")
    _, transform = one_patch_geometry()

    def preprocess(image):
        return torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 255, transform

    monkeypatch.setattr(preprocessing, "preprocess_image", preprocess)
    monkeypatch.setattr(joint, "depth_samples", lambda *args: (
        torch.arange(14 * 14), torch.linspace(1, 3, 14 * 14)))
    grid = GridSpec((-1.0, -1.0, 1.0), (0.5, 0.5, 0.5), (4, 4, 4))
    occupancy = CameraOccupancyNet(grid, channels=8, feature_layers=2)
    backend = TinyBackend()
    adapter = OccTokenAdapter(8, 4)
    runner = joint.JointFrameRunner(backend, occupancy, adapter, image_size=(32, 48))
    frame = calibrated_three_view_frame
    frames = [frame, replace(frame, frame_id=1, timestamp=1.1, timestamp_us=1_100_000),
              replace(frame, frame_id=2, timestamp=2.0, timestamp_us=2_000_000, context="train-b"),
              replace(frame, frame_id=3, timestamp=1.9, timestamp_us=1_900_000, context="train-b")]
    targets = torch.zeros(grid.shape, dtype=torch.long)
    targets[..., -1] = 1
    state = None
    try:
        for index, current in enumerate(frames):
            result = runner.forward((current, tmp_path, {}), state, occ_target=targets,
                                    audit_images=True)
            assert result["state_reset"] == (index != 1)
            assert result["images"].shape[:2] == (1, 3)
            assert result["depth_samples"] == 14 * 14
            joint.weighted_objective(result, 1.0, 1.0).backward()
            state = result["state"]
            assert_detached(state.carry)
            assert_detached(state.ref_hidden)
            assert_detached(state.camera_state)
            assert state.context == current.context and state.timestamp == current.timestamp
        arguments = backend.network.arguments
        assert arguments[0]["past_key_values"] is None
        assert arguments[1]["past_key_values"] is not None
        assert arguments[1]["ref_hidden"] is not None
        assert arguments[1]["camera_state"] is not None
        assert arguments[2]["past_key_values"] is None and arguments[2]["camera_state"] is None
        assert arguments[3]["past_key_values"] is None
        assert backend.resets == 3
        assert all(p.grad is None for p in backend.network.parameters())
    finally:
        runner.close()
    assert backend.closed
    assert not backend.network.encoder._forward_hooks


def test_joint_clips_never_cross_context_or_rewind_boundary(calibrated_three_view_frame):
    from realtime_map.occ_joint_training import make_clips

    frame = calibrated_three_view_frame
    stamps = [("a", 1), ("a", 2), ("a", 3), ("b", 4), ("b", 3), ("b", 4)]
    items = [(replace(frame, context=context, timestamp=stamp), None, {})
             for context, stamp in stamps]
    clips = make_clips(items, 2)
    assert [[(x[0].context, x[0].timestamp) for x in clip] for clip in clips] == [
        [("a", 1), ("a", 2)], [("a", 3)], [("b", 4)], [("b", 3), ("b", 4)]]
    assert make_clips([], 2) == []
    with pytest.raises(ValueError, match="positive"):
        make_clips(items, 0)


def test_supervision_rows_are_selected_by_context_and_timestamp(
    monkeypatch, calibrated_three_view_frame, tmp_path
):
    from realtime_map import occ_training

    frame = calibrated_three_view_frame
    rows = [{"context": context, "timestamp_micros": frame.timestamp_us,
             "lidar": {"file": f"{context}.bin"}} for context in ("train", "heldout")]
    (tmp_path / "source-manifest.json").write_text(json.dumps({"frames": rows}))
    monkeypatch.setattr(occ_training, "read_waymo", lambda root, context: [
        replace(frame, context=context)])
    train, validation = occ_training.context_split([tmp_path], {"heldout"})
    assert train[0][2]["lidar"]["file"] == "train.bin"
    assert validation[0][2]["lidar"]["file"] == "heldout.bin"


@pytest.mark.parametrize("field,replacement,error", [
    ("format", "adapter-v2", "joint training state"),
    ("dataset", {"train": [["heldout", 1]]}, "data or context split"),
    ("config", {"clip_frames": 5}, "configuration"),
    ("abot_checkpoint_sha256", "different-model", "same frozen ABot"),
    ("stream_boundary", "mid-frame", "stream boundary"),
])
def test_joint_resume_rejects_changed_training_contract(field, replacement, error):
    from realtime_map.occ_joint_training import validate_resume

    identity = {"train": [["a", 1]], "validation": [["b", 1]]}
    config = {"clip_frames": 2}
    saved = {"format": "realtime-map-occ-joint-v1", "dataset": identity,
             "config": config, "abot_checkpoint_sha256": "abot-hash",
             "stream_boundary": "reset_between_optimizer_clips"}
    validate_resume(saved, identity, config, "abot-hash")
    saved[field] = replacement
    with pytest.raises(ValueError, match=error):
        validate_resume(saved, identity, config, "abot-hash")


def test_joint_export_reloads_identical_soft_hints_without_labels(
    calibrated_three_view_frame, tmp_path
):
    from realtime_map.occ_joint_training import export_inference
    from realtime_map.occ_model import OccRuntime, file_sha256
    from realtime_map.waymo import camera_batch

    torch.manual_seed(47)
    grid = GridSpec((-1.0, -1.0, 1.0), (0.5, 0.5, 0.5), (4, 4, 4))
    occupancy = CameraOccupancyNet(grid, channels=8, feature_layers=2).eval()
    adapter = OccTokenAdapter(8, 4).eval()
    projector = SoftVoxelProjector()
    with torch.no_grad():
        adapter.projection[-1].weight.fill_(0.1)
    frozen_abot = tmp_path / "frozen_abot.bin"
    frozen_abot.write_bytes(b"fixed-frozen-model-weights")
    state = {"steps": 2, "training_contexts": ["train-a"],
             "dataset": {"validation": [["validation-b", 1]]},
             "config": {"cameras": list(CAMERAS), "image_size": [32, 48]},
             "model_config": occupancy.config(), "occ_state_dict": occupancy.state_dict(),
             "adapter_config": {"token_dim": 8, "hidden_dim": 4},
             "adapter_state_dict": adapter.state_dict(), "projection": projector.config(),
             "abot_checkpoint_sha256": file_sha256(frozen_abot)}
    export_inference(tmp_path, state, {"predicted": {"frames": 1}})
    runtime = OccRuntime.from_checkpoints(tmp_path / "occ.pt", tmp_path / "adapter.pt", "cpu")
    assert isinstance(runtime.projector, SoftVoxelProjector)
    assert runtime.projector.config() == projector.config()
    frame = calibrated_three_view_frame
    _, transform = one_patch_geometry()
    with torch.no_grad():
        logits, support = occupancy(*camera_batch(frame, CAMERAS, (32, 48)))
        prediction = VoxelPrediction(logits.softmax(-1)[0, ..., 1], support[0], grid,
                                     frame.timestamp, CAMERAS)
        expected = projector(prediction, frame.cameras[0], transform, frame.timestamp)
        actual, timing = runtime.prepare(frame, transform)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        tokens = torch.randn(1, 1, 8)
        torch.testing.assert_close(runtime.adapter(tokens, actual), adapter(tokens, expected), rtol=0, atol=0)
    assert timing["occ_predict_ms"] >= 0 and timing["occ_project_ms"] >= 0
    assert not any(p.requires_grad for p in runtime.source.model.parameters())
    runtime.validate_abot_checkpoint(frozen_abot)
    frozen_abot.write_bytes(b"different-model-weights")
    with pytest.raises(ValueError, match="different ABot checkpoint"):
        runtime.validate_abot_checkpoint(frozen_abot)
    checkpoint = torch.load(tmp_path / "adapter.pt", weights_only=True)
    checkpoint["projection"]["version"] = 999
    torch.save(checkpoint, tmp_path / "adapter.pt")
    with pytest.raises(ValueError, match="Unsupported joint OCC projection"):
        OccRuntime.from_checkpoints(tmp_path / "occ.pt", tmp_path / "adapter.pt", "cpu")
