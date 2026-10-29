# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from realtime_map.occ_fusion import EncoderConditioner, OccTokenAdapter
from realtime_map.occ_model import CameraOccupancyNet
from realtime_map.occupancy import (
    GridSpec,
    VoxelPrediction,
    VoxelProjector,
    lidar_supervision,
    prediction_from_logits,
)
from realtime_map.waymo import CameraView, patch_rays, read_waymo


def camera():
    return CameraView(
        "FRONT",
        None,
        np.array([[14.0, 0, 6.5], [0, 14.0, 6.5], [0, 0, 1.0]]),
        np.zeros(5),
        np.eye(4),
        14,
        14,
    )


def transform(**kwargs):
    defaults = dict(
        target_height=14,
        target_width=14,
        source_height=14,
        source_width=14,
        resized_height=14,
        crop_top=0,
        pad_top=0,
        pad_bottom=0,
    )
    defaults.update(kwargs)
    # Real FovTransform is immutable/hashable; use a tuple-backed test equivalent.
    from collections import namedtuple

    return namedtuple("Transform", defaults)(**defaults)


def test_voxel_first_hit_unknown_and_calibrated_depth():
    spec = GridSpec((-1.0, -1.0, 1.0), (0.2, 0.2, 0.2), (10, 10, 20))
    prob = torch.zeros(spec.shape)
    prob[:, :, 5] = 0.95
    valid = torch.ones_like(prob, dtype=torch.bool)
    prior = VoxelPrediction(prob, valid, spec, 1.0, ("FRONT",))
    hints = VoxelProjector()(prior, camera(), transform(), 1.0)
    depth = hints[0, 0, 0].expm1().item()
    assert 2.0 <= depth <= 2.2
    assert hints[0, 0, 1].item() == pytest.approx(0.95)
    assert hints[0, 0, 3].item() == 1
    unknown = VoxelPrediction(prob, torch.zeros_like(valid), spec, 1.0, ("FRONT",))
    assert not VoxelProjector()(unknown, camera(), transform(), 1.0).any()
    with pytest.raises(ValueError, match="current frame"):
        VoxelProjector()(prior, camera(), transform(), 1.1)


def test_rays_parallel_outside_grid_do_not_create_hits():
    spec = GridSpec((5.0, -1.0, 1.0), (0.2, 0.2, 0.2), (10, 10, 10))
    p = torch.ones(spec.shape)
    hints = VoxelProjector()(
        VoxelPrediction(p, p.bool(), spec, 0.0, ("FRONT",)), camera(), transform(), 0.0
    )
    assert not hints.any()


def test_patch_geometry_matches_pixel_centre_resize_and_padding():
    c = camera()
    rays, valid = patch_rays(c, transform())
    np.testing.assert_allclose(rays, [[0, 0, 1]], atol=1e-7)
    t = transform(target_height=42, pad_top=14, pad_bottom=14)
    rays, valid = patch_rays(c, t)
    np.testing.assert_array_equal(valid, [False, True, False])
    np.testing.assert_allclose(rays[1], [0, 0, 1], atol=1e-7)


def test_lidar_labels_leave_unseen_unknown_and_hits_win():
    spec = GridSpec((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (6, 2, 2))
    origin = np.array([[0.1, 0.5, 0.5], [0.1, 0.5, 0.5]])
    target = np.array([[2.5, 0.5, 0.5], [4.5, 0.5, 0.5]])
    labels = lidar_supervision(origin, target, spec)
    assert labels[0, 0, 0] == 0 and labels[1, 0, 0] == 0
    assert labels[2, 0, 0] == 1 and labels[4, 0, 0] == 1
    assert labels[5, 0, 0] == -1 and (labels[:, 1, :] == -1).all()


class TinyEncoder(nn.Module):
    def forward(self, x):
        return {"x_norm_patchtokens": x, "other": torch.tensor(7)}


@pytest.mark.parametrize("path", ["sdpa", "paged"])
def test_fusion_is_inside_causal_decoder_with_one_encoder_call(path):
    class TinyStream(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TinyEncoder()
            self.memory = None

        def forward(self, x):
            tokens = self.encoder(x)["x_norm_patchtokens"]
            result = tokens if self.memory is None else tokens + self.memory
            self.memory = tokens.detach()
            return result

        def _forward_frame_paged(self, x):
            return self.forward(x)

    model = TinyStream()
    adapter = OccTokenAdapter(8, 4)
    conditioner = EncoderConditioner(model.encoder, adapter)
    tokens = torch.randn(1, 3, 8)
    hints = torch.ones(1, 3, 4)
    forward = model if path == "sdpa" else model._forward_frame_paged
    baseline = forward(tokens)
    model.memory = None
    with conditioner.condition(hints):
        zero = forward(tokens)
    torch.testing.assert_close(zero, baseline, rtol=0, atol=0)
    with torch.no_grad():
        adapter.projection[-1].bias.fill_(0.25)
    model.memory = None
    with conditioner.condition(hints):
        enhanced = forward(tokens)
    assert torch.allclose(enhanced, baseline + 0.25)
    # Previous conditioned state influences the next frame even without new hints.
    later = forward(tokens)
    torch.testing.assert_close(later, tokens + enhanced)
    model.memory = None
    with conditioner.condition(hints):
        loss = forward(tokens).square().mean()
    loss.backward()
    assert adapter.projection[-1].weight.grad.abs().sum() > 0
    assert conditioner.calls == 3
    conditioner.close()


def test_exception_does_not_leak_a_prior_into_next_frame():
    encoder = TinyEncoder()
    adapter = OccTokenAdapter(8)
    conditioner = EncoderConditioner(encoder, adapter)
    with pytest.raises(RuntimeError, match="injected"):
        with conditioner.condition(torch.ones(1, 2, 4)):
            raise RuntimeError("injected")
    assert conditioner.pending is None
    tokens = torch.randn(1, 2, 8)
    assert encoder(tokens)["x_norm_patchtokens"] is tokens
    conditioner.close()


@pytest.mark.parametrize("views", [1, 3])
def test_camera_occ_network_is_trainable_and_masks_outside_fov(views):
    spec = GridSpec((-1.0, -1.0, 1.0), (0.5, 0.5, 0.5), (4, 4, 4))
    model = CameraOccupancyNet(spec, channels=8, feature_layers=2)
    images = torch.rand(1, views, 3, 32, 48)
    k = torch.tensor([[30.0, 0, 23.5], [0, 30.0, 15.5], [0, 0, 1.0]])[
        None, None
    ].repeat(1, views, 1, 1)
    d = torch.zeros(1, views, 5)
    extrinsic = torch.eye(4)[None, None].repeat(1, views, 1, 1)
    logits, valid = model(images, k, d, extrinsic)
    assert logits.shape == (1, 4, 4, 4, 2) and valid.any()
    loss = logits[valid].square().mean()
    loss.backward()
    assert model.backbone[0].weight.grad.abs().sum() > 0
    behind = extrinsic.clone()
    behind[:, :, 2, 3] = 100
    assert not model(images, k, d, behind)[1].any()


def write_waymo(root, contexts=("a",)):
    import json
    from PIL import Image

    (root / "extrinsics").mkdir()
    (root / "images").mkdir()
    np.savetxt(root / "extrinsics/0.txt", np.eye(4))
    rows = []
    for i, context in enumerate(contexts):
        Image.new("RGB", (14, 14)).save(root / f"images/{i}.jpg")
        rows.append(
            {
                "context": context,
                "timestamp_micros": 1_000_000 + i * 100_000,
                "sensors": {
                    "FRONT": {
                        "camera_id": 0,
                        "image": f"images/{i}.jpg",
                        "width": 14,
                        "height": 14,
                        "image_intrinsic": dict(
                            f_u=14,
                            f_v=14,
                            c_u=6.5,
                            c_v=6.5,
                            k1=0,
                            k2=0,
                            p1=0,
                            p2=0,
                            k3=0,
                        ),
                    }
                },
            }
        )
    (root / "source-manifest.json").write_text(json.dumps({"frames": rows}))


def test_waymo_contexts_cannot_be_silently_concatenated(tmp_path):
    write_waymo(tmp_path, ("a", "b"))
    with pytest.raises(ValueError, match="independent stream"):
        read_waymo(tmp_path)
    frames = read_waymo(tmp_path, context="a")
    assert len(frames) == 1 and frames[0].context == "a"
    # No source world poses, labels or LiDAR are needed for inference.
    np.testing.assert_allclose(frames[0].cameras[0].ego_from_camera[:3, 2], [1, 0, 0])


def test_checkpoint_rejects_untrained_or_wrong_camera_inputs(tmp_path):
    from realtime_map.occ_model import PredictedOccSource

    model = CameraOccupancyNet(GridSpec((-1, -1, 1), (0.5, 0.5, 0.5), (4, 4, 4)), 8, 2)
    state = {
        "format": "realtime-map-occ-v1",
        "model_config": model.config(),
        "state_dict": model.state_dict(),
        "training": {"steps": 0},
        "cameras": ["FRONT"],
        "image_size": [32, 48],
    }
    file = tmp_path / "model.pt"
    torch.save(state, file)
    with pytest.raises(ValueError, match="trained"):
        PredictedOccSource(file)
    state["training"]["steps"] = 1
    torch.save(state, file)
    with pytest.raises(ValueError, match="camera set"):
        PredictedOccSource(file, cameras=("FRONT", "FRONT_LEFT", "FRONT_RIGHT"))


def test_external_parking_logits_combine_wall_and_obstacles():
    grid = GridSpec((0, 0, 0), (1, 1, 1), (1, 1, 1))
    logits = torch.tensor([[[[[0.0, 1.0, 2.0]]]]])
    prediction = prediction_from_logits(
        logits, torch.ones(grid.shape, dtype=torch.bool), grid, 0.0, ("FRONT",)
    )
    expected = logits.softmax(-1)[0, 0, 0, 0, 1:].sum()
    torch.testing.assert_close(prediction.occupied_probability.squeeze(), expected)


def test_observed_empty_rays_are_not_unknown():
    spec = GridSpec((-1.0, -1.0, 1.0), (0.2, 0.2, 0.2), (10, 10, 10))
    p = torch.zeros(spec.shape)
    prior = VoxelPrediction(
        p, torch.ones_like(p, dtype=torch.bool), spec, 0.0, ("FRONT",)
    )
    hints = VoxelProjector()(prior, camera(), transform(), 0.0)
    assert hints[0, 0, 0] == 0 and hints[0, 0, 1] == 0
    assert hints[0, 0, 2] == 1 and hints[0, 0, 3] == 1


def test_context_train_validation_split_has_no_frame_leakage(tmp_path):
    from realtime_map.occ_training import context_split

    write_waymo(tmp_path, ("a", "b"))
    train, val = context_split([tmp_path], {"b"})
    assert {f.context for f, _, _ in train} == {"a"}
    assert {f.context for f, _, _ in val} == {"b"}
    with pytest.raises(ValueError, match="disjoint"):
        context_split([tmp_path], set())


def test_association_switch_metric_is_separate_from_geometry():
    from realtime_map.waymo_evaluation import association_scores

    scores = association_scores(
        {0: [("car", 1)], 1: [("car", 2)]}, {0: ["car"], 1: ["car"]}
    )
    assert scores["id_switches"] == 1 and scores["matched_fraction"] == 1
