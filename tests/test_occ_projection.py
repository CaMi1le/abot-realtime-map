# SPDX-License-Identifier: Apache-2.0
from collections import namedtuple
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from realtime_map.occ_projection import SoftVoxelProjector
from realtime_map.occupancy import GridSpec, VoxelPrediction
from realtime_map.waymo import CameraView


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
    return namedtuple("Transform", defaults)(**defaults)


def predict(probability, valid=None, grid=None):
    if grid is None:
        grid = GridSpec((-0.5, -0.5, 1.0), (1, 1, 1), (1, 1, len(probability)))
    probability = probability.reshape(grid.shape)
    if valid is None:
        valid = torch.ones_like(probability, dtype=torch.bool)
    return VoxelPrediction(probability, valid.reshape(grid.shape), grid, 0, ("FRONT",))


def test_first_hit_distribution_has_depth_and_opacity_gradients():
    probability = torch.tensor([0.2, 0.6], requires_grad=True)
    hints = SoftVoxelProjector()(predict(probability), camera(), transform(), 0)
    # First-cell mass .2; second-cell mass (1-.2)*.6=.48.
    depth = hints[0, 0, 0].expm1()
    assert depth.item() == pytest.approx((0.2 * 1.5 + 0.48 * 2.5) / 0.68)
    assert hints[0, 0, 1].item() == pytest.approx(0.68)
    gradient = torch.autograd.grad(depth, probability, retain_graph=True)[0]
    assert gradient[0] < 0 and gradient[1] > 0
    opacity_gradient = torch.autograd.grad(hints[0, 0, 1], probability)[0]
    torch.testing.assert_close(opacity_gradient, torch.tensor([0.4, 0.8]))


def test_one_voxel_has_one_opacity_event_regardless_of_path_length():
    probability = torch.tensor([0.3], requires_grad=True)
    grid = GridSpec((-1, -1, 1), (2, 2, 10), (1, 1, 1))
    hints = SoftVoxelProjector()(
        predict(probability, grid=grid), camera(), transform(), 0
    )
    assert hints[0, 0, 1].item() == pytest.approx(0.3)
    assert hints[0, 0, 0].expm1().item() == pytest.approx(6)
    hints[0, 0, 1].backward()
    assert probability.grad.item() == pytest.approx(1)


def test_diagonal_ray_crossing_voxel_corners_does_not_double_count_hits():
    grid = GridSpec((0, -0.5, 1), (1, 1, 1), (3, 1, 3))
    intrinsic = camera().intrinsic.copy()
    intrinsic[0, 2] = -7.5  # Patch centre has ray direction [1, 0, 1].
    hints = SoftVoxelProjector()(
        predict(torch.full(grid.shape, 0.4), grid=grid),
        replace(camera(), intrinsic=intrinsic),
        transform(),
        0,
    )
    assert hints[0, 0, 1].item() == pytest.approx(1 - 0.6**2)
    expected_depth = (0.4 * 1.5 + 0.6 * 0.4 * 2.5) / (1 - 0.6**2)
    assert hints[0, 0, 0].expm1().item() == pytest.approx(expected_depth)


def test_opaque_front_voxel_occludes_back_voxels():
    probability = torch.tensor([1.0, 0.7], requires_grad=True)
    hints = SoftVoxelProjector()(predict(probability), camera(), transform(), 0)
    assert hints[0, 0, 0].expm1().item() == pytest.approx(1.5)
    torch.testing.assert_close(hints[0, 0, 1:], torch.tensor([1.0, 0.0, 1.0]))
    hints.sum().backward()
    assert torch.isfinite(probability.grad).all()
    assert probability.grad[1].item() == 0


def test_unknown_is_neither_an_occupancy_hit_nor_free_space():
    probability = torch.tensor([0.9, 0.6], requires_grad=True)
    known = torch.tensor([False, True])
    hints = SoftVoxelProjector()(predict(probability, known), camera(), transform(), 0)
    assert hints[0, 0, 0].expm1().item() == pytest.approx(2.5)
    torch.testing.assert_close(hints[0, 0, 1:], torch.tensor([0.6, 0.4, 0.5]))
    hints[0, 0, 1].backward()
    assert probability.grad[0].item() == 0
    assert probability.grad[1].abs().item() > 0


@pytest.mark.parametrize("known", [False, True])
def test_empty_and_unknown_rays_remain_finite_with_gradients(known):
    probability = torch.zeros(3, requires_grad=True)
    hints = SoftVoxelProjector()(
        predict(probability, torch.full((3,), known)), camera(), transform(), 0
    )
    torch.testing.assert_close(
        hints[0, 0], torch.tensor([0.0, 0.0, float(known), float(known)])
    )
    hints.sum().backward()
    assert torch.isfinite(probability.grad).all()
    if not known:
        assert not probability.grad.any()


def test_parallel_outside_ray_and_padded_patches_stay_zero():
    probability = torch.ones(2)
    projector = SoftVoxelProjector()
    grid = GridSpec((2, -0.5, 1), (1, 1, 1), (1, 1, 2))
    assert not projector(
        predict(probability, grid=grid), camera(), transform(), 0
    ).any()
    hints = projector(
        predict(probability),
        camera(),
        transform(target_height=42, pad_top=14, pad_bottom=14),
        0,
    )
    assert hints.shape == (1, 3, 4)
    assert not hints[0, [0, 2]].any()
    assert hints[0, 1, 1].item() == pytest.approx(1)


def test_geometry_cache_refreshes_grid_and_camera_but_not_predictions():
    projector = SoftVoxelProjector()
    first = predict(torch.tensor([0.5]))
    projector(first, camera(), transform(), 0)
    cached = projector._geometry
    fresh_probability = torch.tensor([0.2], requires_grad=True)
    second = projector(predict(fresh_probability), camera(), transform(), 0)
    assert projector._geometry is cached
    second[0, 0, 1].backward()
    assert fresh_probability.grad.abs().item() > 0
    farther = replace(first, grid=replace(first.grid, minimum=(-0.5, -0.5, 3)))
    result = projector(farther, camera(), transform(), 0)
    assert projector._geometry is not cached
    assert result[0, 0, 0].expm1().item() == pytest.approx(3.5)
    shifted_pose = np.eye(4)
    shifted_pose[2, 3] = 1
    result = projector(
        farther, replace(camera(), ego_from_camera=shifted_pose), transform(), 0
    )
    assert result[0, 0, 0].expm1().item() == pytest.approx(2.5)


def test_projection_budget_is_checked_before_allocating_ray_geometry():
    with pytest.raises(ValueError, match="budget"):
        SoftVoxelProjector(max_samples_per_ray=1)(
            predict(torch.zeros(2)), camera(), transform(), 0
        )
    with pytest.raises(ValueError, match="budget"):
        SoftVoxelProjector(max_rays=1)(
            predict(torch.zeros(2)), camera(), transform(target_width=28), 0
        )
