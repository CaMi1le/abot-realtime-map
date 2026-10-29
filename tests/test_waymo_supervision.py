# SPDX-License-Identifier: Apache-2.0
"""Offline export preserves native calibration and refuses corrupt/mixed data."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from realtime_map.waymo import WAYMO_FROM_CV, read_waymo


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_waymo_supervision", ROOT / "scripts/prepare-waymo-supervision.py"
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
DECODER = ROOT.parent / "reconstruction/scripts/waymo_numpy_adapter.py"


def source_frame():
    jpeg = io.BytesIO()
    Image.new("RGB", (16, 12), (70, 110, 150)).save(jpeg, format="JPEG")
    world_pose = np.eye(4)
    world_pose[:3, 3] = [100, 200, 300]
    frame = {
        "context": "test-context",
        "timestamp_micros": 1000000,
        "pose": world_pose,
        "cameras": {},
        "images": {},
    }
    for camera_id in (1, 2, 3):
        extrinsic = np.eye(4)
        extrinsic[1, 3] = camera_id / 10
        frame["cameras"][camera_id] = {
            "width": 16,
            "height": 12,
            "extrinsic": extrinsic,
            "intrinsic": [10.0, 11.0, 7.5, 5.5, 0.01, -0.02, 0.003, -0.004, 0.005],
            "serialized_sha256": "calibration-hash",
        }
        frame["images"][camera_id] = {"jpeg": jpeg.getvalue(), "pose": world_pose}
    return frame


def mock_decoder(monkeypatch, frames):
    def lidar(frame, ground):
        cloud = np.zeros((2, 14), np.float32)
        cloud[:, 3:6] = [[3, 0, 0], [4, 1, 1]]
        cloud[:, 10] = ground(cloud[:, 3:6])
        return cloud, []

    decoder = SimpleNamespace(
        records=lambda _: iter(
            (i, frame, f"payload-{i}") for i, frame in enumerate(frames)
        ),
        native_lidar=lidar,
    )
    monkeypatch.setattr(exporter, "load_decoder", lambda _: decoder)


def test_native_export_loads_fixed_calibration_and_measured_rays(tmp_path, monkeypatch):
    frame = source_frame()
    mock_decoder(monkeypatch, [frame])
    source = tmp_path / "source.tfrecord"
    source.write_bytes(b"immutable-source")
    output = tmp_path / "prepared"
    manifest = exporter.export_source(source, output, Path(__file__), 1)
    loaded = read_waymo(output)
    assert len(loaded) == 1 and len(loaded[0].cameras) == 3
    assert loaded[0].timestamp_us == frame["timestamp_micros"]
    for i, camera in enumerate(loaded[0].cameras, 1):
        np.testing.assert_array_equal(
            camera.ego_from_camera, frame["cameras"][i]["extrinsic"] @ WAYMO_FROM_CV
        )
        np.testing.assert_array_equal(
            camera.distortion, [0.01, -0.02, 0.003, -0.004, 0.005]
        )
        assert camera.path.read_bytes() == frame["images"][i]["jpeg"]
    lidar = manifest["frames"][0]["lidar"]
    cloud = np.fromfile(output / lidar["file"], "<f4").reshape(-1, 14)
    np.testing.assert_array_equal(cloud[:, 3:6], [[3, 0, 0], [4, 1, 1]])
    assert (cloud[:, 10] == -1).all()
    assert (
        manifest["source"]["file_sha256"]
        == hashlib.sha256(b"immutable-source").hexdigest()
    )
    assert source.read_bytes() == b"immutable-source"
    with pytest.raises(FileExistsError):
        exporter.export_source(source, output, Path(__file__))


def test_exported_scene_supports_waymo_trajectory_and_depth_evaluation(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from realtime_map.waymo_evaluation import evaluate_waymo

    frames, camera_poses = [], []
    for index, offset in enumerate(([0, 0, 0], [1, 0, 0], [1, 1, 0])):
        frame = source_frame()
        frame["timestamp_micros"] += index * 100000
        frame["pose"][:3, 3] += offset
        for image in frame["images"].values():
            image["pose"] = frame["pose"].copy()
        camera_poses.append(
            frame["images"][1]["pose"]
            @ frame["cameras"][1]["extrinsic"]
            @ WAYMO_FROM_CV
        )
        frames.append(frame)
    mock_decoder(monkeypatch, frames)
    source = tmp_path / "source.tfrecord"
    source.write_bytes(b"immutable-source")
    scene = tmp_path / "prepared"
    manifest = exporter.export_source(source, scene, Path(__file__), len(frames))
    assert all("source_world_from_camera" in row["sensors"]["FRONT"]
               for row in manifest["frames"])

    run = tmp_path / "run"
    (run / "frames").mkdir(parents=True)
    poses = np.stack(camera_poses)
    poses[:, :3, 3] /= 2
    np.save(run / "poses.npy", poses)
    inputs = []
    for index, frame in enumerate(frames):
        inputs.append({"context": frame["context"],
                       "timestamp": frame["timestamp_micros"] / 1e6})
        points = np.zeros((12, 16, 3), dtype=np.float32)
        points[..., 2] = 2
        np.savez(run / "frames" / f"{index:06d}.npz",
                 points=points, confidence=np.ones((12, 16), dtype=np.float32))
    (run / "input.json").write_text(json.dumps(inputs))

    result = evaluate_waymo(run, scene)
    assert result["trajectory_available"] and result["depth_available"]
    assert result["alignment_scale"] == pytest.approx(2)
    assert result["ate_rmse_m"] == pytest.approx(0, abs=1e-10)
    assert result["depth_samples"] == 6
    # Each frame has measured depths 3 and 4; aligned predicted depths are both 4.
    assert result["depth_rmse_m"] == pytest.approx(np.sqrt(0.5))
    assert result["depth_abs_rel"] == pytest.approx(1 / 6)


@pytest.mark.parametrize(
    "fault", ["context", "timestamp", "calibration", "short_source"]
)
def test_export_rejects_invalid_sequences_atomically(tmp_path, monkeypatch, fault):
    first = source_frame()
    second = copy.deepcopy(first)
    second["timestamp_micros"] += 100000
    if fault == "context":
        second["context"] = "another-context"
    elif fault == "timestamp":
        second["timestamp_micros"] = first["timestamp_micros"]
    elif fault == "calibration":
        second["cameras"][1]["extrinsic"][0, 3] = 0.5
    frames = [first] if fault == "short_source" else [first, second]
    mock_decoder(monkeypatch, frames)
    source = tmp_path / "source.tfrecord"
    source.write_bytes(b"source")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError):
        exporter.export_source(source, output, Path(__file__), 2)
    assert not output.exists()
    assert not list(tmp_path.glob("*.partial-*"))


def test_actual_decoder_rejects_corrupt_tfrecord_before_publication(tmp_path):
    if not DECODER.is_file():
        pytest.skip("Workspace Waymo decoder is an optional sibling dependency")
    source = tmp_path / "bad.tfrecord"
    source.write_bytes(b"bad CRC header")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="CRC32C"):
        exporter.export_source(source, output, DECODER)
    assert not output.exists()
