#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export local, CRC-checked Waymo records for camera-only OCC/ABot training.

Original FRONT/FRONT_LEFT/FRONT_RIGHT JPEGs and fixed calibration are retained.
Measured LiDAR rays and world poses are offline supervision, never online input.
The source is immutable; publication is atomic and existing output is refused.
"""

import argparse
import datetime
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image


CAMERAS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT")
INTRINSIC_NAMES = ("f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_decoder(path):
    spec = importlib.util.spec_from_file_location("waymo_supervision_decoder", path)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    return decoder


def unknown_ground(points):
    # Ground and scene-flow labels are not used by the training objectives.
    return np.full(len(points), -1, dtype=np.float32)


def export_source(source, output, decoder_path, max_frames=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if max_frames is not None and max_frames < 1:
        raise ValueError("--max-frames must be positive")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    decoder = load_decoder(decoder_path)
    started = time.perf_counter()
    source_hash = sha256_file(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output.name + ".partial-", dir=output.parent
    ) as temporary:
        work = Path(temporary) / "export"
        for directory in ("images", "extrinsics", "lidar"):
            (work / directory).mkdir(parents=True)
        rows, receipts, fixed_extrinsics = [], [], {}
        context, previous = None, None
        records = decoder.records(source)
        if max_frames is not None:
            records = itertools.islice(records, max_frames)
        for source_index, frame, payload_sha256 in records:
            context = context or frame["context"]
            timestamp = int(frame["timestamp_micros"])
            if context != frame["context"]:
                raise ValueError(
                    "Each prepared root must contain exactly one official context"
                )
            if previous is not None and timestamp <= previous:
                raise ValueError(
                    "Source timestamps must increase strictly within the context"
                )
            previous = timestamp
            index = len(rows)
            sensors = {}
            for camera_id, name in enumerate(CAMERAS):
                calibration = frame["cameras"][camera_id + 1]
                image = frame["images"][camera_id + 1]
                jpeg = image["jpeg"]
                with Image.open(io.BytesIO(jpeg)) as decoded:
                    if decoded.size != (calibration["width"], calibration["height"]):
                        raise ValueError(
                            "Original image dimensions disagree with calibration"
                        )
                    decoded.verify()
                extrinsic = calibration["extrinsic"]
                if camera_id not in fixed_extrinsics:
                    fixed_extrinsics[camera_id] = extrinsic.copy()
                    np.savetxt(
                        work / "extrinsics" / f"{camera_id}.txt", extrinsic, fmt="%.17g"
                    )
                elif not np.array_equal(fixed_extrinsics[camera_id], extrinsic):
                    raise ValueError("Fixed camera extrinsic changed within a context")
                relative = f"images/{index:06d}_{camera_id}.jpg"
                (work / relative).write_bytes(jpeg)
                sensors[name] = {
                    "camera_id": camera_id,
                    "image": relative,
                    "width": calibration["width"],
                    "height": calibration["height"],
                    "image_intrinsic": dict(
                        zip(INTRINSIC_NAMES, calibration["intrinsic"], strict=True)
                    ),
                    "source_jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                    "source_calibration_sha256": calibration["serialized_sha256"],
                    "source_world_from_camera": (image["pose"] @ extrinsic)
                    .reshape(-1)
                    .tolist(),
                    "source_image_timing": {
                        key: image[key]
                        for key in (
                            "pose_timestamp_seconds",
                            "shutter",
                            "camera_trigger_time",
                            "camera_readout_done_time",
                        )
                        if key in image
                    },
                }
            lidar_started = time.perf_counter()
            cloud, laser_receipts = decoder.native_lidar(frame, unknown_ground)
            cloud = np.asarray(cloud, dtype="<f4")
            if (
                cloud.ndim != 2
                or cloud.shape[1] != 14
                or not len(cloud)
                or not np.isfinite(cloud).all()
            ):
                raise ValueError(
                    "LiDAR conversion must contain finite, measured Nx14 rays"
                )
            lidar_seconds = time.perf_counter() - lidar_started
            relative = f"lidar/{index:06d}.bin"
            cloud.tofile(work / relative)
            rows.append(
                {
                    "frame_index": index,
                    "source_record_index": source_index,
                    "context": context,
                    "timestamp_micros": timestamp,
                    "source_world_from_vehicle": frame["pose"].reshape(-1).tolist(),
                    "sensors": sensors,
                    "lidar": {
                        "file": relative,
                        "points": len(cloud),
                        "sha256": sha256_file(work / relative),
                        "dtype": "little-endian float32",
                        "shape": [len(cloud), 14],
                        "coordinate_frame": "vehicle at source frame timestamp; TOP per-pixel deskew retained",
                        "columns": [
                            "origin_x",
                            "origin_y",
                            "origin_z",
                            "point_x",
                            "point_y",
                            "point_z",
                            "flow_x",
                            "flow_y",
                            "flow_z",
                            "flow_label",
                            "ground_label",
                            "intensity",
                            "elongation",
                            "laser_id",
                        ],
                        "ground_label": "-1 unknown; not computed or used in training",
                        "returns": "first return from all five calibrated LiDAR sensors",
                        "laser_receipts": laser_receipts,
                        "decode_seconds": lidar_seconds,
                    },
                }
            )
            receipts.append(
                {
                    "index": source_index,
                    "payload_sha256": payload_sha256,
                    "crc32c_verified": True,
                }
            )
            print(
                json.dumps(
                    {
                        "frame": index + 1,
                        "lidar_points": len(cloud),
                        "lidar_decode_seconds": lidar_seconds,
                    }
                ),
                flush=True,
            )
        if not rows:
            raise ValueError("Source contains no complete Waymo records")
        if max_frames is not None and len(rows) != max_frames:
            raise ValueError(
                f"Requested {max_frames} records but source contains only {len(rows)}"
            )
        # The source hash also guards against concurrent replacement during export.
        if sha256_file(source) != source_hash:
            raise ValueError("Source changed during export")
        elapsed = time.perf_counter() - started
        manifest = {
            "dataset": "Waymo Perception v1",
            "context": context,
            "cameras": list(CAMERAS),
            "frames": rows,
            "pixel_processing": "Original source JPEG bytes; no image resize or distortion change",
            "offline_metadata": "LiDAR and world poses are offline supervision only; read_waymo exposes images and fixed calibration",
            "source": {
                "local_tfrecord": str(source),
                "file_sha256": source_hash,
                "file_bytes": source.stat().st_size,
                "records": receipts,
                "decoder": str(Path(decoder_path).resolve()),
                "decoder_sha256": sha256_file(decoder_path),
                "exported_at_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            },
            "export_seconds": elapsed,
        }
        source_manifest = source.parent / "source-manifest.json"
        if source_manifest.is_file():
            # Preserve generation-pinned download evidence without changing the source.
            provenance = json.loads(source_manifest.read_text())
            manifest["source"]["upstream"] = provenance.get("source", {})
        (work / "source-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        work.rename(output)
    summary = {
        "context": context,
        "frames": len(rows),
        "images": len(rows) * len(CAMERAS),
        "lidar_points": sum(r["lidar"]["points"] for r in rows),
        "span_seconds": (rows[-1]["timestamp_micros"] - rows[0]["timestamp_micros"])
        / 1e6,
        "export_seconds": elapsed,
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Existing local TFRecord or validated prefix",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New prepared root; must not exist"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Export exactly this many initial records; otherwise export all",
    )
    parser.add_argument(
        "--decoder",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reconstruction/scripts/waymo_numpy_adapter.py",
    )
    args = parser.parse_args()
    export_source(args.source, args.output, args.decoder, args.max_frames)
