#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch a bounded, CRC-checked prefix of ONE official Waymo context.

Reuses the workspace's TensorFlow-free decoder. Exports original front-three
JPEG bytes and fixed calibration; world poses remain offline metadata only.
Requires existing gcloud authorization for the official dataset bucket.
"""
import argparse
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image


def prepare(args):
    if not 1 <= args.frames <= 200 or args.read_cap_mib <= 0:
        raise ValueError("Use 1–200 frames and a positive source read cap")
    if not args.source.startswith("gs://waymo_open_dataset_v_"):
        raise ValueError("Expected an official Waymo dataset source URI")
    spec = importlib.util.spec_from_file_location("waymo_decoder", args.decoder)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    auth = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
    if auth.returncode or not auth.stdout.strip():
        raise RuntimeError("Existing gcloud source authorization is unavailable")
    bucket, key = args.source[5:].split("/", 1)
    url = ("https://storage.googleapis.com/storage/v1/b/" + bucket + "/o/"
           + urllib.parse.quote(key, safe="") + "?alt=media&ifGenerationMatch="
           + urllib.parse.quote(args.generation, safe=""))
    cap = args.read_cap_mib * 1024**2
    request = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + auth.stdout.strip(), "Range": f"bytes=0-{cap-1}"})
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    (out / "extrinsics").mkdir()
    rows, receipts, context, previous = [], [], None, None
    consumed = 0
    with urllib.request.urlopen(request, timeout=60) as response, (out / "source-prefix.tfrecord").open("wb") as saved:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith("bytes 0-"):
            raise ValueError("Provider did not honor bounded source range")
        if response.headers.get("x-goog-generation") != args.generation:
            raise ValueError("Source object generation mismatch")
        content_range = response.headers["Content-Range"]
        for index in range(args.frames):
            header = response.read(12)
            consumed += len(header)
            if len(header) != 12 or decoder.crc32c(header[:8]) != struct.unpack("<I", header[8:])[0]:
                raise ValueError("Truncated source prefix or invalid TFRecord header CRC32C")
            length = struct.unpack("<Q", header[:8])[0]
            if not 0 < length <= 64 * 1024**2 or consumed + length + 4 > cap:
                raise ValueError("Record exceeds source read cap; increase --read-cap-mib")
            payload = response.read(length)
            trailer = response.read(4)
            consumed += len(payload) + len(trailer)
            if len(payload) != length or len(trailer) != 4 or decoder.crc32c(payload) != struct.unpack("<I", trailer)[0]:
                raise ValueError("Truncated source prefix or invalid TFRecord payload CRC32C")
            frame = decoder.parse_frame(payload)
            context = context or frame["context"]
            stamp = frame["timestamp_micros"]
            if context != frame["context"] or (previous is not None and stamp <= previous):
                raise ValueError("Expected increasing timestamps within one official context")
            previous = stamp
            sensors = {}
            for camera_id, name in enumerate(("FRONT", "FRONT_LEFT", "FRONT_RIGHT")):
                calibration, image = frame["cameras"][camera_id + 1], frame["images"][camera_id + 1]
                jpeg = image["jpeg"]
                with Image.open(io.BytesIO(jpeg)) as decoded:
                    if decoded.size != (calibration["width"], calibration["height"]):
                        raise ValueError("Original image dimensions disagree with calibration")
                    decoded.verify()
                relative = f"images/{index:06d}_{camera_id}.jpg"
                (out / relative).write_bytes(jpeg)
                extrinsic_file = out / "extrinsics" / f"{camera_id}.txt"
                if index == 0:
                    np.savetxt(extrinsic_file, calibration["extrinsic"], fmt="%.17g")
                elif not np.allclose(np.loadtxt(extrinsic_file), calibration["extrinsic"], atol=1e-12, rtol=0):
                    raise ValueError("Fixed camera extrinsic changed")
                sensors[name] = {
                    "camera_id": camera_id, "image": relative,
                    "width": calibration["width"], "height": calibration["height"],
                    "image_intrinsic": dict(zip(
                        ("f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3"),
                        calibration["intrinsic"], strict=True)),
                    "source_jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                    "source_calibration_sha256": calibration["serialized_sha256"],
                    "source_world_from_camera": (image["pose"] @ calibration["extrinsic"]).reshape(-1).tolist(),
                }
            rows.append({"frame_index": index, "context": context, "timestamp_micros": stamp,
                         "source_world_from_vehicle": frame["pose"].reshape(-1).tolist(), "sensors": sensors})
            record = header + payload + trailer
            saved.write(record)
            receipts.append({"index": index, "offset": consumed - len(record), "record_bytes": len(record),
                             "sha256": hashlib.sha256(record).hexdigest(), "crc32c_verified": True})
            print(f"frame={index+1}/{args.frames} source_MiB={consumed / 1024**2:.1f}", flush=True)
    manifest = {"dataset": "Waymo Perception v1", "context": context, "frames": rows,
                "cameras": ["FRONT", "FRONT_LEFT", "FRONT_RIGHT"],
                "pixel_processing": "Original source JPEG bytes; no image resize or distortion change",
                "offline_metadata": "World poses are retained for offline use; read_waymo inference uses fixed extrinsics only",
                "source": {"uri": args.source, "generation": args.generation,
                           "requested_range": f"bytes=0-{cap-1}", "provider_content_range": content_range,
                           "consumed_bytes": consumed, "records": receipts,
                           "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}}
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"context": context, "frames": len(rows), "images": len(rows)*3,
                      "span_seconds": (rows[-1]["timestamp_micros"]-rows[0]["timestamp_micros"])/1e6,
                      "source_bytes": consumed, "output": str(out)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--read-cap-mib", type=int, default=128)
    parser.add_argument("--decoder", type=Path, default=Path(__file__).resolve().parents[2] / "reconstruction/scripts/waymo_numpy_adapter.py")
    prepare(parser.parse_args())
