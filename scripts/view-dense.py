#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inspect saved dense ABot observations without voxel overview coarsening."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import viser

from realtime_map.mapping import transform_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--port", type=int, default=8893)
    args = parser.parse_args()
    files = sorted((args.result / "frames").glob("*.npz"))
    if not files:
        raise ValueError("Dense observations are missing; replay without --no-dense")
    report = json.loads((args.result / "report.json").read_text())
    reference = report["initial_voxel_model_units"] * 150
    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="ABot dense observations")
    if server.get_port() != args.port:
        raise RuntimeError("Requested port is already occupied")
    server.scene.set_up_direction("-y")
    server.gui.configure_theme(dark_mode=True)
    server.gui.add_markdown(
        "# ABot dense observations\nSaved inference · **scale uncalibrated**\n\n"
        "Inspect local geometry before voxel merging. Sampling and point diameter "
        "change display only; the model output remains 504 × 280. "
        "Accumulated dynamic objects can leave trails.")
    mode = server.gui.add_dropdown("Geometry", ("Dense observations", "Voxel overview"))
    frame = server.gui.add_slider("Last frame", min=0, max=max(1, len(files)-1), step=1,
                                  initial_value=(len(files)-1)//2)
    window = server.gui.add_slider("Trailing frames", min=1, max=min(10, max(2, len(files))), step=1, initial_value=1)
    stride = server.gui.add_slider("Pixel stride", min=1, max=4, step=1, initial_value=1)
    confidence = server.gui.add_slider("Confidence threshold", min=0, max=1, step=.05, initial_value=.5)
    far = server.gui.add_slider("Far depth / reference", min=1, max=10, step=.5, initial_value=3)
    diameter = server.gui.add_slider("Point diameter (% reference)", min=.02, max=1, step=.02, initial_value=.2)
    reset = server.gui.add_button("Reset local view")
    info = server.gui.add_markdown("")
    source = server.gui.add_image(np.zeros((280, 504, 3), np.uint8), label="RGB seen by ABot", format="jpeg")
    state = {}

    def fit(client):
        pose = state["pose"]
        client.camera.position = pose[:3, 3] + pose[:3, :3] @ (reference * np.array([-.45, -.22, -.25]))
        client.camera.look_at = pose[:3, 3] + pose[:3, :3] @ (reference * np.array([.05, .1, .8]))
        client.camera.up_direction = tuple(-pose[:3, 1])
        client.camera.fov = .85

    @server.on_client_connect
    def connect(client):
        if state:
            fit(client)

    def update(_=None):
        last = min(int(frame.value), len(files)-1)
        first = max(0, last-int(window.value)+1)
        with np.load(files[last]) as saved:
            state["pose"] = saved["pose"]
            source.image = saved["colors"]
        if mode.value == "Voxel overview":
            with np.load(args.result / "map.npz") as saved:
                xyz, rgb = saved["points"], saved["colors"]
            description = "Voxel overview · pixel stride, confidence and range controls apply to dense observations"
        else:
            points, colors = [], []
            for path in files[first:last+1]:
                with np.load(path) as saved:
                    start, stop = saved["valid_rows"].astype(int)
                    grid = (slice(start, stop, int(stride.value)), slice(None, None, int(stride.value)))
                    local = saved["points"][grid].reshape(-1, 3)
                    rgb = saved["colors"][grid].reshape(-1, 3)
                    conf = saved["confidence"][grid].reshape(-1)
                    valid = (np.isfinite(local).all(1) & np.isfinite(conf)
                             & (local[:, 2] > 0) & (local[:, 2] <= reference*far.value)
                             & (conf >= confidence.value))
                    points.append(transform_points(local[valid], saved["pose"]))
                    colors.append(rgb[valid])
            xyz, rgb = np.concatenate(points), np.concatenate(colors)
            description = f"Frames {first}–{last} · stride {int(stride.value)} · confidence ≥ {confidence.value:.2f}"
        handle = server.scene.add_point_cloud(
            "/observations", xyz, rgb, point_size=reference*diameter.value/100,
            point_shape="circle", precision="float32", point_shading="flat")
        state["cloud"] = handle
        info.content = (f"**{len(xyz):,} points**\n\n{description}\n\n"
                        f"Reference depth: {reference:.4g} model units · no metric scale calibration\n\n"
                        "Orbit: drag · pan: right drag · zoom: scroll. Reducing point diameter adds no geometry.")

    def resize(_):
        if "cloud" in state:
            state["cloud"].point_size = reference*diameter.value/100

    def reset_view(_):
        for client in server.get_clients().values():
            fit(client)

    for control in (mode, frame, window, stride, confidence, far):
        control.on_update(update)
    diameter.on_update(resize)
    reset.on_click(reset_view)
    update()
    print(f"Inspect dense observations: http://127.0.0.1:{server.get_port()}", flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
