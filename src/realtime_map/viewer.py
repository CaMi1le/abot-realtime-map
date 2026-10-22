# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from scipy.spatial.transform import Rotation
import viser


class MapViewer:
    def __init__(self, port=8890, follow_camera=False, input_label='10 Hz input', render_size=(1280,800)):
        self.render_size = render_size
        self.follow_camera = follow_camera
        self.input_label = input_label
        title = 'Live driving reconstruction' if follow_camera else 'Live indoor mapping'
        self.server = viser.ViserServer(host="127.0.0.1", port=port, label=title)
        if self.server.get_port() != port: raise RuntimeError(f"Viewer port {port} is already in use")
        self.server.scene.set_up_direction("-y")
        self.server.gui.configure_theme(dark_mode=True, control_layout="floating")
        self.server.gui.add_markdown(f"# {title}\nRGB only · **scale uncalibrated**")
        self.status = self.server.gui.add_markdown("Waiting for model…")
        self.input_image = self.server.gui.add_image(np.zeros((280, 504, 3), np.uint8), label="Camera input", format="jpeg")
        self.clouds = []; self.pending_xyz = []; self.pending_rgb = []
        self.chunk = 0; self.trajectory = []
        self.center = np.array([0., 0., .3]); self.extent = .5; self.point_size = .001
        self.bounds_min = self.bounds_max = None
        self.follow_distance = None
        self.auto_fit = self.server.gui.add_checkbox("Follow camera" if follow_camera else "Fit growing map", True)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.render_futures = []
        self.server.on_client_connect(self._connect)

    def _connect(self, client):
        client.camera.position = self.center + self.extent * np.array([.45, -.4, -.85])
        client.camera.look_at = self.center
        client.camera.up_direction = (0.0, -1.0, 0.0)

    def wait_client(self, timeout=120):
        start = time.monotonic()
        while not self.server.get_clients():
            if time.monotonic() - start > timeout:
                raise TimeoutError("No browser connected for browser-latency measurement")
            time.sleep(.2)

    def update(self, result, added_xyz, added_rgb, voxel_map, rebuilt, lag_s):
        self.point_size = voxel_map.voxel * .6
        if len(added_xyz):
            lo, hi = np.min(added_xyz,axis=0), np.max(added_xyz,axis=0)
            self.bounds_min = lo if self.bounds_min is None else np.minimum(self.bounds_min,lo)
            self.bounds_max = hi if self.bounds_max is None else np.maximum(self.bounds_max,hi)
            span = float(np.max(self.bounds_max-self.bounds_min))
            if not self.follow_camera and self.trajectory and self.auto_fit.value and span > self.extent*1.5:
                self.center = (self.bounds_min+self.bounds_max)/2
                self.extent = span
                for client in self.server.get_clients().values(): self._connect(client)
        if not self.trajectory and len(added_xyz):
            self.center = np.median(added_xyz, axis=0)
            self.extent = max(float(np.max(np.ptp(added_xyz, axis=0))), voxel_map.voxel * 100)
            for client in self.server.get_clients().values(): self._connect(client)
        if rebuilt:
            for handle in self.clouds: handle.remove()
            self.clouds = []; self.pending_xyz = []; self.pending_rgb = []; self.chunk = 0
            added_xyz, added_rgb = voxel_map.arrays()
        self.pending_xyz.append(added_xyz); self.pending_rgb.append(added_rgb)
        xyz = np.concatenate(self.pending_xyz); rgb = np.concatenate(self.pending_rgb)
        if len(xyz):
            handle = self.server.scene.add_point_cloud(f"/map/chunk-{self.chunk}", xyz, rgb,
                point_size=self.point_size, point_shape="circle", precision="float32")
            if len(self.clouds) <= self.chunk: self.clouds.append(handle)
            else: self.clouds[self.chunk] = handle
        if len(xyz) >= 20000:
            self.pending_xyz = []; self.pending_rgb = []; self.chunk += 1
        self.trajectory.append(result.camera_pose[:3, 3].copy())
        poses = np.asarray(self.trajectory)
        if len(poses) > 1:
            self.server.scene.add_line_segments("/trajectory", np.stack([poses[:-1], poses[1:]], 1),
                                                 (80, 215, 255), thickness=2, thickness_units="screen")
        q = Rotation.from_matrix(result.camera_pose[:3, :3]).as_quat()
        if self.follow_camera and self.auto_fit.value:
            # Camera-local chase view: visible road stays a useful size on long routes.
            if self.follow_distance is None: self.follow_distance = voxel_map.voxel * 150
            distance = self.follow_distance
            pose = result.camera_pose
            self.extent = distance
            for client in self.server.get_clients().values():
                client.camera.position = pose[:3, 3] + pose[:3, :3] @ np.array([distance*.5, -distance*.7, -distance*.8])
                client.camera.look_at = pose[:3, 3] + pose[:3, :3] @ np.array([0, 0, distance*.7])
                client.camera.up_direction = tuple(-pose[:3, 1])
        self.server.scene.add_camera_frustum("/camera", fov=1.05, aspect=504/280,
            thickness=1.5, thickness_units="screen",
            scale=self.extent*.08, color=(255, 180, 60), wxyz=q[[3, 0, 1, 2]], position=poses[-1])
        self.input_image.image = result.colors[result.valid_rows]
        self.status.content = (f"**Frame {result.frame_id}** · {len(voxel_map.cells):,} map points\n\n"
            f"Model: {result.model_ms:.1f} ms · server lag: {lag_s:.3f} s\n\n"
            f"{self.input_label} · voxel {voxel_map.voxel:.5f} model units")
        self.server.flush()

    def request_ack(self, frame_id, scheduled, sink):
        # One render in flight. Bounded backpressure is included in the measured
        # end-to-end path; no silently skipped measurement frames.
        if self.render_futures:
            self.render_futures[-1].result()
        clients = self.server.get_clients()
        if not clients:
            raise RuntimeError("Browser disconnected during measured replay")
        client = next(iter(clients.values()))
        def ack():
            try:
                client.get_render(height=self.render_size[1], width=self.render_size[0], timeout=5)
                sink.append({"frame_id": frame_id, "latency_s": time.perf_counter() - scheduled})
            except Exception as exc:
                sink.append({"frame_id": frame_id, "error": type(exc).__name__ + ": " + str(exc)})
        future = self.executor.submit(ack)
        self.render_futures = [future]

    def finish(self):
        self.executor.shutdown(wait=True)

    def add_refined(self, xyz, rgb, poses):
        corrected = self.server.scene.add_point_cloud("/corrected", xyz, rgb, point_size=self.point_size,
                                                     point_shape="circle", visible=False)
        selector = self.server.gui.add_dropdown("Trajectory", ("Raw streaming", "After loop closure"))
        @selector.on_update
        def change(_):
            refined = selector.value == "After loop closure"
            corrected.visible = refined
            for cloud in self.clouds: cloud.visible = not refined
            points = poses[:, :3, 3] if refined else np.asarray(self.trajectory)
            self.server.scene.add_line_segments("/trajectory", np.stack([points[:-1], points[1:]], 1),
                                               (80, 215, 255), thickness=2, thickness_units="screen")
