# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import psutil

from .backend import ABotStream
from .data import read_tum, repeat_frames
from .mapping import MapChunks, VoxelMap, filtered_points, write_ply, write_trajectory


def summary(values):
    if not values: return None
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    dataset_format = getattr(args, 'dataset_format', 'tum')
    if dataset_format == 'kitti':
        from .kitti import read_kitti
        frames = read_kitti(Path(args.input), args.max_frames)
    elif dataset_format == 'waymo':
        from .waymo import read_waymo
        frames = read_waymo(args.input, args.max_frames, getattr(args, 'waymo_context', None))
    else:
        frames = read_tum(Path(args.input), args.fps, args.max_frames)
    if args.stress_seconds:
        frames = list(repeat_frames(frames, args.stress_seconds))
    inputs = ([{"frame_id":f.frame_id,"timestamp":f.timestamp,"path":str(f.path),
                "context":f.context,"cameras":[c.name for c in f.cameras]} for f in frames]
              if dataset_format == 'waymo' else [asdict(f) for f in frames])
    (out / "input.json").write_text(json.dumps(inputs, indent=2, default=str))
    viewer = None
    if args.viewer:
        from .viewer import MapViewer
        viewer = MapViewer(args.port, follow_camera=getattr(args, 'follow_camera', False),
                           render_size=(args.render_width,args.render_height),
                           input_label='Native KITTI cadence · all frames' if dataset_format == 'kitti' else f'{args.fps:g} Hz input')
    backend_options = {"max_frames":max(22000, len(frames)+64)}
    if getattr(args, 'device', 'cuda') != 'cuda':
        backend_options['device'] = args.device
    occ_runtime = None
    if getattr(args, 'occ_checkpoint', None):
        from .occ_model import OccRuntime
        occ_runtime = OccRuntime.from_checkpoints(args.occ_checkpoint, args.occ_adapter,
                         getattr(args, 'device', 'cuda'), args.occ_mode)
        backend_options['occ_runtime'] = occ_runtime
    backend = ABotStream(args.checkpoint, args.attention, **backend_options)
    execution_device = getattr(args, 'device', 'cuda')
    cuda_active = execution_device.startswith('cuda') and getattr(backend.torch.cuda, 'is_available', lambda:False)()
    (out / "hardware.json").write_text(json.dumps({
        "hostname": __import__('socket').gethostname(),
        "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES'),
        "gpu": backend.torch.cuda.get_device_name(execution_device) if cuda_active else 'CPU',
        "torch": getattr(backend.torch, '__version__', 'test'),
        "input_shape": [280, 504], "amp_dtype": str(getattr(backend.wrapper, 'compute_dtype', 'test')),
        "source_commit": "6e3b2e1257464a74dfceff22eb33190bd228276e",
    }, indent=2))
    # Kernel warm-up is untimed and MUST NOT seed the evaluated trajectory.
    initial_voxel = args.voxel
    for i, f in enumerate(frames[:min(12, len(frames))]):
        warm_result = backend.step(f)
        if i == 0 and not initial_voxel:
            depth = warm_result.local_points[warm_result.valid_rows, :, 2]
            depth = depth[np.isfinite(depth) & (depth > 0)]
            if not len(depth): raise ValueError("First frame has no valid depth for automatic voxel size")
            initial_voxel = float(np.median(depth) / 150)
    del warm_result
    backend.reset()
    if cuda_active:
        backend.torch.cuda.reset_peak_memory_stats(execution_device)
    if viewer and args.measure_browser:
        viewer.status.content = "Model ready. Waiting for browser…"
        viewer.wait_client()
        time.sleep(2)
    voxel_map = VoxelMap(initial_voxel, args.max_points)
    chunks = MapChunks(out / 'map-chunks', initial_voxel, args.max_points, args.chunk_frames) if getattr(args, 'chunk_frames', 0) else None
    source_fps = (len(frames)-1)/(frames[-1].timestamp-frames[0].timestamp) if len(frames)>1 else args.fps
    poses = []; rows = []; acks = []; error = None
    started = time.perf_counter()
    (out / "started.json").write_text(json.dumps({"wall_time": time.time(), "perf_counter": started}))
    try:
        with (out / "timings.jsonl").open("w", buffering=1) as log:
            for i, frame in enumerate(frames):
                scheduled = started + frame.timestamp - frames[0].timestamp
                if args.realtime:
                    time.sleep(max(0, scheduled - time.perf_counter()))
                arrival = time.perf_counter()
                result = backend.step(frame)
                inference_done = time.perf_counter()
                poses.append(result.camera_pose)
                xyz, rgb = filtered_points(result, args.confidence, args.point_stride)
                added_xyz, added_rgb, rebuilt = voxel_map.add(xyz, rgb)
                preview_map = voxel_map
                if chunks:
                    chunk_xyz, chunk_rgb, chunk_rebuilt = chunks.add(i, xyz, rgb)
                    if args.follow_camera:
                        added_xyz, added_rgb, rebuilt = chunk_xyz, chunk_rgb, chunk_rebuilt
                        preview_map = chunks.current
                mapping_done = time.perf_counter()
                if not args.no_dense:
                    # Preserve dense local observations and their RGB/confidence;
                    # storage grows on disk, never as a list of CUDA tensors.
                    np.savez(out / "frames" / f"{i:06d}.npz", points=result.local_points,
                        colors=result.colors, confidence=result.confidence.astype(np.float16),
                        pose=result.camera_pose, timestamp=result.timestamp,
                        valid_rows=np.array([result.valid_rows.start, result.valid_rows.stop]))
                storage_done = time.perf_counter()
                lag = time.perf_counter() - scheduled
                viewer_done = storage_done
                if viewer:
                    viewer.update(result, added_xyz, added_rgb, preview_map, rebuilt, lag)
                    viewer_done = time.perf_counter()
                    if args.measure_browser: viewer.request_ack(i, scheduled, acks)
                row = {"frame_id": i, "timestamp": frame.timestamp, "model_ms": result.model_ms,
                       "occ_predict_ms": result.occ_predict_ms, "occ_project_ms": result.occ_project_ms,
                       "preprocess_ms": result.preprocess_ms,
                       "inference_wall_ms": (inference_done-arrival)*1000,
                       "mapping_ms": (mapping_done-inference_done)*1000,
                       "storage_ms": (storage_done-mapping_done)*1000,
                       "viewer_update_ms": (viewer_done-storage_done)*1000,
                       "browser_backpressure_ms": (time.perf_counter()-viewer_done)*1000,
                       "server_latency_s": time.perf_counter() - scheduled,
                       "processing_ms": (time.perf_counter() - arrival) * 1000,
                       "backlog_frames": max(0, (arrival - scheduled) * source_fps),
                       "gpu_allocated_mib": backend.torch.cuda.memory_allocated(execution_device)/2**20 if cuda_active else 0.,
                       "gpu_reserved_mib": backend.torch.cuda.memory_reserved(execution_device)/2**20 if cuda_active else 0.,
                       "cpu_rss_mib": psutil.Process().memory_info().rss/2**20,
                       "map_points": len(voxel_map.cells)}
                rows.append(row); log.write(json.dumps(row) + "\n")
                if i % 50 == 0:
                    print(f"frame={i}/{len(frames)} model={result.model_ms:.1f}ms lag={row['server_latency_s']:.2f}s points={len(voxel_map.cells)}", flush=True)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if viewer: viewer.finish()
        backend.reset()
        if chunks: chunks.flush()
        np.save(out / "poses.npy", np.asarray(poses))
        np.savetxt(out / "timestamps.txt", [f.timestamp for f in frames[:len(poses)]])
        write_trajectory(out / "trajectory.txt", [f.timestamp for f in frames[:len(poses)]], poses)
        xyz, rgb = voxel_map.arrays(); write_ply(out / "map.ply", xyz, rgb)
        np.savez(out / "map.npz", points=xyz, colors=rgb)
        (out / "browser-acks.json").write_text(json.dumps(acks, indent=2))
        warm = min(12, len(rows))
        valid = rows[warm:]
        browser = [a["latency_s"] for a in acks if a["frame_id"] >= warm and "latency_s" in a]
        completed = len(rows) == len(frames) and error is None
        browser_ok = (args.measure_browser and completed and len(browser) == len(valid)
                      and bool(browser) and np.percentile(browser, 95) < 1)
        report = {"completed": completed, "error": error, "frames": len(rows), "expected_frames": len(frames),
                  "wall_seconds": time.perf_counter()-started, "target_fps": args.fps,
                  "dataset_format": dataset_format, "source_fps": source_fps,
                  "sampling": "all native timestamps; no frames dropped" if dataset_format in ('kitti','waymo') else 'timestamp sampling',
                  "occ": {"enabled":occ_runtime is not None,
                          "mode":getattr(args,'occ_mode','predicted') if occ_runtime else None,
                          "cameras":list(occ_runtime.source.cameras) if occ_runtime else [],
                          "online_prediction_in_timing":bool(occ_runtime),
                          "fusion_timing":"included in model_ms; no additional ABot forward"},
                  "attention_backend": backend.wrapper.attention_backend,
                  "scale": "uncalibrated", "warmup_frames_excluded": warm,
                  "synthetic_repeat_memory_test": bool(args.stress_seconds),
                  "model_ms": summary([r["model_ms"] for r in valid]),
                  "processing_ms": summary([r["processing_ms"] for r in valid]),
                  "stage_ms": {key: summary([r[key] for r in valid]) for key in
                               ['preprocess_ms','occ_predict_ms','occ_project_ms','inference_wall_ms','mapping_ms','storage_ms','viewer_update_ms','browser_backpressure_ms']},
                  "backlog_frames": summary([r["backlog_frames"] for r in valid]),
                  "cpu_rss_mib": summary([r["cpu_rss_mib"] for r in valid]),
                  "server_latency_s": summary([r["server_latency_s"] for r in valid]) if args.realtime else None,
                  "browser_latency_upper_bound_s": summary(browser),
                  "browser_measurement": "server-scheduled arrival through rendered browser image response; includes return-trip time",
                  "browser_render_resolution": [args.render_width, args.render_height] if args.measure_browser else None,
                  "browser_ack_errors": [a for a in acks if "error" in a],
                  "native_cadence_realtime_pass": bool(browser_ok and args.realtime),
                  "realtime_pass": bool(browser_ok and args.realtime and source_fps >= 9.99),
                  "peak_gpu_mib": backend.torch.cuda.max_memory_allocated(execution_device)/2**20 if cuda_active else 0.,
                  "preview_points": len(xyz), "initial_voxel_model_units": initial_voxel,
                  "final_voxel_model_units": voxel_map.voxel,
                  "archived_map_chunks": len(chunks.entries) if chunks else 0}
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    if completed and args.loop:
        from .evaluation import refine
        corrected, corrected_map = refine(backend, frames, np.asarray(poses), out, args)
        if viewer: viewer.add_refined(*corrected_map, corrected)
    if viewer and args.keep_open:
        outcome = 'Complete. Results saved.' if completed else f'Incomplete: {error}. Partial results saved.'
        viewer.status.content = f"**Finished: {len(rows)}/{len(frames)} frames**\n\nScale uncalibrated. {outcome}"
        print("Viewer stays available; Ctrl+C stops only this viewer.", flush=True)
        while True: time.sleep(1)
    if error: raise RuntimeError(error)
