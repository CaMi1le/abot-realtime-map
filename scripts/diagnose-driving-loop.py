# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Inspect saved loop constraints against GT, without changing reconstruction.

Pair errors use inv(GT relative pose) @ estimated relative pose. The two
estimated variants each retain their existing global Sim(3) alignment.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def diagnose(root):
    root = Path(root)
    gt = np.load(root / "gt-camera2.npy")
    variants = {"raw": np.load(root / "poses-aligned.npy"),
                "offline_loop": np.load(root / "poses-aligned-loop.npy")}
    edges = json.loads((root / "loop/loop_edges.json").read_text())
    edges.sort(key=lambda edge: edge["src_pos"])
    info = json.loads((root / "loop/loop_info.json").read_text())
    times = np.loadtxt(root / "timestamps.txt")
    times -= times[0]
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1))]
    error = {name: np.linalg.norm(poses[:, :3, 3] - gt[:, :3, 3], axis=1)
             for name, poses in variants.items()}
    pairs = []
    for edge in edges:
        i, j = edge["src_pos"], edge["dst_pos"]
        truth = np.linalg.inv(gt[j]) @ gt[i]
        pair = {"source_frame": i, "destination_frame": j,
                "source_time_s": float(times[i]), "destination_time_s": float(times[j]),
                "source_distance_m": float(distance[i]), "destination_distance_m": float(distance[j]),
                "gt_camera_separation_m": float(np.linalg.norm(truth[:3, 3])),
                "retrieval_score": edge["score"]}
        for name, poses in variants.items():
            relative = np.linalg.inv(poses[j]) @ poses[i]
            residual = np.linalg.inv(truth) @ relative
            pair[name] = {"translation_error_m": float(np.linalg.norm(residual[:3, 3])),
                          "rotation_error_deg": float(np.degrees(Rotation.from_matrix(residual[:3, :3]).magnitude()))}
        pairs.append(pair)

    summary = {}
    for name, errors in error.items():
        worst = int(errors.argmax())
        summary[name] = {
            "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
            "max_position_error_m": float(errors[worst]),
            "max_error_frame": worst,
            "max_error_route_distance_m": float(distance[worst]),
            "max_error_time_s": float(times[worst]),
            "detected_pairs_translation_median_m": float(np.median([p[name]["translation_error_m"] for p in pairs])),
            "detected_pairs_translation_max_m": float(max(p[name]["translation_error_m"] for p in pairs)),
        }
    report = {
        "protocol": "Read-only diagnosis; GT used only for evaluation. Each trajectory uses its saved global Sim(3). Pair error compares inv(destination) @ source with the GT relative pose; GT separation is not assumed zero.",
        "scope": "Only detected loop pairs; no claim about all true revisits or retrieval recall.",
        "frames": len(gt), "detected_loop_pairs": len(pairs),
        "summary": summary, "pairs": pairs,
        "last_pcg_converged": info["gpu_pgo_last_pcg_converged"],
        "last_pcg_termination": info["gpu_pgo_last_pcg_termination"],
        "last_pcg_relative_residual": info["gpu_pgo_last_pcg_relative_residual"],
    }
    (root / "loop-diagnosis.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    ax = axes[0, 0]
    ax.plot(gt[:, 0, 3], gt[:, 2, 3], color="black", lw=1, label="Ground truth")
    poses = variants["offline_loop"]
    scatter = ax.scatter(poses[::3, 0, 3], poses[::3, 2, 3], c=error["offline_loop"][::3],
                         s=5, cmap="viridis", label="After offline loop")
    src = np.array([e["src_pos"] for e in edges]); dst = np.array([e["dst_pos"] for e in edges])
    ax.scatter(poses[src, 0, 3], poses[src, 2, 3], s=45, marker="*", color="red", label="Later revisit endpoint")
    ax.scatter(poses[dst, 0, 3], poses[dst, 2, 3], s=18, marker="s", color="magenta", label="Earlier endpoint")
    ax.set(xlabel="x (m)", ylabel="z (m)", title="Remaining error and detected loop locations")
    ax.axis("equal"); ax.legend(fontsize=8)
    fig.colorbar(scatter, ax=ax, label="Position error after loop (m)", shrink=.8)

    ax = axes[0, 1]
    for name, label, color in [("raw", "Streaming", "#147ba6"), ("offline_loop", "Offline loop", "#dd8500")]:
        ax.plot(distance / 1000, error[name], lw=1, color=color,
                label=f"{label}: RMSE {summary[name]['ate_rmse_m']:.2f} m")
    worst = summary["offline_loop"]["max_error_frame"]
    ax.scatter(distance[worst]/1000, error["offline_loop"][worst], color="red", s=20)
    ax.set(xlabel="Ground-truth route distance (km)", ylabel="Position error (m)",
           title="Whole trajectory; separate global Sim(3) fits")
    ax.legend(fontsize=8)

    ax = axes[1, 0]; positions = np.arange(len(pairs))
    for name, offset, label, color in [("raw", -.18, "Streaming", "#147ba6"),
                                       ("offline_loop", .18, "Offline loop", "#dd8500")]:
        ax.bar(positions+offset, [p[name]["translation_error_m"] for p in pairs], width=.36, label=label, color=color)
    ax.set_xticks(positions, [str(i+1) for i in positions])
    ax.set(xlabel="Detected pair, ordered by later frame", ylabel="Relative translation error (m)",
           title="Detected revisit pairs: error relative to GT")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for k, pair in enumerate(pairs):
        ax.plot([pair["destination_time_s"], pair["source_time_s"]], [k+1, k+1], color="#999999", lw=1)
        ax.scatter(pair["destination_time_s"], k+1, color="magenta", marker="s", s=22)
        ax.scatter(pair["source_time_s"], k+1, color="red", marker="*", s=50)
    ax.set(xlabel="Original sequence time (s)", ylabel="Detected pair", xlim=(0, times[-1]),
           title=f"Constraint coverage: {len(pairs)} pairs")
    ax.set_yticks(np.arange(1, len(pairs)+1))
    for ax in axes.flat: ax.grid(alpha=.2)
    sequence = json.loads((root / "evaluation.json").read_text())["sequence"]
    fig.suptitle(f"KITTI {sequence} · saved default loop result · no new inference or GT-based correction", fontsize=12)
    fig.savefig(root / "loop-diagnosis.png", dpi=160)
    fig.savefig(root / "loop-diagnosis.pdf")
    plt.close(fig)
    print(json.dumps({"run": str(root), "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    for run in args.runs: diagnose(run)
