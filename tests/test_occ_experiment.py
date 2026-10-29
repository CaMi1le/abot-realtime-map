# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from realtime_map.occ_experiment import compare_runs


def run_files(root, occ=True, frames=120):
    root.mkdir()
    report = {
        "completed": True,
        "source_fps": 10,
        "warmup_frames_excluded": 12,
        "occ": {"enabled": occ, "online_prediction_in_timing": occ},
    }
    (root / "report.json").write_text(json.dumps(report))
    (root / "input.json").write_text(
        json.dumps(
            [
                {"frame_id": i, "timestamp": i / 10, "context": "a", "path": f"{i}.jpg"}
                for i in range(frames)
            ]
        )
    )
    (root / "config.json").write_text(json.dumps({"realtime": True, "device": "cpu"}))
    (root / "hardware.json").write_text(json.dumps({"gpu": "CPU"}))
    rows = [
        {
            "model_ms": 50,
            "processing_ms": 80,
            "occ_predict_ms": 5 if occ else 0,
            "occ_project_ms": 1 if occ else 0,
            "server_latency_s": 0.08,
            "backlog_frames": 0,
        }
        for _ in range(frames)
    ]
    (root / "timings.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def test_realtime_comparison_requires_online_complete_matched_input(tmp_path):
    a, b = tmp_path / "base", tmp_path / "occ"
    run_files(a, False)
    run_files(b)
    assert compare_runs(a, b)["sustained_cadence_pass"]
    report = json.loads((b / "report.json").read_text())
    report["occ"]["online_prediction_in_timing"] = False
    (b / "report.json").write_text(json.dumps(report))
    assert not compare_runs(a, b)["sustained_cadence_pass"]
    config = json.loads((b / "config.json").read_text())
    config["no_dense"] = True
    (b / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="no_dense"):
        compare_runs(a, b)


def test_short_smoke_never_passes_realtime_acceptance(tmp_path):
    a, b = tmp_path / "base", tmp_path / "occ"
    run_files(a, False, 20)
    run_files(b, True, 20)
    assert not compare_runs(a, b)["sustained_cadence_pass"]
