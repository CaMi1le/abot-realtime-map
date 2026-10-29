# SPDX-License-Identifier: Apache-2.0
"""Compare matched causal runs; cached priors cannot earn an online pass."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_run(root):
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    inputs = json.loads((root / "input.json").read_text())
    config = json.loads((root / "config.json").read_text())
    rows = [
        json.loads(line)
        for line in (root / "timings.jsonl").read_text().splitlines()
        if line
    ]
    return report, inputs, config, rows


def compare_runs(baseline, candidate):
    base, bi, bc, br = load_run(baseline)
    test, ti, tc, tr = load_run(candidate)

    def identity(inputs):
        return [
            (r["frame_id"], r["timestamp"], r.get("context"), r["path"]) for r in inputs
        ]

    if identity(bi) != identity(ti):
        raise ValueError("Experiments must use identical source frames and timestamps")
    for key in [
        "attention",
        "device",
        "confidence",
        "point_stride",
        "voxel",
        "max_points",
        "chunk_frames",
        "viewer",
        "measure_browser",
        "render_width",
        "render_height",
        "no_dense",
        "realtime",
        "checkpoint",
    ]:
        if bc.get(key) != tc.get(key):
            raise ValueError(f"Unmatched runtime configuration: {key}")
    bh = json.loads((Path(baseline) / "hardware.json").read_text())
    th = json.loads((Path(candidate) / "hardware.json").read_text())
    if any(
        bh.get(k) != th.get(k)
        for k in [
            "hostname",
            "gpu",
            "cuda_visible_devices",
            "torch",
            "input_shape",
            "amp_dtype",
            "source_commit",
        ]
    ):
        raise ValueError("Baseline and candidate hardware/model profiles differ")
    if base.get("occ", {}).get("enabled"):
        raise ValueError("Reference run must be unconditioned ABot")
    warm = max(
        base.get("warmup_frames_excluded", 12), test.get("warmup_frames_excluded", 12)
    )
    br, tr = br[warm:], tr[warm:]

    def stats(rows, key):
        v = np.array([r[key] for r in rows], float)
        return (
            {"mean": float(v.mean()), "p95": float(np.percentile(v, 95))}
            if len(v)
            else None
        )

    metrics = {}
    for key in [
        "model_ms",
        "processing_ms",
        "occ_predict_ms",
        "occ_project_ms",
        "server_latency_s",
        "backlog_frames",
    ]:
        b = stats(br, key) if br and all(key in r for r in br) else None
        c = stats(tr, key) if tr and all(key in r for r in tr) else None
        metrics[key] = {
            "baseline": b,
            "candidate": c,
            "p95_delta": c["p95"] - b["p95"] if b and c else None,
        }
    cadence = test.get("source_fps")
    online = test.get("occ", {}).get("online_prediction_in_timing", False)
    enough = len(tr) >= 100
    # A short smoke result cannot establish sustained realtime. Require actual
    # original-speed replay and inspect backlog evolution, not average FPS alone.
    if enough:
        lag = np.array([r["backlog_frames"] for r in tr])
        split = max(1, len(lag) // 4)
        backlog_growth = float(np.median(lag[-split:]) - np.median(lag[:split]))
    else:
        backlog_growth = None
    sustained = bool(
        test.get("completed")
        and online
        and enough
        and tc.get("realtime")
        and cadence
        and metrics["processing_ms"]["candidate"]["mean"] < 1000 / cadence
        and backlog_growth <= 1.0
    )
    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "metrics": metrics,
        "sustained_cadence_pass": sustained,
        "post_warmup_frames": len(tr),
        "backlog_growth_frame_periods": backlog_growth,
        "gate": "At least 100 post-warmup frames; full causal replay; online OCC timing; mean service below input period; tail backlog growth <=1 frame period",
        "browser_realtime_pass": test.get("native_cadence_realtime_pass", False),
        "quality_status": "Use independently evaluated geometry and tracking; timing alone does not establish improvement",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = compare_runs(args.baseline, args.candidate)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
