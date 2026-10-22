# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp: float
    path: Path


def read_tum(root: Path, fps=10.0, max_frames=None):
    if fps <= 0:
        raise ValueError("fps must be positive")
    entries = []
    for line in (root / "rgb.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            t, path = line.split()[:2]
            entries.append((float(t), root / path))
    if not entries:
        raise ValueError("No RGB frames")
    times = np.array([t for t, _ in entries])
    if np.any(np.diff(times) <= 0):
        raise ValueError("RGB timestamps must increase strictly")
    # Select the first available sample at/after each tick. Never use a future
    # frame early: playback schedules the selected frame's original timestamp.
    count = int(np.floor((times[-1] - times[0]) * fps + 1e-6)) + 1
    ticks = times[0] + np.arange(count) / fps
    # Match nominally equal floating-point timestamps within one microsecond.
    indices = np.unique(np.searchsorted(times + 1e-6, ticks))
    indices = indices[indices < len(entries)]
    result = [FramePacket(i, entries[j][0], entries[j][1]) for i, j in enumerate(indices)]
    if max_frames is not None:
        result = result[:max_frames]
    if not result:
        raise ValueError("No selected frames")
    return result


def repeat_frames(frames, seconds):
    """Continuous synthetic repeat for memory testing, NOT geometric accuracy."""
    duration = frames[-1].timestamp - frames[0].timestamp + .1
    n = 0
    while True:
        for frame in frames:
            t = frame.timestamp - frames[0].timestamp + n * duration
            if t >= seconds:
                return
            yield FramePacket(n * len(frames) + frame.frame_id, t, frame.path)
        n += 1
