# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Apply the official offline loop backend to a saved streaming run."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from realtime_map.backend import ABotStream
from realtime_map.data import FramePacket
from realtime_map.evaluation import refine,evaluate
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--attention',default='sdpa');a=p.parse_args()
out=Path(a.run);config=json.loads((out/'config.json').read_text())
frames=[FramePacket(f['frame_id'],f['timestamp'],Path(f['path'])) for f in json.loads((out/'input.json').read_text())]
backend=ABotStream('models/abot_recon.safetensors',a.attention)
refine(backend,frames,np.load(out/'poses.npy'),out,SimpleNamespace(**config))
if config.get('dataset_format') == 'kitti':
 from realtime_map.kitti import evaluate as evaluate_kitti
 evaluate_kitti(out,config['input'],suffix='-loop')
else:
 evaluate(out,config['input'],suffix='-loop')
