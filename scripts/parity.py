# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Real-checkpoint numerical parity and reset checks, not synthetic success."""
import json
from pathlib import Path
import argparse
import numpy as np
import torch
from realtime_map.backend import ABotStream
from realtime_map.data import read_tum

p=argparse.ArgumentParser(); p.add_argument('--attention', default='sdpa'); p.add_argument('--frames',type=int,default=16)
p.add_argument('--input',default='data/rgbd_dataset_freiburg1_desk')
p.add_argument('--dataset-format',choices=['tum','kitti'],default='tum')
p.add_argument('--output')
a=p.parse_args()
backend=ABotStream('models/abot_recon.safetensors',a.attention)
if a.dataset_format=='kitti':
    from realtime_map.kitti import read_kitti
    frames=read_kitti(Path(a.input),max_frames=a.frames)
else:
    frames=read_tum(Path(a.input),max_frames=a.frames)
reference=backend.api.infer([f.path for f in frames],loop_closure=False)
results=list(backend.infer_stream(frames))
report={}
for key,ours,theirs in [('poses',np.stack([x.camera_pose for x in results]),reference.camera_poses.numpy()),
                       ('points',np.stack([x.local_points for x in results]),reference.local_points.numpy()),
                       ('confidence',np.stack([x.confidence for x in results]),reference.confidence.numpy())]:
    diff=np.abs(ours-theirs)
    report[key]={'max_abs':float(diff.max()),'mean_abs':float(diff.mean())}
    np.testing.assert_allclose(ours,theirs,rtol=.002,atol=.002)
again=list(backend.infer_stream(frames[:3]))
np.testing.assert_allclose(again[0].camera_pose,results[0].camera_pose,atol=1e-6)
try:
    list(backend.infer_stream([]))
    raise AssertionError('empty stream should fail')
except ValueError: pass
report.update({'passed':True,'frames':len(frames),'backend':a.attention,'reset_checked':True,'empty_checked':True})
Path(a.output or f'artifacts/parity-{a.attention}.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
