# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
from types import SimpleNamespace
import json
import numpy as np
import pytest
from realtime_map.backend import FrameResult
from realtime_map import runner


def test_inference_failure_preserves_completed_frames(tmp_path, monkeypatch):
    data = tmp_path/'data'; data.mkdir()
    (data/'rgb.txt').write_text('\n'.join(f'{i/10:.2f} unused-{i}.jpg' for i in range(25)))
    class Backend:
        def __init__(self,*args,**kwargs):
            self.torch=SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda:None,
                memory_allocated=lambda:0,memory_reserved=lambda:0,max_memory_allocated=lambda:0))
            self.wrapper=SimpleNamespace(attention_backend='TEST_ONLY')
        def reset(self):pass
        def step(self,frame):
            if frame.frame_id==20:raise RuntimeError('injected device failure')
            return FrameResult(frame.frame_id,frame.timestamp,np.eye(4),np.ones((2,2,3),np.float32),
                               np.ones((2,2,3),np.uint8),np.ones((2,2)),slice(0,2),1,1)
    monkeypatch.setattr(runner,'ABotStream',Backend)
    out=tmp_path/'result'
    args=SimpleNamespace(output=str(out),input=str(data),fps=10,max_frames=None,stress_seconds=None,
        viewer=False,checkpoint='unused',attention='sdpa',voxel=0,max_points=100,
        realtime=False,confidence=.5,point_stride=1,no_dense=False,measure_browser=False,
        loop=False,keep_open=False,chunk_frames=7,follow_camera=False)
    with pytest.raises(RuntimeError,match='injected device failure'):runner.run(args)
    report=json.loads((out/'report.json').read_text())
    assert report['completed'] is False and report['realtime_pass'] is False
    assert report['frames']==20 and report['expected_frames']==25
    assert np.load(out/'poses.npy').shape==(20,4,4)
    assert len(list((out/'frames').glob('*.npz')))==20
    assert (out/'map.ply').exists()
    chunks=json.loads((out/'map-chunks/manifest.json').read_text())['chunks']
    assert [(c['first_frame'],c['last_frame']) for c in chunks]==[(0,6),(7,13),(14,19)]
