# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
import json
from types import SimpleNamespace
import numpy as np
import pytest
from realtime_map.kitti import read_kitti, load_groundtruth, segment_errors
from realtime_map.mapping import MapChunks, filtered_points


def test_native_timestamps_are_not_resampled(tmp_path):
    times = [0, .1042, .2088, .3129]
    np.savetxt(tmp_path/'times.txt', times)
    (tmp_path/'image_2').mkdir()
    for i in range(4): (tmp_path/'image_2'/f'{i:06d}.png').touch()
    frames = read_kitti(tmp_path)
    assert [f.timestamp for f in frames] == times
    assert [f.frame_id for f in frames] == list(range(4))
    (tmp_path/'image_2'/'000003.png').unlink()
    assert len(read_kitti(tmp_path, 3)) == 3
    with pytest.raises(FileNotFoundError): read_kitti(tmp_path)


def test_camera2_groundtruth_includes_nonzero_z_offset(tmp_path):
    seq = tmp_path/'dataset'/'sequences'/'05'; seq.mkdir(parents=True)
    poses_dir = seq.parent.parent/'poses'; poses_dir.mkdir()
    world0 = np.eye(4)
    world0[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
    world0[:3,3] = [5,6,7]
    np.savetxt(poses_dir/'05.txt', world0[:3].reshape(1,12))
    intrinsic = np.array([[700,0,600],[0,700,180],[0,0,1.]])
    t20 = np.array([-.06, .01, .007])
    projection = intrinsic @ np.column_stack([np.eye(3),t20])
    (seq/'calib.txt').write_text('P2: '+' '.join(map(str,projection.ravel())))
    gt, transform = load_groundtruth(seq)
    np.testing.assert_allclose(transform[:3,3], -t20)
    np.testing.assert_allclose(gt[0,:3,3], world0[:3,3]-world0[:3,:3]@t20)


def test_segment_errors_and_padding_exclusion():
    gt = np.repeat(np.eye(4)[None], 1001, axis=0)
    gt[:,0,3] = np.arange(1001)
    pred = gt.copy(); pred[:,0,3] *= 1.1
    error = segment_errors(pred,gt,lengths=(100,),start_step=100)
    # KITTI chooses the first frame strictly beyond requested distance.
    np.testing.assert_allclose(error[:,3], .101)
    np.testing.assert_allclose(error[:,4], 0)
    points = np.ones((6,3,3)); points[0] = 999; points[-1] = 999
    result = SimpleNamespace(local_points=points, colors=np.ones_like(points,dtype=np.uint8),
                             confidence=np.ones((6,3)), valid_rows=slice(1,5),camera_pose=np.eye(4))
    xyz, _ = filtered_points(result, stride=1)
    assert len(xyz) == 12 and xyz.max() == 1


def test_chunk_boundaries_and_partial_flush_preserve_detail(tmp_path):
    chunks = MapChunks(tmp_path,.01,100,2)
    rgb = np.array([[1,2,3]],np.uint8)
    for i in range(5): chunks.add(i,np.array([[i,0,1]],np.float32),rgb)
    chunks.flush(); chunks.flush()
    manifest = json.loads((tmp_path/'manifest.json').read_text())['chunks']
    assert [(c['first_frame'],c['last_frame']) for c in manifest] == [(0,1),(2,3),(4,4)]
    assert sum(c['points'] for c in manifest) == 5
    assert all(c['voxel_model_units'] == .01 for c in manifest)
    assert np.load(tmp_path/'00000.npz')['points'][:,0].max() == 1
