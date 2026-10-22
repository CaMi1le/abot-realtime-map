# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
import numpy as np
import pytest
from realtime_map.data import read_tum
from realtime_map.mapping import VoxelMap, transform_points, write_ply
from realtime_map.evaluation import sim3, associate


def test_known_camera_to_world_transform():
    pose = np.array([[0,-1,0,4],[1,0,0,5],[0,0,1,6],[0,0,0,1]], float)
    np.testing.assert_allclose(transform_points(np.array([[1,2,3]]), pose), [[2,6,9]])


def test_voxel_cap_preserves_occupied_regions():
    cloud = VoxelMap(.01, 40)
    rng = np.random.default_rng(8)
    for _ in range(5):
        cloud.add(rng.uniform(-2,2,(300,3)), np.ones((300,3),np.uint8)*100)
        assert len(cloud.cells) <= 40
    p, c = cloud.arrays()
    assert len(set(map(tuple, p > 0))) == 8
    assert len(p) == len(c)


def test_sim3_recovers_transform():
    rng = np.random.default_rng(2); x = rng.normal(size=(40,3))
    r = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    y = 2.3*x@r.T + [1,2,3]
    scale, rot, trans = sim3(x,y)
    np.testing.assert_allclose(scale*x@rot.T+trans,y,atol=1e-10)
    with pytest.raises(ValueError): sim3(np.zeros((10,3)), np.ones((10,3)))


def test_causal_time_sampling(tmp_path):
    times = np.arange(31)/30
    (tmp_path/'rgb.txt').write_text('\n'.join(f'{t} rgb/{i}.png' for i,t in enumerate(times)))
    frames = read_tum(tmp_path,10)
    assert len(frames)==11
    assert all(f.timestamp >= i/10-1e-10 for i,f in enumerate(frames))
    assert len({f.path for f in frames})==11
    with pytest.raises(ValueError): read_tum(tmp_path,0)
    (tmp_path/'rgb.txt').write_text('# Empty\n')
    with pytest.raises(ValueError): read_tum(tmp_path)


def test_nearest_timestamp_tolerance():
    idx, valid = associate([1.01,1.98,10], [1,2,3])
    np.testing.assert_array_equal(idx,[0,1,2])
    np.testing.assert_array_equal(valid,[True,True,False])


def test_ply_binary_record_count(tmp_path):
    path=tmp_path/'map.ply'; write_ply(path,np.array([[1,2,3]],np.float32),np.array([[10,20,30]],np.uint8))
    header,body=path.read_bytes().split(b'end_header\n')
    assert b'element vertex 1' in header and len(body)==15


def test_voxel_samples_keep_their_colors_and_do_not_alias():
    cloud=VoxelMap(.1,20)
    xyz=np.array([[-.01,0,0],[.01,0,0],[.05,0,0],[1,0,0]],np.float32)
    rgb=np.array([[10,0,0],[20,0,0],[30,0,0],[40,0,0]],np.uint8)
    cloud.add(xyz,rgb)
    xyz[:]=999;rgb[:]=255
    points,colors=cloud.arrays()
    assert len(points)==3
    assert set(colors[:,0])=={10,20,40}
    cloud.add(np.array([[.02,0,0]],np.float32),np.array([[99,0,0]],np.uint8))
    assert len(cloud.cells)==3
