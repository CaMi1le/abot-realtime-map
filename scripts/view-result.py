# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Inspect saved outputs locally; does not perform or claim live inference."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import viser

p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--sequences',nargs='*',default=[]);p.add_argument('--port',type=int,default=8891);a=p.parse_args()
root=Path(a.result)
server=viser.ViserServer(host='127.0.0.1',port=a.port,label='Indoor mapping results')
server.scene.set_up_direction('-y')
server.gui.configure_theme(dark_mode=True)
server.gui.add_markdown('# Indoor mapping results\nSaved reconstruction · **scale uncalibrated**')
maps={};poses={};voxels={}
roots=[root]+[Path(x) for x in a.sequences]
for folder in roots:
 voxel=json.loads((folder/'report.json').read_text())['final_voxel_model_units']
 for title,suffix in [('Raw streaming',''),('After loop closure','-loop')]:
    if not (folder/f'map{suffix}.npz').exists():continue
    label=f'{folder.name} · {title}' if len(roots)>1 else title
    with np.load(folder/f'map{suffix}.npz') as f:maps[label]=(f['points'],f['colors'])
    poses[label]=np.load(folder/f'poses{suffix}.npy');voxels[label]=voxel
choice=server.gui.add_dropdown('Map',tuple(maps))
slider=server.gui.add_slider('Recorded camera frame',min=0,max=len(next(iter(poses.values())))-1,step=1,initial_value=0)
server.gui.add_markdown('Orbit: drag · pan: right drag · zoom: scroll.\nThe full saved map is shown; the slider selects a recorded camera pose.')
@server.on_client_connect
def connect(client):
    voxel=voxels[choice.value]
    xyz=maps[choice.value][0];center=np.median(xyz,axis=0) if len(xyz) else np.zeros(3)
    extent=max(float(np.max(np.ptp(xyz,axis=0))),voxel*100)
    client.camera.position=center+extent*np.array([.45,-.4,-.85]);client.camera.look_at=center;client.camera.up_direction=(0,-1,0)
def camera(_=None):
    voxel=voxels[choice.value]
    pose=poses[choice.value][int(slider.value)]
    q=Rotation.from_matrix(pose[:3,:3]).as_quat()
    server.scene.add_camera_frustum('/camera',fov=1.05,aspect=504/280,scale=voxel*20,
        thickness=1.5,thickness_units='screen',
        color=(255,180,60),wxyz=q[[3,0,1,2]],position=pose[:3,3])
def change(_=None):
    voxel=voxels[choice.value]
    slider.max=len(poses[choice.value])-1
    slider.value=min(slider.value,slider.max)
    xyz,rgb=maps[choice.value]
    server.scene.add_point_cloud('/map',xyz,rgb,point_size=voxel*.6,point_shape='circle')
    t=poses[choice.value][:,:3,3]
    server.scene.add_line_segments('/trajectory',np.stack([t[:-1],t[1:]],1),(80,215,255),thickness=2,thickness_units='screen')
    camera()
    for client in server.get_clients().values():connect(client)
choice.on_update(change);slider.on_update(camera);change()
print(f'Inspect saved result: http://127.0.0.1:{server.get_port()}',flush=True)
while True:time.sleep(1)
