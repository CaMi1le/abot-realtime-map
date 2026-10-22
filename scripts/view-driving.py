# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Local, CPU-only inspection of saved driving maps and their detailed chunks."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import viser

p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--port',type=int,default=8892);a=p.parse_args()
roots={Path(x).name:Path(x) for x in a.runs}
server=viser.ViserServer(host='127.0.0.1',port=a.port,label='Driving reconstruction results')
if server.get_port()!=a.port:raise RuntimeError('Requested port already occupied')
server.scene.set_up_direction('-y');server.gui.configure_theme(dark_mode=True)
server.gui.add_markdown('# Driving reconstruction\nSaved RGB-only maps · **scale uncalibrated**\n\nBlue: streaming trajectory. Pink: GT aligned to raw trajectory. Orange: offline loop. Detailed chunks retain the original streaming map.')
sequence=server.gui.add_dropdown('Sequence',tuple(roots))
modes=['Detailed road chunk','Road chunk + neighbors','Full route overview']
if all((r/'map-loop.npz').exists() for r in roots.values()):modes.append('Full route after offline loop')
mode=server.gui.add_dropdown('View',tuple(modes))
ground_truth=server.gui.add_checkbox('Show offline-aligned ground truth',False)
chunk=server.gui.add_slider('Road chunk',min=0,max=1,step=1,initial_value=0)
info=server.gui.add_markdown('')
source=server.gui.add_image(np.zeros((152,504,3),np.uint8),label='Front camera (middle of chunk)',format='jpeg')
state={}
def fit(client):
 xyz=state['xyz'];pose=state['pose'];distance=state['distance']
 if not mode.value.startswith('Full route'):
  client.camera.position=pose[:3,3]+pose[:3,:3]@np.array([distance*.5,-distance*.7,-distance*.8])
  client.camera.look_at=pose[:3,3]+pose[:3,:3]@np.array([0,0,distance*.7])
  client.camera.up_direction=tuple(-pose[:3,1])
 else:
  route=state['route'];center=(route.min(0)+route.max(0))/2;extent=max(np.ptp(route,axis=0).max(),distance)
  client.camera.position=center+extent*np.array([.25,-.8,-.4]);client.camera.look_at=center;client.camera.up_direction=(0,-1,0)
@server.on_client_connect
def connect(client):
 if state:fit(client)
def update(_=None):
 root=roots[sequence.value];report=json.loads((root/'report.json').read_text())
 entries=json.loads((root/'map-chunks/manifest.json').read_text())['chunks']
 chunk.max=max(1,len(entries)-1);index=min(int(chunk.value),len(entries)-1);entry=entries[index]
 poses=np.load(root/'poses.npy');middle=(entry['first_frame']+entry['last_frame'])//2;pose=poses[middle]
 corrected_view=mode.value=='Full route after offline loop'
 if mode.value.startswith('Full route'):
  with np.load(root/('map-loop.npz' if corrected_view else 'map.npz')) as f:xyz,rgb=f['points'],f['colors']
  voxel=report['final_voxel_model_units']
 else:
  selected=entries[max(0,index-1):index+2] if mode.value=='Road chunk + neighbors' else [entry]
  points=[];colors=[]
  for item in selected:
   with np.load(root/'map-chunks'/item['file']) as f:points.append(f['points']);colors.append(f['colors'])
  xyz,rgb=np.concatenate(points),np.concatenate(colors)
  voxel=min(item['voxel_model_units'] for item in selected)
 server.scene.add_point_cloud('/map',xyz,rgb,point_size=voxel*.6,point_shape='circle')
 trajectory=poses[:,:3,3]
 server.scene.add_line_segments('/trajectory',np.stack([trajectory[:-1],trajectory[1:]],1),(80,215,255),thickness=2,thickness_units='screen')
 if (root/'gt-camera2.npy').exists():
  gt=np.load(root/'gt-camera2.npy')[:,:3,3]
  alignment=np.load(root/'alignment.npz')
  gt=(gt-alignment['translation'])@alignment['rotation']/alignment['scale']
  server.scene.add_line_segments('/ground-truth',np.stack([gt[:-1],gt[1:]],1),(245,120,230),
   thickness=2,thickness_units='screen',visible=ground_truth.value)
 if (root/'poses-loop.npy').exists():
  corrected=np.load(root/'poses-loop.npy')[:,:3,3]
  state['loop_handle']=server.scene.add_line_segments('/offline-loop',np.stack([corrected[:-1],corrected[1:]],1),(255,180,50),
   thickness=2,thickness_units='screen',visible=ground_truth.value or corrected_view)
  if corrected_view:pose=np.load(root/'poses-loop.npy')[middle]
 elif 'loop_handle' in state:state.pop('loop_handle').remove()
 q=Rotation.from_matrix(pose[:3,:3]).as_quat()
 distance=report['initial_voxel_model_units']*150
 server.scene.add_camera_frustum('/camera',fov=1.05,aspect=1241/376,scale=distance*.08,thickness=1.5,
  thickness_units='screen',wxyz=q[[3,0,1,2]],position=pose[:3,3],color=(255,180,60))
 evaluation=json.loads((root/'evaluation.json').read_text())
 latency=report['browser_latency_upper_bound_s']
 latency_text=f"{latency['p95']:.3f} s" if latency else 'not measured'
 info.content=(f"**{root.name}** · {evaluation['gt_route_length_m']/1000:.2f} km · {len(poses):,} frames\n\n"
  f"ATE after global Sim(3): {evaluation['ate_rmse_m']:.2f} m · browser P95: {latency_text}\n\n"
  f"Chunk {index+1}/{len(entries)} · frames {entry['first_frame']}–{entry['last_frame']} · {len(xyz):,} points")
 if (root/'evaluation-loop.json').exists():
  loop_eval=json.loads((root/'evaluation-loop.json').read_text())
  info.content+=f"\n\nOffline loop ATE: {loop_eval['ate_rmse_m']:.2f} m (separate global Sim(3))"
 snapshot=root/'source-thumbnails'/f'{middle:06d}.jpg'
 if snapshot.exists():source.image=np.asarray(Image.open(snapshot))
 state.update(xyz=xyz,pose=pose,distance=distance,route=trajectory)
 for client in server.get_clients().values():fit(client)
def change_sequence(_):chunk.value=0;update()
sequence.on_update(change_sequence);mode.on_update(update);chunk.on_update(update);update()
ground_truth.on_update(update)
print(f'Saved driving maps: http://127.0.0.1:{server.get_port()}',flush=True)
while True:time.sleep(1)
