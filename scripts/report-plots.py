# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from realtime_map.evaluation import associate
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();root=Path(a.run)
config=json.loads((root/'config.json').read_text())
gt=np.loadtxt(Path(config['input'])/'groundtruth.txt')
fig,axes=plt.subplots(1,2,figsize=(12,5))
axes[0].plot(gt[:,1],gt[:,2],label='Ground truth',color='black',alpha=.6)
for name,suffix in [('Raw',''),('Offline loop','-loop')]:
 if not (root/f'alignment{suffix}.npz').exists():continue
 poses=np.load(root/f'poses{suffix}.npy')
 with np.load(root/f'alignment{suffix}.npz') as f:q=f['scale']*poses[:,:3,3]@f['rotation'].T+f['translation']
 axes[0].plot(q[:,0],q[:,1],label=name)
axes[0].axis('equal');axes[0].legend();axes[0].set_title('Trajectory after global Sim(3) alignment');axes[0].set_xlabel('GT x (m)');axes[0].set_ylabel('GT y (m)')
timings=[json.loads(x) for x in (root/'timings.jsonl').read_text().splitlines()]
axes[1].plot([x['frame_id'] for x in timings],[x['server_latency_s'] for x in timings],label='Server completion')
acks=json.loads((root/'browser-acks.json').read_text())
acks=[x for x in acks if 'latency_s' in x]
if acks:axes[1].plot([x['frame_id'] for x in acks],[x['latency_s'] for x in acks],label='Browser render + return')
axes[1].axhline(1,color='red',linestyle='--',label='1 second target')
axes[1].set_title('Latency from scheduled source arrival');axes[1].set_xlabel('Frame');axes[1].set_ylabel('Seconds');axes[1].legend()
fig.tight_layout();fig.savefig(root/'trajectory-latency.png',dpi=160);plt.close(fig)
