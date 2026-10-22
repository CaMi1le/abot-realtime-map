# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Plot actual KITTI trajectories and measured latency, without smoothing."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run)
report=json.loads((run/'report.json').read_text());evaluation=json.loads((run/'evaluation.json').read_text())
pred=np.load(run/'poses-aligned.npy');gt=np.load(run/'gt-camera2.npy')
times=np.loadtxt(run/'timestamps.txt');times=times-times[0]
rows=[json.loads(line) for line in (run/'timings.jsonl').read_text().splitlines()]
acks=json.loads((run/'browser-acks.json').read_text())
distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(gt[:,:3,3],axis=0),axis=1))]
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
ax=axes[0,0];ax.plot(gt[:,0,3],gt[:,2,3],label='Ground truth (color camera 2)',color='black',lw=1.3)
ax.plot(pred[:,0,3],pred[:,2,3],label='ABot-Recon, one global Sim(3)',color='#147ba6',lw=1)
if (run/'poses-aligned-loop.npy').exists():
 loop=np.load(run/'poses-aligned-loop.npy');ax.plot(loop[:,0,3],loop[:,2,3],label='Offline loop correction',lw=1)
ax.scatter(gt[0,0,3],gt[0,2,3],marker='o',color='green',label='Start')
ax.set(xlabel='x (m)',ylabel='z (m)',title=f"KITTI {evaluation['sequence']} · {distance[-1]/1000:.2f} km");ax.axis('equal');ax.legend(fontsize=8)
ax=axes[0,1];ax.plot(distance,np.linalg.norm(pred[:,:3,3]-gt[:,:3,3],axis=1),lw=1)
ax.set(xlabel='Ground-truth route distance (m)',ylabel='Position error (m)',title=f"Global Sim(3) ATE RMSE: {evaluation['ate_rmse_m']:.2f} m")
ax=axes[1,0];ax.plot(times,[r['server_latency_s'] for r in rows],lw=.8,label='Server completion')
good=[x for x in acks if 'latency_s' in x]
if good:ax.plot(times[[x['frame_id'] for x in good]],[x['latency_s'] for x in good],lw=.7,label='Browser render + return trip')
ax.axhline(1,color='red',ls='--',lw=1,label='1 s target');ax.legend(fontsize=8)
ax.set(xlabel='Original sequence time (s)',ylabel='Scheduled-arrival latency (s)',title=f"All {len(times)} native frames · {report['source_fps']:.3f} Hz")
ax=axes[1,1];ax.plot(times,[r['gpu_allocated_mib'] for r in rows],label='GPU allocated',lw=1)
ax.plot(times,[r['cpu_rss_mib'] for r in rows],label='CPU RSS',lw=1)
ax.set(xlabel='Original sequence time (s)',ylabel='MiB',title='Memory during real streaming inference');ax.legend(fontsize=8)
for ax in axes.flat:ax.grid(alpha=.2)
fig.suptitle('Front-facing monocular RGB · uncalibrated inference scale · ground truth only for offline evaluation',fontsize=11)
fig.savefig(run/'driving-results.png',dpi=160);fig.savefig(run/'driving-results.pdf');plt.close(fig)
