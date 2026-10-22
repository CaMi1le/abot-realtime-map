# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Summarize the timed synthetic repeat separately from geometric evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('artifacts/stress')
report=json.loads((root/'report.json').read_text())
rows=[json.loads(x) for x in (root/'timings.jsonl').read_text().splitlines()]
steady=rows[100:]
gpu=np.array([r['gpu_allocated_mib'] for r in steady])
cpu=np.array([r['cpu_rss_mib'] for r in steady])
count=np.array([r['map_points'] for r in rows])
result={'purpose':'600-second synthetic repeated RGB memory test; not long-route geometric accuracy',
 'completed':report['completed'],'frames':len(rows),'wall_seconds':report['wall_seconds'],
 'gpu_allocated_min_mib_after_100':float(gpu.min()),'gpu_allocated_max_mib_after_100':float(gpu.max()),
 'gpu_last_100_minus_first_100_mib':float(gpu[-100:].mean()-gpu[:100].mean()),
 'cpu_rss_min_mib_after_100':float(cpu.min()),'cpu_rss_max_mib_after_100':float(cpu.max()),
 'cpu_rss_last_100_mean_mib':float(cpu[-100:].mean()),'max_preview_points':int(count.max()),
 'max_backlog_frames':max(r['backlog_frames'] for r in rows),
 'gpu_memory_stable_within_32mib':bool(gpu.max()-gpu.min()<=32),
 'preview_cap_respected':bool(count.max()<=1_000_000),'dense_archival_enabled':False,
 'browser_measured':False}
(root/'memory-report.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
t=np.array([r['timestamp'] for r in rows]);t-=t[0]
for a,key,label in zip(ax,['gpu_allocated_mib','cpu_rss_mib','map_points'],['GPU allocated (MiB)','CPU RSS (MiB)','Preview points']):
 a.plot(t,[r[key] for r in rows]);a.set_ylabel(label);a.grid(alpha=.2)
ax[-1].set_xlabel('Synthetic replay time (seconds)')
fig.suptitle('10-minute memory test: repeated RGB, no dense archival, no browser')
fig.tight_layout();fig.savefig(root/'memory.png',dpi=150)
print(json.dumps(result,indent=2))
