# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Add standard trajectory exports and derived queue summaries to saved runs."""
from pathlib import Path
import json
import numpy as np
from realtime_map.mapping import write_trajectory
for root in Path('artifacts').iterdir():
 if not (root/'poses.npy').exists():continue
 poses=np.load(root/'poses.npy');times=np.atleast_1d(np.loadtxt(root/'timestamps.txt'))
 write_trajectory(root/'trajectory.txt',times,poses)
 rows=[json.loads(x) for x in (root/'timings.jsonl').read_text().splitlines()][12:]
 if not rows:continue
 summary={key:{'min':min(r[key] for r in rows),'max':max(r[key] for r in rows),
   'p95':float(np.percentile([r[key] for r in rows],95))}
   for key in ['backlog_frames','gpu_allocated_mib','gpu_reserved_mib','cpu_rss_mib','map_points']}
 (root/'operational-summary.json').write_text(json.dumps(summary,indent=2))
