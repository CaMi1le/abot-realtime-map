# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Give MASt3R exactly the same selected RGB frames and source timestamps."""
from pathlib import Path
import json,os
run=Path('artifacts/room')
if not (run/'input.json').exists():
 from realtime_map.data import read_tum
 frames=[{'timestamp':f.timestamp,'path':str(f.path)} for f in read_tum(Path('data/rgbd_dataset_freiburg1_room'))]
else:
 frames=json.loads((run/'input.json').read_text())
dst=Path('data/mast3r-freiburg1_room');dst.mkdir(exist_ok=True)
source=Path('data/rgbd_dataset_freiburg1_room').resolve()
if not (dst/'rgb').exists():os.symlink(source/'rgb',dst/'rgb',target_is_directory=True)
(dst/'rgb.txt').write_text('\n'.join(f"{f['timestamp']:.9f} rgb/{Path(f['path']).name}" for f in frames)+'\n')
print(len(frames),'identical sampled input frames')

parent=Path('data/tum');parent.mkdir(exist_ok=True)
link=parent/dst.name
if not link.exists():os.symlink(Path('..')/dst.name,link,target_is_directory=True)
