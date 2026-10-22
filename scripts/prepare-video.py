# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Convert a local MP4 to a timestamped RGB replay dataset.

This one-time extraction is not included in subsequent replay latency metrics.
"""
import argparse,json
from pathlib import Path
import cv2
p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('output');a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=False);(root/'rgb').mkdir()
cap=cv2.VideoCapture(a.video)
if not cap.isOpened():raise ValueError('Cannot open video')
fps=cap.get(cv2.CAP_PROP_FPS)
if fps<=0:raise ValueError('Video has no valid frame rate')
lines=[];i=0;last=-1
while True:
 ok,bgr=cap.read()
 if not ok:break
 timestamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
 if timestamp<=last:timestamp=i/fps
 if timestamp<=last:raise ValueError('Non-monotonic video timestamps')
 path=f'rgb/{i:06d}.png'
 if not cv2.imwrite(str(root/path),bgr):raise IOError(path)
 lines.append(f'{timestamp:.9f} {path}');last=timestamp;i+=1
cap.release()
if not lines:raise ValueError('Video has no decodable frames')
(root/'rgb.txt').write_text('\n'.join(lines)+'\n')
(root/'source.json').write_text(json.dumps({'video':str(Path(a.video).resolve()),'fps':fps,'frames':i,
 'timing_note':'Offline extraction; replay measures image decode through browser render, not video demux.'},indent=2))
print(i,'frames extracted')
