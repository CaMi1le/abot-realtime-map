# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Package local maps, thumbnails, and evidence; dense frames stay on the inference host.

Exported data retains its original dataset/model terms and is not part of the
public source distribution.
"""
import argparse,json,tarfile
from pathlib import Path
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
with tarfile.open(a.output,'w') as archive:
 for name in a.runs:
  root=Path(name);manifest=json.loads((root/'map-chunks/manifest.json').read_text())
  inputs=json.loads((root/'input.json').read_text());thumbs=root/'source-thumbnails';thumbs.mkdir(exist_ok=True)
  for chunk in manifest['chunks']:
   middle=(chunk['first_frame']+chunk['last_frame'])//2
   image=Image.open(inputs[middle]['path']).convert('RGB');image.thumbnail((640,480));image.save(thumbs/f'{middle:06d}.jpg',quality=90)
  for f in sorted(root.rglob('*')):
   relative=f.relative_to(root)
   if not f.is_file() or relative.parts[0] in {'frames','loop'}:continue
   archive.add(f,arcname=str(Path(root.name)/relative),recursive=False)
  for f in (root/'loop').glob('*.json'):
   archive.add(f,arcname=str(Path(root.name)/'loop'/f.name),recursive=False)
print(a.output)
