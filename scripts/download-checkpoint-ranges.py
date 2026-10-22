# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Resumable range download from the official pinned HF checkpoint URL."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

SIZE=4003866352
SHA='ea41a7659f6087069e6b3aac8830cc1c62d7c4a5c27a7d2679b51ba97cabcd2e'
URL='https://huggingface.co/acvlab/ABot-Recon/resolve/c69c26ca1853afc9c9212014459e729d7339ecdf/abot_recon.safetensors?download=true'
ROOT=Path(__file__).resolve().parents[1]
PARTS=ROOT/'models'/'checkpoint-parts'; PARTS.mkdir(parents=True,exist_ok=True)
CHUNK=64*1024**2

def part(i):
    a=i*CHUNK;b=min(SIZE,(i+1)*CHUNK)-1
    dst=PARTS/f'{i:04d}'
    if dst.exists() and dst.stat().st_size==b-a+1:return dst
    tmp=dst.with_suffix('.partial')
    subprocess.run(['curl','-fsSL','--connect-timeout','20','--max-time','900','--retry','3','--retry-all-errors',
        '--max-filesize',str(b-a+1),'--range',f'{a}-{b}',f'{URL}&chunk={i}', '-o',str(tmp)],check=True)
    if tmp.stat().st_size!=b-a+1:raise ValueError(f'Range {i} returned wrong byte count')
    tmp.rename(dst); print(f'chunk {i} ready',flush=True);return dst

with ThreadPoolExecutor(max_workers=8) as pool:
    for f in as_completed([pool.submit(part,i) for i in range((SIZE+CHUNK-1)//CHUNK)]):f.result()
target=ROOT/'models'/'abot_recon.safetensors'
with target.with_suffix('.assembling').open('wb') as out:
    for part in sorted(PARTS.glob('[0-9][0-9][0-9][0-9]')):
        with part.open('rb') as src:shutil.copyfileobj(src,out)
with target.with_suffix('.assembling').open('rb') as src:assert hashlib.file_digest(src,'sha256').hexdigest()==SHA
target.with_suffix('.assembling').rename(target)
print('Verified checkpoint SHA256:',SHA,flush=True)
