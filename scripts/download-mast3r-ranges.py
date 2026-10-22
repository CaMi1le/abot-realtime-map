# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Resumable download of the official MASt3R baseline weights."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
from pathlib import Path
import shutil,subprocess,json
ROOT=Path(__file__).resolve().parents[1]
NAME='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
SIZE=2754910614;CHUNK=64*1024**2
URL='https://download.europe.naverlabs.com/ComputerVision/MASt3R/'+NAME
parts=ROOT/'models'/'mast3r-parts';parts.mkdir(parents=True,exist_ok=True)
def get(i):
 a=i*CHUNK;b=min(SIZE,(i+1)*CHUNK)-1;p=parts/f'{i:04d}'
 if p.exists() and p.stat().st_size==b-a+1:return
 subprocess.run(['curl','-fsSL','--http1.1','--retry','3','--retry-all-errors','--connect-timeout','20','--max-time','600','--speed-time','30','--speed-limit','1024','--max-filesize',str(b-a+1),'--range',f'{a}-{b}',URL+f'?chunk={i}','-o',str(p)+'.partial'],check=True)
 tmp=Path(str(p)+'.partial');assert tmp.stat().st_size==b-a+1;tmp.rename(p);print('chunk',i,flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:
 for f in as_completed([pool.submit(get,i) for i in range((SIZE+CHUNK-1)//CHUNK)]):f.result()
dst=ROOT/'models'/'mast3r'/NAME
with dst.with_suffix('.assembling').open('wb') as out:
 for p in sorted(parts.glob('[0-9][0-9][0-9][0-9]')):
  with p.open('rb') as src:shutil.copyfileobj(src,out)
dst.with_suffix('.assembling').replace(dst)
digest=hashlib.file_digest(dst.open('rb'),'sha256').hexdigest()
dst.with_suffix('.source.json').write_text(json.dumps({'source':URL,'size':SIZE,'sha256':digest},indent=2))
print('Ready',digest,flush=True)
