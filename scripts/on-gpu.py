# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Run on one currently idle GPU, without terminating or touching other jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
if not cmd: p.error('command required')
lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
busy={x.strip() for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()}
for line in lines:
    index,uuid,used,util=map(str.strip,line.split(','))
    if uuid in busy or int(used)>1024 or int(util)>10: continue
    lock=open(Path.home()/'.cache'/f'realtime-map-{uuid}.lock','w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();continue
    os.environ['CUDA_VISIBLE_DEVICES']=uuid
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['PATH']=str(Path(__file__).resolve().parents[1]/'.venv/bin')+os.pathsep+os.environ['PATH']
    os.environ.setdefault('CC', 'gcc-12')
    os.environ.setdefault('CXX', 'g++-12')
    os.environ.setdefault('MAX_JOBS', '4')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
    print(json.dumps({'gpu_index':index,'gpu_uuid':uuid,'command':cmd}),flush=True)
    raise SystemExit(subprocess.call(cmd,env=os.environ))
raise SystemExit('No idle GPU available; existing jobs were left untouched.')
