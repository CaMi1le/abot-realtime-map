# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
from pathlib import Path
import argparse,json
import numpy as np
from realtime_map.evaluation import sim3,associate
p=argparse.ArgumentParser();p.add_argument('abot');p.add_argument('mast3r');p.add_argument('--dataset',default='data/rgbd_dataset_freiburg1_room');a=p.parse_args()
abot=Path(a.abot);mast=np.loadtxt(a.mast3r);gt=np.loadtxt(Path(a.dataset)/'groundtruth.txt')
at=np.loadtxt(abot/'timestamps.txt');ap=np.load(abot/'poses.npy')
aid,av=associate(mast[:,0],at,.001);gid,gv=associate(mast[:,0],gt[:,0]);good=av&gv
report={'protocol':'Same matched MASt3R keyframe timestamps; independent global Sim(3) for each trajectory; no GT enters either inference.',
        'mast3r_keyframes':len(mast),'matched_keyframes':int(good.sum()),'abot_input_frames':len(ap)}
for label,pred in [('abot',ap[aid[good],:3,3]),('mast3r',mast[good,1:4])]:
 s,r,t=sim3(pred,gt[gid[good],1:4]);e=np.linalg.norm(s*pred@r.T+t-gt[gid[good],1:4],axis=1)
 report[label]={'ate_rmse_m':float(np.sqrt(np.mean(e**2))),'alignment_scale':float(s)}
(abot/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
