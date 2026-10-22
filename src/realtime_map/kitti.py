# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""KITTI input and offline pose evaluation; calibration/GT never enter the model."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .data import FramePacket
from .evaluation import sim3


def read_kitti(root, max_frames=None):
 root=Path(root);times=np.atleast_1d(np.loadtxt(root/'times.txt'))
 if len(times)==0 or np.any(~np.isfinite(times)) or np.any(np.diff(times)<=0):raise ValueError('Invalid KITTI timestamps')
 n=len(times) if max_frames is None else min(len(times),max_frames)
 if n<=0:raise ValueError('No selected KITTI frames')
 result=[FramePacket(i,float(times[i]),root/'image_2'/f'{i:06d}.png') for i in range(n)]
 missing=[str(f.path) for f in result if not f.path.is_file()]
 if missing:raise FileNotFoundError(f'{len(missing)} missing RGB frames; first: {missing[0]}')
 return result


def load_groundtruth(root):
 root=Path(root);raw=np.loadtxt(root.parent.parent/'poses'/f'{root.name}.txt').reshape(-1,3,4)
 poses=np.repeat(np.eye(4)[None],len(raw),axis=0);poses[:,:3,:]=raw
 calib={}
 for line in (root/'calib.txt').read_text().splitlines():
  key,values=line.split(':',1);calib[key]=np.fromstring(values,sep=' ')
 # P2 = K2 [I | t_2_0], all rectified camera rotations agree. The pose
 # files refer to rectified grayscale cam0; inference uses color cam2.
 p2=calib['P2'].reshape(3,4)
 t_2_0=np.linalg.solve(p2[:,:3],p2[:,3])
 t_0_2=np.eye(4);t_0_2[:3,3]=-t_2_0
 return poses@t_0_2,t_0_2


def segment_errors(pred,gt,lengths=(100,200,300,400,500,600,700,800),start_step=10):
 """KITTI segment convention, applied AFTER one global scale alignment.

 This explicitly differs from official submissions with unadjusted metric poses.
 """
 distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(gt[:,:3,3],axis=0),axis=1))]
 errors=[]
 for first in range(0,len(gt),start_step):
  for length in lengths:
   last=np.searchsorted(distance,distance[first]+length,side='right')
   if last>=len(gt):continue
   delta_gt=np.linalg.inv(gt[first])@gt[last]
   delta_pred=np.linalg.inv(pred[first])@pred[last]
   error=np.linalg.inv(delta_pred)@delta_gt
   translation=float(np.linalg.norm(error[:3,3])/length)
   rotation=float(Rotation.from_matrix(error[:3,:3]).magnitude()/length)
   errors.append([first,int(last),length,translation,rotation])
 return np.asarray(errors).reshape(-1,5)


def evaluate(run,dataset,suffix=''):
 run,dataset=Path(run),Path(dataset);pred=np.load(run/f'poses{suffix}.npy')
 frames=json.loads((run/'input.json').read_text());ids=np.array([f['frame_id'] for f in frames[:len(pred)]])
 all_gt,extrinsic=load_groundtruth(dataset);gt=all_gt[ids]
 scale,rot,trans=sim3(pred[:,:3,3],gt[:,:3,3])
 aligned=pred.astype(float).copy();aligned[:,:3,:3]=rot@pred[:,:3,:3]
 aligned[:,:3,3]=scale*pred[:,:3,3]@rot.T+trans
 errors=np.linalg.norm(aligned[:,:3,3]-gt[:,:3,3],axis=1)
 relatives=np.linalg.inv(aligned[:-1])@aligned[1:]
 gt_relatives=np.linalg.inv(gt[:-1])@gt[1:]
 differences=np.linalg.inv(gt_relatives)@relatives
 segment=segment_errors(aligned,gt)
 distance=float(np.linalg.norm(np.diff(gt[:,:3,3],axis=0),axis=1).sum())
 times=np.array([f['timestamp'] for f in frames[:len(pred)]])
 speed=np.linalg.norm(np.diff(gt[:,:3,3],axis=0),axis=1)/np.diff(times)
 report={'dataset':'KITTI odometry','sequence':dataset.name,'camera':'image_2 (front-left color)',
  'frames':len(pred),'available_frames':len(all_gt),'full_sequence':len(pred)==len(all_gt),
  'alignment':'one global Sim(3), applied to camera2 positions and orientations',
  'calibration_used_for':'offline camera0-to-camera2 GT conversion only; never inference',
  'gt_camera0_from_camera2':extrinsic.tolist(),'alignment_scale':float(scale),
  'gt_route_length_m':distance,'ate_rmse_m':float(np.sqrt(np.mean(errors**2))),
  'source_duration_seconds':float(times[-1]-times[0]),
  'gt_speed_kmh':{'mean':float(speed.mean()*3.6),'p95':float(np.percentile(speed,95)*3.6),'max':float(speed.max()*3.6)},
  'ate_p95_m':float(np.percentile(errors,95)),'ate_max_m':float(errors.max()),
  'delta1_rpe_translation_rmse_m':float(np.sqrt(np.mean(np.sum(differences[:,:3,3]**2,axis=1)))),
  'delta1_rpe_rotation_rmse_deg':float(np.sqrt(np.mean(Rotation.from_matrix(differences[:,:3,:3]).magnitude()**2))*180/np.pi),
  'segment_protocol':'KITTI 100..800m, start every 10 frames, AFTER global Sim(3); not official metric-scale leaderboard score',
  'segment_count':len(segment),
  'segment_translation_percent':float(100*segment[:,3].mean()) if len(segment) else None,
  'segment_rotation_deg_per_100m':float(segment[:,4].mean()*180/np.pi*100) if len(segment) else None,
  'final_position_error_m':float(errors[-1]),'metric_scale_available_during_inference':False,
  'geometry_evaluation':'No LiDAR used in this run; geometry inspected visually, no dense accuracy claim'}
 np.savez(run/f'alignment{suffix}.npz',scale=scale,rotation=rot,translation=trans)
 np.save(run/f'poses-aligned{suffix}.npy',aligned);np.save(run/'gt-camera2.npy',gt)
 np.save(run/f'segment-errors{suffix}.npy',segment)
 (run/f'evaluation{suffix}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 return report

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('dataset');p.add_argument('--suffix',default='');a=p.parse_args();evaluate(a.run,a.dataset,a.suffix)
