# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Read selected public KITTI ZIP members using HTTP ranges, with CRC validation.

No login bypass or mirror: bytes come from KITTI's publicly readable S3 objects.
Only front-left RGB and small calibration/pose archives are fetched.
"""
import argparse,io,json,os,struct,time,urllib.request,zipfile,zlib
from pathlib import Path,PurePosixPath
from concurrent.futures import ThreadPoolExecutor,as_completed
BASE='https://s3.eu-central-1.amazonaws.com/avg-kitti/'

class RangeFile(io.RawIOBase):
 def __init__(self,url):
  self.url=url;self.position=0
  with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as r:
   self.size=int(r.headers['Content-Length']);self.etag=r.headers.get('ETag')
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.position
 def seek(self,offset,whence=0):
  self.position=offset if whence==0 else (self.position if whence==1 else self.size)+offset
  if self.position<0:raise ValueError('Negative seek')
  return self.position
 def read(self,n=-1):
  n=min(self.size-self.position,n if n>=0 else self.size-self.position)
  if n<=0:return b''
  if n>128*2**20:raise ValueError('Unexpected oversized ZIP metadata read')
  data=self.range(self.position,self.position+n-1);self.position+=len(data);return data
 def range(self,start,end):
  for attempt in range(5):
   try:
    req=urllib.request.Request(self.url+f'?realtime-map-range={start}-{end}',headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=90) as r:
     if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
      raise ValueError('Server did not honor exact byte range')
     if self.etag and r.headers.get('ETag')!=self.etag:raise ValueError('Source changed during download')
     data=r.read(end-start+2)
    if len(data)!=end-start+1:raise IOError('Truncated range')
    return data
   except Exception:
    if attempt==4:raise
    time.sleep(min(8,2**attempt))

def main():
 p=argparse.ArgumentParser();p.add_argument('--sequences',nargs='+',default=['05','02']);p.add_argument('--root',default='data/kitti');p.add_argument('--workers',type=int,default=8);p.add_argument('--index-only',action='store_true');a=p.parse_args()
 root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
 source=RangeFile(BASE+'data_odometry_color.zip')
 index_path=root/'color-index.json'
 if index_path.exists():
  manifest=json.loads(index_path.read_text());assert manifest['etag']==source.etag;entries=manifest['entries']
 else:
  with zipfile.ZipFile(source) as archive:
   entries=[{'name':i.filename,'offset':i.header_offset,'compressed_size':i.compress_size,'size':i.file_size,'crc':i.CRC,'method':i.compress_type} for i in archive.infolist() if not i.is_dir()]
  index_path.write_text(json.dumps({'url':source.url,'etag':source.etag,'archive_bytes':source.size,'entries':entries}))
 selected=[]
 for seq in a.sequences:
  selected.extend(e for e in entries if f'/sequences/{seq}/image_2/' in e['name'] or e['name'].endswith((f'/sequences/{seq}/times.txt',f'/sequences/{seq}/calib.txt')))
 print('Selected',len(selected),'members',sum(e['compressed_size'] for e in selected)/2**30,'GiB',flush=True)
 (root/'selected-source.json').write_text(json.dumps({'url':source.url,'etag':source.etag,'sequences':a.sequences,'camera':'image_2','members':selected},indent=2))
 if a.index_only:return
 groups=[]
 for seq in a.sequences:
  members=sorted([e for e in selected if f'/sequences/{seq}/' in e['name']],key=lambda e:e['offset'])
  group=[];start=end=0
  for e in members:
   dst=root/e['name']
   if dst.exists() and dst.stat().st_size==e['size'] and zlib.crc32(dst.read_bytes())==e['crc']:continue
   item_end=min(source.size-1,e['offset']+30+len(e['name'].encode())+65535+e['compressed_size'])
   if group and (item_end-start>16*2**20 or e['offset']>end+65536):groups.append((start,end,group));group=[]
   if not group:start=e['offset']
   group.append(e);end=item_end
  if group:groups.append((start,end,group))
 def download(group):
  start,end,members=group;data=source.range(start,end)
  for e in members:
   offset=e['offset']-start
   head=struct.unpack_from('<IHHHHHIIIHH',data,offset)
   if head[0]!=0x04034b50:raise ValueError('ZIP local header mismatch')
   name_len,extra_len=head[-2:];payload=offset+30+name_len+extra_len
   packed=data[payload:payload+e['compressed_size']]
   raw=zlib.decompress(packed,-15) if e['method']==8 else packed
   if len(raw)!=e['size'] or zlib.crc32(raw)!=e['crc']:raise ValueError('ZIP member CRC mismatch: '+e['name'])
   if '..' in PurePosixPath(e['name']).parts or e['name'].startswith('/'):raise ValueError('Unsafe ZIP path')
   dst=root/e['name'];dst.parent.mkdir(parents=True,exist_ok=True)
   tmp=dst.with_suffix(dst.suffix+'.partial');tmp.write_bytes(raw);tmp.replace(dst)
  return len(data)
 total=0;started=time.monotonic()
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(download,g) for g in groups]
  for i,f in enumerate(as_completed(futures)):
   total+=f.result()
   if i%10==0 or i+1==len(groups):print(f'{i+1}/{len(groups)} ranges, {total/2**20:.0f} MiB, {time.monotonic()-started:.1f}s',flush=True)
 for name in ['data_odometry_calib.zip','data_odometry_poses.zip']:
  dst=root/name
  if not dst.exists():urllib.request.urlretrieve(BASE+name,dst)
  with zipfile.ZipFile(dst) as archive:
   for info in archive.infolist():
    if not info.is_dir() and (any(f'/sequences/{s}/' in info.filename for s in a.sequences) or any(info.filename.endswith(f'/poses/{s}.txt') for s in a.sequences)):
     if '..' in PurePosixPath(info.filename).parts or info.filename.startswith('/'):raise ValueError('Unsafe path')
     archive.extract(info,root)
 (root/'download-complete.json').write_text(json.dumps({'sequences':a.sequences,'members_crc_verified':len(selected),'source_etag':source.etag,'elapsed_seconds':time.monotonic()-started},indent=2))
 print('Complete; all selected image members validated by ZIP CRC32',flush=True)
if __name__=='__main__':main()
