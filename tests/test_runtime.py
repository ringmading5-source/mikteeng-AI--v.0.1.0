import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np
from mikteeng_rspm import MikteengRSPM,load,save,assert_exact
from mikteeng_rspm.server import make_server

class Runtime(unittest.TestCase):
 def test_checkpoint_and_resume_cli(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); e=MikteengRSPM(dim=16); row=dict(history=e.B[0].tolist(),trigger=e.B[1].tolist(),target=e.B[2].tolist())
   data=d/'data.jsonl'; data.write_text((json.dumps(row)+'\n')*20); checkpoint=d/'model.json'
   cmd=[sys.executable,'-m','mikteeng_rspm','train',str(data),'--dim','16','--output',str(checkpoint),'--checkpoint-every','7']
   p=subprocess.run(cmd,capture_output=True,text=True,check=True); self.assertEqual(json.loads(p.stdout)['observations'],20)
   trained=load(checkpoint); self.assertEqual(trained.step,20); self.assertEqual(trained.query(row['history'],row['trigger'])['status'],'MATCH')
   p=subprocess.run(cmd+['--resume',str(checkpoint)],capture_output=True,text=True,check=True); self.assertEqual(load(checkpoint).step,40)
   s=load(checkpoint).snapshot(); save(load(checkpoint),d/'copy.json'); assert_exact(s,load(d/'copy.json').snapshot())
 def test_http_no_training_and_complete_immutability(self):
  e=MikteengRSPM(dim=16); h,t,y=e.B[:3]
  for _ in range(20): e.train(h,t,y)
  s=e.snapshot(); server=make_server(e,port=0); thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  url=f'http://127.0.0.1:{server.server_port}'
  try:
   health=json.load(urllib.request.urlopen(url+'/health')); self.assertEqual(health['model'],'Mikteeng RSPM'); self.assertFalse(health['training_enabled'])
   for trigger in (t,e.B[4]):
    request=urllib.request.Request(url+'/api/predict',data=json.dumps(dict(history=h.tolist(),trigger=trigger.tolist())).encode(),headers={'Content-Type':'application/json'})
    result=json.load(urllib.request.urlopen(request)); self.assertIn(result['status'],('MATCH','UNMATCHED_TRIGGER'))
   request=urllib.request.Request(url+'/api/predict',data=b'{"history":[],"trigger":[]}')
   with self.assertRaises(urllib.error.HTTPError) as exc: urllib.request.urlopen(request)
   self.assertEqual(exc.exception.code,400)
   request=urllib.request.Request(url+'/api/train',data=b'{}')
   with self.assertRaises(urllib.error.HTTPError) as exc: urllib.request.urlopen(request)
   self.assertEqual(exc.exception.code,404)
   assert_exact(s,e.snapshot())
  finally: server.shutdown();server.server_close();thread.join()
 def test_committed_demo_checkpoint(self):
  model=load(Path(__file__).resolve().parents[1]/'models/synthetic_vector_demo.json')
  self.assertEqual(model.config['dim'],32); self.assertFalse(model.B.flags.writeable)
 def test_invalid_checkpoint_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.json';p.write_text('{"format":"old-mikteeng"}')
   with self.assertRaises(ValueError): load(p)

if __name__=='__main__': unittest.main()
