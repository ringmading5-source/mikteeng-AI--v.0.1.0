"""Read-only RSPM HTTP service. Training is an offline CLI operation."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
from .cli import jsonable

PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Mikteeng RSPM</title><style>body{font:16px system-ui;max-width:900px;margin:40px auto;padding:20px;background:#101828;color:#eef2ff}textarea{width:100%;height:220px;background:#1d2939;color:white;border:1px solid #667085;padding:10px;box-sizing:border-box}button{padding:12px 24px;margin-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Mikteeng RSPM</h1><p>Fixed basis · Context-local learning · Bounded outcome hypotheses</p><p>Submit JSON with <code>history</code> and <code>trigger</code> vectors. This model predicts vectors; it does not currently generate conversational text or speech. The bundled checkpoint is trained on synthetic vector rules.</p><p id="info"></p><textarea id="input" aria-label="Vector query">{"history": [], "trigger": []}</textarea><button id="run">Predict</button><button id="sample">Load demo query</button><pre id="result"></pre><script>fetch('/health').then(r=>r.json()).then(d=>document.getElementById('info').textContent='Vector dimension: '+d.dimension);document.getElementById('sample').onclick=async()=>{document.getElementById('input').value=JSON.stringify(await(await fetch('/api/example')).json(),null,2)};document.getElementById('run').onclick=async()=>{try{let body=JSON.parse(document.getElementById('input').value);let r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});document.getElementById('result').textContent=JSON.stringify(await r.json(),null,2)}catch(e){document.getElementById('result').textContent=e.message}};</script></html>'''

def make_server(model,host='127.0.0.1',port=8000):
    class Handler(BaseHTTPRequestHandler):
        def respond(self,code,data,kind='application/json'):
            payload=data.encode() if isinstance(data,str) else json.dumps(jsonable(data),allow_nan=False).encode()
            self.send_response(code); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(payload))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(payload)
        def do_GET(self):
            if self.path=='/': return self.respond(200,PAGE,'text/html; charset=utf-8')
            if self.path=='/health': return self.respond(200,{'status':'ready','model':'Mikteeng RSPM','version':'0.2.0','dimension':model.config['dim'],'training_enabled':False,'capabilities':['vector_prediction','multimodal_hypotheses'],'contexts':len(model.contexts)})
            if self.path=='/api/example':
                for c in model.contexts.values():
                    if c['stores']: return self.respond(200,{'history':c['key'],'trigger':next(iter(c['stores'].values()))['key']})
                return self.respond(404,{'error':'Checkpoint has no trained triggers'})
            return self.respond(404,{'error':'Not found'})
        def do_POST(self):
            if self.path!='/api/predict': return self.respond(404,{'error':'Only vector inference is exposed'})
            origin=self.headers.get('Origin')
            if origin:
                parsed=urlsplit(origin)
                if parsed.scheme not in ('http','https') or parsed.netloc!=self.headers.get('Host') or parsed.path:
                    return self.respond(403,{'error':'Origin not allowed'})
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=2*1024*1024: return self.respond(413,{'error':'Request size limit exceeded'})
                row=json.loads(self.rfile.read(n))
                if not isinstance(row,dict): raise ValueError('Expected JSON object')
                self.respond(200,model.query(row['history'],row['trigger']))
            except (ValueError,TypeError,KeyError): self.respond(400,{'error':'Supply finite nonzero history and trigger vectors of the configured dimension'})
            except Exception: self.respond(500,{'error':'Prediction failed'})
        def log_message(self,*args): pass
    server=ThreadingHTTPServer((host,port),Handler); server.daemon_threads=True
    return server
