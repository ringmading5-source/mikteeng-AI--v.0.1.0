"""Local Mikteeng chatbot. No external AI API and no server-side training on import."""
import argparse, json, re, threading, os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from mikteeng_ai import MikteengAI
from mikteeng_ai._models.generator import render,tokens
ROOT=Path(__file__).resolve().parents[1]

class Chatbot:
 def __init__(self,root=ROOT):
  self.model=MikteengAI.load(root/'models/sentence_chatbot.mkteeng')
  self.concepts=[json.loads(line) for line in (root/'data/concepts.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
  self.lock=threading.Lock()
 def coverage(self):
  subjects={}
  for row in self.concepts:subjects.setdefault(row['subject'],[]).append(row['topic'])
  return dict(subjects=subjects,concepts=len(self.concepts),scope='Introductory definitions only; not a complete syllabus')
 def ask(self,message,mode='generated',context=''):
  if not isinstance(message,str) or not message.strip() or len(message)>2000:raise ValueError('Enter a question between 1 and 2000 characters.')
  if mode not in ('generated','reference','continuation'):raise ValueError('Unknown response mode.')
  if not isinstance(context,str) or len(context)>8000:raise ValueError('Passage context must be text, up to 8000 characters.')
  if mode=='continuation':
   if not context.strip():raise ValueError('Add preceding sentences in the passage context box.')
   with self.lock:result=self.model.continue_passage(context,message,max_words=55)
   return dict(text=result['text'],status='experimental_continuation',mode=mode,topics=[],generation_status=result['status'],trace=result['trace'],note='Uses preceding sentence states. Trained on eight small synthetic patterns; outside those patterns predictions can be wrong.')
  # Coverage metadata and output validation do not supply tokens to the generator.
  lower=message.lower()
  matches=[r for r in self.concepts if re.search(r'(?<!\w)'+re.escape(r['topic'].lower())+r'(?!\w)',lower)]
  # Prefer longer concepts: electric current should not match an incidental shorter label.
  matches.sort(key=lambda r:len(r['topic']),reverse=True)
  if not matches:
   return dict(text='I do not have a recognised topic for this question in my introductory curriculum. Choose a topic from the coverage list.',status='outside_coverage',mode=mode,topics=[])
  if mode=='reference':
   return dict(text='\n\n'.join(r['answer'] for r in matches[:4]),status='curriculum_reference',mode=mode,topics=[r['topic'] for r in matches[:4]],note='These are stored curriculum definitions, not generated answers.')
  with self.lock: result=self.model.ask(message,max_words=55)
  exact=result['text']==render(tokens(matches[0]['answer'])) and len(matches)==1
  return dict(text=result['text'],status='definition_text_match' if exact else 'unverified_generation',mode=mode,topics=[r['topic'] for r in matches[:4]],generation_status=result['status'],trace=result['trace'],note='Generated token by token by Mikteeng. A match checks wording of a definition only; it does not validate reasoning or fully answer complex questions.')

def serve(port, host="127.0.0.1"):
 server=make_server(port,host)
 print(f"Mikteeng chatbot listening on {host}:{server.server_port}",flush=True)
 server.serve_forever()

def make_server(port, host="127.0.0.1"):

 bot=Chatbot()
 class Handler(BaseHTTPRequestHandler):
  def respond(self,status,body,content='application/json'):
   payload=body if isinstance(body,bytes) else json.dumps(body).encode()
   self.send_response(status);self.send_header('Content-Type',content);self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
  def do_GET(self):
   if self.path=='/':self.respond(200,(ROOT/'chatbot/index.html').read_bytes(),'text/html; charset=utf-8')
   elif self.path=='/health':self.respond(200,dict(status='ready',service='mikteeng-ai',modalities=['text']))
   elif self.path=='/api/coverage':self.respond(200,bot.coverage())
   else:self.respond(404,dict(error='Not found'))
  def do_POST(self):
   if self.path!='/api/chat':return self.respond(404,dict(error='Not found'))
   origin=self.headers.get('Origin')
   if origin:
    parsed=urlsplit(origin)
    allowed={f'http://127.0.0.1:{port}',f'http://localhost:{port}'}
    if not (origin in allowed or (parsed.scheme in ('http','https') and parsed.netloc==self.headers.get('Host') and not parsed.path)):
     return self.respond(403,dict(error='Origin not allowed'))
   try:
    length=int(self.headers.get('Content-Length','0'))
    if not 0<length<=12000:return self.respond(413,dict(error='Request too large or empty'))
    data=json.loads(self.rfile.read(length))
    if not isinstance(data,dict):raise ValueError('Expected a JSON object.')
    self.respond(200,bot.ask(data.get('message'),data.get('mode','generated'),data.get('context','')))
   except (ValueError,TypeError) as exc:self.respond(400,dict(error=str(exc)))
   except Exception:self.respond(500,dict(error='Prediction failed. Check the server terminal.'))
 return ThreadingHTTPServer((host,port),Handler)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=int(os.environ.get('PORT','8000')));parser.add_argument('--host',default=os.environ.get('HOST','127.0.0.1'));args=parser.parse_args();serve(args.port,args.host)
