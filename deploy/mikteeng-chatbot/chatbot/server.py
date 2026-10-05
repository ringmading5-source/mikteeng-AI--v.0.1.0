"""Local Mikteeng chatbot. No external AI API and no server-side training on import."""
import argparse, json, re, threading, os, io, base64, wave
import numpy as np
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from mikteeng_ai import MikteengAI
from mikteeng_ai._models.generator import render,tokens
ROOT=Path(__file__).resolve().parents[1]
RELEASE='2026-10-05-learning'
CAPABILITIES=['definitions','sentence_continuation','word_completion','structured_composition','waveform_continuation']

class Chatbot:
 def __init__(self,root=ROOT):
  self.model=MikteengAI.load_trusted(root/'models/sentence_chatbot.mkteeng')
  self.concepts=[json.loads(line) for line in (root/'data/concepts.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
  self.lock=threading.Lock()
 def coverage(self):
  subjects={}
  for row in self.concepts:subjects.setdefault(row['subject'],[]).append(row['topic'])
  return dict(subjects=subjects,concepts=len(self.concepts),scope='Introductory definitions only; not a complete syllabus')
 def ask(self,message,mode='generated',context=''):
  if not isinstance(message,str) or not message.strip() or len(message)>2000:raise ValueError('Enter a question between 1 and 2000 characters.')
  if mode not in ('generated','reference','continuation','word','composition'):raise ValueError('Unknown response mode.')
  if not isinstance(context,str) or len(context)>8000:raise ValueError('Passage context must be text, up to 8000 characters.')
  if mode=='word':
   seed=message.strip()
   if not re.fullmatch('[a-z]{1,32}',seed):raise ValueError('Enter 1 to 32 lowercase English letters for word completion.')
   current='^'+seed;trace=[];stopped=False
   with self.lock:
    for _ in range(20):
     character=self.model.word_formation_model.predict_next(current)['character'];trace.append(character)
     if character=='$':stopped=True;break
     current+=character
   return dict(text=current[1:],status='experimental_word_completion',mode=mode,topics=[],stopped=stopped,trace=trace,note='Character predictions from 36 English training words. Invalid forms are possible; novel-word generation and meaning are unverified.')
  if mode=='composition':
   spec=json.loads(message)
   if not isinstance(spec,dict) or any(k not in spec for k in ('state','goal','actions')):raise ValueError('Supply a JSON object with state, goal and actions.')
   if not all(isinstance(spec[k],dict) for k in ('state','goal')) or not isinstance(spec['actions'],list) or not isinstance(spec.get('constraints',[]),list):raise ValueError('State and goal must be objects; actions and constraints must be lists.')
   if any(not isinstance(v,dict) for v in spec['actions']+spec.get('constraints',[])):raise ValueError('Actions and constraints must contain objects.')
   # This checkpoint learned only this toy schema; do not imply arbitrary task support.
   if set(spec['state'])!={'a','b','c'} or not spec['goal'] or not set(spec['goal'])<=set(spec['state']):raise ValueError('This demo supports binary state fields a, b and c only.')
   if any(type(v) is not int or v not in (0,1) for v in [*spec['state'].values(),*spec['goal'].values()]):raise ValueError('State and goal values must be 0 or 1.')
   for action in spec['actions']:
    if set(action)!={'field','value'} or action['field'] not in ('a','b','c') or type(action['value']) is not int or action['value'] not in (0,1):raise ValueError('Each action needs field a/b/c and value 0/1.')
   for forbidden in spec.get('constraints',[]):
    if not forbidden or not set(forbidden)<=set(spec['state']) or any(type(v) is not int or v not in (0,1) for v in forbidden.values()):raise ValueError('Constraints must be nonempty binary state objects.')
   with self.lock:
    state=self.model.build_deliberation_self(spec['state'],spec['goal'],spec['actions'],constraints=spec.get('constraints',[]))
    result=self.model.generate_from_self(state,max_depth=6,max_predictions=256,beam_width=16)
   return dict(text=json.dumps(dict(actions=result['response'],final_state=result.get('final_state')),indent=2),status=result['status'],mode=mode,topics=[],predictions_evaluated=result['predictions_evaluated'],note='A generated plan for the learned three-field toy domain. No external actions are executed and this does not interpret natural-language tasks.')
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

 def continue_audio(self,payload):
  if not isinstance(payload,bytes) or not 0<len(payload)<=128044:raise ValueError('Upload a mono PCM16 WAV of at most four seconds.')
  try:
   with wave.open(io.BytesIO(payload),'rb') as wav:
    if wav.getnchannels()!=1 or wav.getsampwidth()!=2 or wav.getframerate()!=16000 or wav.getcomptype()!='NONE':raise ValueError('WAV must be mono PCM16 at 16000 Hz; conversion is not automatic.')
    count=wav.getnframes()
    if not 640<=count<=64000:raise ValueError('WAV must contain 40 ms to four seconds of audio.')
    raw=wav.readframes(count)
    if len(raw)!=count*2:raise ValueError('Truncated WAV data.')
  except (wave.Error,EOFError) as exc:raise ValueError('Invalid WAV file.') from exc
  samples=np.frombuffer(raw,dtype='<i2')
  with self.lock:
   state=self.model.build_speech_prediction_self(samples,16000)
   result=self.model.predict_speech(state,frames=16)
  output=io.BytesIO()
  with wave.open(output,'wb') as wav:
   wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
   wav.writeframes(np.rint(np.clip(result['samples'],-1,1)*32767).astype('<i2').tobytes())
  return dict(status=result['status'],wav_base64=base64.b64encode(output.getvalue()).decode(),sample_rate=16000,duration_ms=128,clipped_samples=result['clipped_samples'],note='128 ms of predicted waveform from a model trained on synthetic signals. Real speech and text-to-speech have not been verified.')

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
   elif self.path=='/health':self.respond(200,dict(status='ready',service='mikteeng-ai',release=RELEASE,modalities=['text','audio'],capabilities=CAPABILITIES))
   elif self.path=='/api/coverage':self.respond(200,bot.coverage())
   else:self.respond(404,dict(error='Not found'))
  def do_POST(self):
   if self.path not in ('/api/chat','/api/acoustic'):return self.respond(404,dict(error='Not found'))
   origin=self.headers.get('Origin')
   if origin:
    parsed=urlsplit(origin)
    allowed={f'http://127.0.0.1:{port}',f'http://localhost:{port}'}
    if not (origin in allowed or (parsed.scheme in ('http','https') and parsed.netloc==self.headers.get('Host') and not parsed.path)):
     return self.respond(403,dict(error='Origin not allowed'))
   try:
    length=int(self.headers.get('Content-Length','0'))
    limit=128044 if self.path=='/api/acoustic' else 12000
    if not 0<length<=limit:return self.respond(413,dict(error='Request too large or empty'))
    if self.path=='/api/acoustic':return self.respond(200,bot.continue_audio(self.rfile.read(length)))
    data=json.loads(self.rfile.read(length))
    if not isinstance(data,dict):raise ValueError('Expected a JSON object.')
    self.respond(200,bot.ask(data.get('message'),data.get('mode','generated'),data.get('context','')))
   except (ValueError,TypeError) as exc:self.respond(400,dict(error=str(exc)))
   except Exception:self.respond(500,dict(error='Prediction failed. Check the server terminal.'))
 return ThreadingHTTPServer((host,port),Handler)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=int(os.environ.get('PORT','8000')));parser.add_argument('--host',default=os.environ.get('HOST','127.0.0.1'));args=parser.parse_args();serve(args.port,args.host)
