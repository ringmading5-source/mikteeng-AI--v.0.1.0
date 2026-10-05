import json,time
from pathlib import Path
import numpy as np
from mikteeng_ai import MikteengAI
r=Path(__file__).resolve().parents[1]
rate=16000;frame=128;training_frames=40
frequencies=[220,275,330,440];families=['steady','amplitude_ramp','rising_pitch','falling_pitch','mixture']
def signal(f,phase,family,frames):
 t=np.arange(frame*frames)/rate
 if family=='steady':return .25*np.sin(2*np.pi*f*t+phase)
 if family=='amplitude_ramp':return (.12+.25*t)*np.sin(2*np.pi*f*t+phase)
 if family=='rising_pitch':return .25*np.sin(2*np.pi*(f*t+100*t*t/2)+phase)
 if family=='falling_pitch':return .25*np.sin(2*np.pi*(f*t-100*t*t/2)+phase)
 if family=='mixture':return .15*np.sin(2*np.pi*f*t+phase)+.1*np.sin(2*np.pi*330*t+phase*.5)
 raise ValueError(family)
# Fixed before fitting any model in this experiment.
tests=[{'frequency':f,'phase':p,'family':family} for f in frequencies for p in [.4,1.1,1.8,2.5] for family in families]
settings=[('original_tone_coverage',[220,330,440],['steady'],[0,.7,1.4,2.1],[.2,.9,1.6,2.3]),
 ('extra_frequency',frequencies,['steady'],[0,.7,1.4,2.1],[.2,.9,1.6,2.3]),
 ('dynamic_patterns',frequencies,families,[0,.7,1.4,2.1],[.2,.9,1.6,2.3]),
 ('more_examples_same_patterns',frequencies,families,[0,.35,.7,1.05,1.4,1.75,2.1,2.45],[.2,.55,.9,1.25,1.6,1.95,2.3,2.65])]
report={'architecture':'same original acoustic learner: frame128, w3, k3, Ridge alpha1e-4, sound nodes4096','fixed_test_cases':len(tests)*2,'no_test_fitting':True,'test_description':'80 combinations at 24/128 ms; phases held out, later tasks become covered by training','stages':[],'real_speech_tested':False}
for name,freqs,types,pp,sp in settings:
 primary=[(signal(f,p,kind,training_frames),rate) for f in freqs for kind in types for p in pp]
 secondary=[(signal(f,p,kind,training_frames),rate) for f in freqs for kind in types for p in sp]
 start=time.perf_counter()
 ai=MikteengAI().train_speech_predictions(primary,self_recordings=secondary,frame_size=frame,max_sound_nodes=4096)
 fitting=time.perf_counter()-start;rows=[]
 for spec in tests:
  samples=signal(spec['frequency'],spec['phase'],spec['family'],21);context=samples[:5*frame]
  self_state=ai.build_speech_prediction_self(context,rate)
  for horizon in [3,16]:
   expected=samples[5*frame:(5+horizon)*frame];response=ai.predict_speech(self_state,frames=horizon)
   actual=response['samples'];repeat=np.tile(context[-frame:],horizon)
   mse=float(np.mean((actual-expected)**2));base=min(float(np.mean((repeat-expected)**2)),float(np.mean(expected**2)))
   rows.append({**spec,'future_ms':horizon*frame/rate*1000,'mse':mse,'best_trivial_baseline_mse':base,'beats_baseline_10pct':mse<.9*base,'clipped_samples':response['clipped_samples']})
 summary=[]
 for family in families:
  for future_ms in [24.,128.]:
   ids=[v for v in rows if v['family']==family and v['future_ms']==future_ms]
   summary.append({'family':family,'future_ms':future_ms,'mean_mse':float(np.mean([v['mse'] for v in ids])),'wins':sum(v['beats_baseline_10pct'] for v in ids),'total':len(ids)})
 result={'stage':name,'training_recordings':len(primary)+len(secondary),'training_frames':(len(primary)+len(secondary))*training_frames,'fit_seconds':fitting,'mean_mse':float(np.mean([v['mse'] for v in rows])),'wins':sum(v['beats_baseline_10pct'] for v in rows),'total':len(rows),'sound_nodes':len(ai.speech_prediction_model.sound_nodes.ids),'summary':summary,'cases':rows}
 report['stages'].append(result)
 ai.save(r/f'models/acoustic_curve_{name}.mkteeng')
 (r/'experiments/acoustic_training_curve.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ['summary','cases']}),flush=True)
