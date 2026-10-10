"""Reproducible synthetic checks, not real-world generalization evidence."""
import copy,json,time,unittest,platform,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'tests'))
from test_repairs import toy,score
from tbin_v25 import TBIN25

def main():
    started=time.perf_counter()
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    experiments=[]
    for seed in (18,19,20):
        pairs,held=toy(seed);model=TBIN25()
        baseline=score(model,held)
        history=model.train_unlabeled_pairs(pairs,epochs=150,lr=.025)
        shuffled=TBIN25();wrong=copy.deepcopy(pairs)
        # Deliberately rotate all audio-concept assignments (consistent wrong supervision).
        for i,row in enumerate(wrong):row['audio']=pairs[(i//6)*6+(i+1)%6]['audio']
        shuffled.train_unlabeled_pairs(wrong,epochs=150,lr=.025)
        experiments.append({'seed':seed,'queries':12,'untrained_correct':baseline,
          'trained_correct':score(model,held),'wrong_pair_control_correct':score(shuffled,held),
          'loss_first':history[0]['loss'],'loss_last':history[-1]['loss'],
          'max_batch':model.training_stats['max_batch']})
    report={'version':'1.1','python':platform.python_version(),'numpy':np.__version__,
      'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
      'synthetic_retrieval':experiments,'projection_parameters':sum(w.size for w in model.shared.weights.values()),
      'projection_bytes':sum(w.nbytes for w in model.shared.weights.values()),
      'elapsed_seconds':time.perf_counter()-started,
      'real_world_generalization':'not evaluated; no real paired held-out dataset provided',
      'limits':['fixed engineered features plus learned patch codebook and linear projections',
      'restricted learned-predicate parser, explicit fact/demo supervision',
      'neural ranker training wired in; benefit over graph-only answers not established',
      'bounded batch allocations; not a million-example throughput benchmark',
      'graph and text statistics grow when explicitly enabled',
      'no braid invariants, ASR, TTS, open-ended generation or universal AI claim']}
    Path('TBIN_1_1_test_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
