"""Answer-level numerical diagnostics and held-out correctness calibration.

Probabilities concern agreement with the supplied correctness labels, not
universal truth. A learned error detector precedes isotonic calibration.
"""
import re, math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from .meaning import words

FEATURE_NAMES=('subject_min','relationship_min','word_min','extraction_min',
               'subject_mean','relationship_mean','word_mean','extraction_mean',
               'role_coverage','vocabulary_coverage','sentence_count','selected_fraction',
               'empty','limited')

def _stats(values):
    return (min(values),sum(values)/len(values)) if values else (0.,0.)

def diagnostics(model, result):
    trace=result['sentence_trace'];extract=result['extractions']
    subjects=[s['subject_score'] for s in trace]
    relations=[s['relationship_score'] for s in trace]
    word_scores=[w['score'] for s in trace for w in s['word_trace']]
    extraction_scores=[slot['score'] for e in extract for slot in e['slots'].values()]
    known={w for entry in model.meaning.evidence for w in entry['tokens']}
    covered=total=recognized=0
    for entry in extract:
        relevant={i for i,w in enumerate(entry['tokens']) if re.search(r'\w',w)}
        spans=set()
        for slot in entry['slots'].values():
            if slot['span'] is not None:spans.update(range(*slot['span']))
        covered+=len(relevant & spans);total+=len(relevant)
        recognized+=sum(entry['tokens'][i] in known for i in relevant)
    smin,smean=_stats(subjects);rmin,rmean=_stats(relations)
    wmin,wmean=_stats(word_scores);emin,emean=_stats(extraction_scores)
    values=(smin,rmin,wmin,emin,smean,rmean,wmean,emean,covered/max(1,total),
            recognized/max(1,total),len(trace),len(trace)/max(1,len(extract)),
            float(not result['text']),float(result['status'] in ('word_limit','sentence_limit')))
    return dict(zip(FEATURE_NAMES,map(float,values)))

class ReliabilityModel:
    def fit(self,fit_examples,calibration_examples):
        # Each item is (diagnostic dictionary, independently labelled correct).
        fit=list(fit_examples);cal=list(calibration_examples)
        for group in (fit,cal):
            if not group or any(type(label)!=bool for _,label in group):raise ValueError('correctness labels must be booleans')
            if len({label for _,label in group})<2:raise ValueError('both correct and incorrect examples are needed in each split')
        X=np.array([[f[n] for n in FEATURE_NAMES] for f,y in fit]);y=[int(y) for f,y in fit]
        candidate=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42)).fit(X,y)
        CX=np.array([[f[n] for n in FEATURE_NAMES] for f,y in cal])
        calibration=IsotonicRegression(out_of_bounds='clip').fit(candidate.predict_proba(CX)[:,1],[int(y) for f,y in cal])
        self.detector=candidate;self.calibrator=calibration
        self.fit_count=len(fit);self.calibration_count=len(cal)
        return self
    def predict(self,features):
        raw=float(self.detector.predict_proba([[features[n] for n in FEATURE_NAMES]])[0,1])
        return {'detector_score':raw,'estimated_correctness':float(self.calibrator.predict([raw])[0]),
                'fit_examples':self.fit_count,'calibration_examples':self.calibration_count}


def evaluate_predictions(items,threshold=.9):
    if not 0<=threshold<=1:raise ValueError('threshold must be between 0 and 1')
    items=list(items)
    if not items:raise ValueError('evaluation data is empty')
    accepted=[r for r in items if r.get('has_answer',True) and r['estimated_correctness']>=threshold]
    precision=sum(r['correct'] for r in accepted)/len(accepted) if accepted else None
    interval=None
    if accepted:
        n=len(accepted);z=1.96;denominator=1+z*z/n
        center=(precision+z*z/(2*n))/denominator
        margin=z*math.sqrt(precision*(1-precision)/n+z*z/(4*n*n))/denominator
        interval=[max(0.,center-margin),min(1.,center+margin)]
    bins=[];ece=0
    for a,b in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        rows=[r for r in items if a<=r['estimated_correctness'] and (r['estimated_correctness']<b or b==1)]
        if not rows:continue
        confidence=sum(r['estimated_correctness'] for r in rows)/len(rows)
        accuracy=sum(r['correct'] for r in rows)/len(rows)
        ece+=len(rows)/len(items)*abs(confidence-accuracy)
        bins.append({'lower':float(a),'upper':float(b),'count':len(rows),'mean_estimate':confidence,'observed_accuracy':accuracy})
    return {'total':len(items),'accepted':len(accepted),'withheld':len(items)-len(accepted),
            'precision_among_accepted':precision,'precision_wilson_95':interval,'coverage':len(accepted)/len(items),
            'overall_accuracy':sum(r['correct'] for r in items)/len(items),
            'brier_score':sum((r['estimated_correctness']-int(r['correct']))**2 for r in items)/len(items),
            'expected_calibration_error':ece,'threshold':threshold,'bins':bins}
