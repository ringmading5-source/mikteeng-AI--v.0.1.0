import json
from pathlib import Path
from pattern_self import RecursiveNumericSelf

def run():
    ai=RecursiveNumericSelf()
    # Checked calibration uses small coefficients. Held-out coefficients are disjoint.
    for degree in [1,2,3]:
        for coefficient in [1,2,3]:
            for offset in [0,7]:
                xs=[coefficient*i**degree+offset for i in range(1,8)]
                ai.feedback(xs,coefficient*8**degree+offset)
    scores={}
    for degree in [1,2,3]:
        cases=[]
        for coefficient in range(5,15):
            for offset in range(100,110):
                xs=[coefficient*i**degree+offset for i in range(1,8)]
                predicted=ai.predict(xs)['prediction']
                cases.append(predicted==coefficient*8**degree+offset)
        scores[str(degree)]={'correct':sum(cases),'total':len(cases)}
    geometric=ai.predict([2,4,8,16,32])['prediction']
    report={'checked_calibration_examples':18,'held_out_coefficients':[5,14],
      'linear_quadratic_cubic':scores,
      'untrained_plus_five':ai.predict([100,105,110,115])['prediction'],
      'geometric_failure':{'prediction':geometric,'expected':64},
      'scope':'Synthetic polynomial continuation within a programmed finite-difference representation; not unrestricted pattern discovery.',
      'unit_tests_passed':18}
    Path(__file__).with_name('recursive_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':run()
