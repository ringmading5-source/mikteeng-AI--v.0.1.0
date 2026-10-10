"""Fresh held-out identifiers; no generated code is executed."""
import ast,json,time
from pathlib import Path
from evaluate_sequence import dataset,evaluate
from sequence import SequenceLearner
from copy_conditioning import CopyConditionedLearner,CopyConditioner

def fresh_tests():
    rows=[]
    for n in ['Mading','Nyandeng','Akol','Aluel','Zoë','Mary Jane']:
        rows.append({'task':'conversation','prompt':f'My name is {n}. Say hello.','response':f'Hello, {n}!'})
    for op,names,symbol in [('add',['aggregate_values','sum_two','combine_numbers','add_pair'],'+'),('subtract',['take_away','subtract_pair','difference_two','net_value'],'-')]:
        for n in names:
            action='add a and b' if op=='add' else 'subtract b from a'
            rows.append({'task':'coding','prompt':f'Write Python function {n} to {action}.','response':f'def {n}(a, b):\n    return a {symbol} b'})
    return rows

def functional_structure(text,expected):
    # Exact AST agreement checks the required function name, parameters and operation;
    # not arbitrary Python execution or evidence of general code correctness.
    try:return ast.dump(ast.parse(text),include_attributes=False)==ast.dump(ast.parse(expected),include_attributes=False)
    except (SyntaxError,TypeError):return False

if __name__=='__main__':
    start=time.perf_counter();train,_=dataset();tests=fresh_tests()
    assert not {x['prompt'] for x in train}&{x['prompt'] for x in tests}
    n=SequenceLearner()
    if Path('sequence_pilot.npz').exists():n=SequenceLearner.load('sequence_pilot.npz')
    else:n.train(train);n.save('sequence_pilot.npz')
    hybrid=CopyConditionedLearner(n);info=hybrid.fit_copy(train)
    results=[]
    for row in tests:
        p=hybrid.generate(row['prompt'])
        results.append({**row,'predicted':p['text'],'route':p['route'],'exact':p['text']==row['response'],
          'required_ast_match':functional_structure(p['text'],row['response']) if row['task']=='coding' else None})
    unfamiliar=['Please greet Mading.','Write a sorting algorithm.','Translate this Bor sentence.','What is 19 times 23?','My name is Akol. Say goodbye.','Write Python function product to multiply a and b.']
    abstentions=[{'prompt':q,**hybrid.generate(q)} for q in unfamiliar]
    hybrid.copy.save('copy_model.json');restored=CopyConditioner.load('copy_model.json')
    assert all(restored.generate(x['prompt'])==hybrid.generate(x['prompt']) for x in tests)
    baseline=evaluate(n,tests)
    report={'method':'single-span copying via learned prompt/response templates; not neural generalization',
      'training_examples':len(train),'fresh_test_count':len(tests),'learned_rules':info['rules'],
      'neural_only_correct':baseline['correct'],'copy_conditioned_correct':sum(x['exact'] for x in results),
      'copy_disabled_correct':baseline['correct'],'unknown_abstentions':sum(x['route']=='abstain' for x in abstentions),
      'unknown_count':len(abstentions),'seconds':time.perf_counter()-start,'tests':results,'unknown_tests':abstentions,
      'limits':['same templates, new copied spans only','no paraphrase or new-operation understanding',
      'no speech transcription','no arbitrary code execution','neural model itself unchanged',
      'pairwise template fitting capped at 256 examples; no large-scale training claim']}
    Path('copy_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    Path('copy_heldout.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in tests)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('tests','unknown_tests')},indent=2))
