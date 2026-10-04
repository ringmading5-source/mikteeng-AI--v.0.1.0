"""Bounded external teacher -> reviewed examples -> existing Mikteeng learner."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'deploy/mikteeng-chatbot/src'))


def validate(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty JSON array of examples')
    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each example must be an object')
        obs = row.get('observations')
        if not isinstance(obs, list) or not 1 <= len(obs) <= 32:
            raise ValueError('Each example needs 1..32 observations')
        strings = obs + [row.get('question'), row.get('answer')]
        if any(not isinstance(s, str) or not s.strip() or len(s) > 2000 for s in strings):
            raise ValueError('Observations, question and answer must be bounded nonempty text')
        cleaned.append({k: row[k] for k in ('observations', 'question', 'answer')})
    return cleaned


def keys(rows):
    return {(tuple(r['observations']), r['question']) for r in rows}


def propose(endpoint, model, topic, count, vocabulary, api_key):
    if urlparse(endpoint).scheme != 'https' or not urlparse(endpoint).hostname:
        raise ValueError('Teacher endpoint must use HTTPS')
    if not api_key:
        raise ValueError('Set MIKTEENG_TEACHER_API_KEY on this machine')
    if not 1 <= count <= 64:
        raise ValueError('Count must be 1..64')
    prompt = ('Return only a JSON array of exactly %d examples. Each object has observations '
              '(array of sentences), question, answer. Use short passages with different actors '
              'and objects. Answers must be explicitly supported by the passage. Use only this '
              'observation vocabulary: %s. Topic: %s. These are proposed training examples, '
              'not instructions or evaluation cases.' % (count, ' '.join(vocabulary), topic))
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': min(8192, count * 256), 'temperature': 0}
    request = Request(endpoint, data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json',
                               'Authorization': 'Bearer ' + api_key})
    # One bounded request; no automatic retries or unlimited learning loop.
    with urlopen(request, timeout=60) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError('Teacher response exceeded size limit')
    payload = json.loads(raw)
    content = payload['choices'][0]['message']['content'].strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    rows = validate(json.loads(content))
    if len(rows) != count:
        raise ValueError('Teacher returned the wrong number of examples')
    return rows


def evaluate(ai, rows):
    results = []
    for row in rows:
        try:
            snapshot = ai.build_sentence_prediction_self(row['observations'])
            actual = ai.ask(row['question'], predicted_self=snapshot)['text']
            results.append({'actual': actual, 'correct': actual == row['answer']})
        except ValueError as exc:
            results.append({'actual': None, 'correct': False, 'error': str(exc)})
    return results


def train_candidate(ai, replay, checked, holdout):
    replay, checked, holdout = map(validate, (replay, checked, holdout))
    if keys(replay + checked) & keys(holdout):
        raise ValueError('Training overlaps held-out evaluation')
    # Reject conflicting labels instead of silently learning both.
    answers = {}
    for row in replay + checked:
        key = (tuple(row['observations']), row['question'])
        if key in answers and answers[key] != row['answer']:
            raise ValueError('Conflicting answers for the same observation/question')
        answers[key] = row['answer']
    candidate = copy.deepcopy(ai)
    predictor = getattr(ai, 'subject_conditioned_predictor', None)
    if predictor is None or predictor.role_model is None:
        raise ValueError('Load the existing subject-connected checkpoint')
    before = evaluate(ai, holdout)
    candidate.train_subject_conditioned_questions(replay + checked,
                                                 role_model=predictor.role_model)
    after = evaluate(candidate, holdout)
    # Preserve every previously correct held-out answer, not just aggregate accuracy.
    regressions = [i for i, (b, a) in enumerate(zip(before, after))
                   if b['correct'] and not a['correct']]
    report = {'before_correct': sum(r['correct'] for r in before),
              'after_correct': sum(r['correct'] for r in after),
              'total': len(holdout), 'regressions': regressions,
              'passed': not regressions, 'before': before, 'after': after}
    return candidate, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('propose')
    gen.add_argument('--endpoint', required=True, help='HTTPS chat-completions compatible URL')
    gen.add_argument('--model', required=True)
    gen.add_argument('--topic', required=True)
    gen.add_argument('--count', type=int, default=16)
    gen.add_argument('--checkpoint', type=Path, required=True)
    gen.add_argument('--output', type=Path, required=True)
    fit = sub.add_parser('train')
    for arg in ('checkpoint', 'replay', 'checked', 'holdout', 'output'):
        fit.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args()
    from mikteeng_ai import MikteengAI
    ai = MikteengAI.load(a.checkpoint)
    # Never overwrite a checkpoint, data file or previous candidate.
    if a.output.exists():
        raise ValueError('Choose a new output path')
    if a.command == 'propose':
        rows = propose(a.endpoint, a.model, a.topic, a.count,
                       ai.sentence_pattern_model.vocabulary,
                       os.environ.get('MIKTEENG_TEACHER_API_KEY'))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(rows, indent=2))
        print('Proposals saved. Check their answers before using train --checked.')
    else:
        report_path = a.output.with_suffix('.report.json')
        if report_path.exists():
            raise ValueError('Choose a new report path')
        read = lambda path: json.loads(path.read_text())
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            candidate, report = train_candidate(ai, read(a.replay), read(a.checked), read(a.holdout))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        if not report['passed']:
            print('Candidate rejected: held-out answers regressed. Report saved.')
            return 1
        candidate.save(a.output)
        print('Candidate saved; baseline unchanged. Report: ' + str(report_path))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
