"""Fixed-basis, bounded context/trigger outcome learner. NumPy only."""
import copy
import numpy as np


def assert_exact(a, b, path='state'):
    """Compare complete snapshots, including array dtype, shape and bit patterns."""
    assert type(a) is type(b), path
    if isinstance(a, np.ndarray):
        assert a.shape == b.shape and a.dtype == b.dtype, path
        assert a.tobytes() == b.tobytes(), path
    elif isinstance(a, dict):
        assert list(a) == list(b), path
        for k in a: assert_exact(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): assert_exact(x, y, f'{path}[{i}]')
    else:
        assert a == b, path


class BoundedRSPM:
    def __init__(self, dim=64, seed=42, max_contexts=10, max_triggers=8,
                 max_modes=4, max_candidates=8, evidence=3, ttl=30,
                 tau_context=.85, tau_trigger=.90, margin=.05, tol=.05, lr=.5):
        for n in (dim, max_contexts, max_triggers, max_modes, max_candidates, evidence, ttl):
            if not isinstance(n, int) or n < 1: raise ValueError('positive integer capacities required')
        self.config = dict(dim=dim, max_contexts=max_contexts, max_triggers=max_triggers,
            max_modes=max_modes, max_candidates=max_candidates, evidence=evidence,
            ttl=ttl, tau_context=tau_context, tau_trigger=tau_trigger, margin=margin, tol=tol, lr=lr)
        q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, dim)))
        self.B = q.T
        self.B.flags.writeable = False
        self.contexts = {}
        self.step = 0
        self.next_context = 0
        self.next_trigger = 0
        self.next_mode = 0

    def vector(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.config['dim'],) or not np.isfinite(x).all() or np.linalg.norm(x) < 1e-12:
            raise ValueError('finite nonzero vector of configured dimension required')
        return x / np.linalg.norm(x)

    def L(self, x):
        x = np.asarray(x, dtype=float)
        a = np.maximum(0., self.B @ x)
        v = a @ self.B
        return v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-12 else np.zeros_like(v)

    def snapshot(self):
        return copy.deepcopy(dict(config=self.config, B=self.B, contexts=self.contexts,
            step=self.step, next_context=self.next_context, next_trigger=self.next_trigger,
            next_mode=self.next_mode, basis_writeable=self.B.flags.writeable))

    def restore(self, s):
        s = copy.deepcopy(s)
        for k in ('config', 'B', 'contexts', 'step', 'next_context', 'next_trigger', 'next_mode'):
            setattr(self, k, s[k])
        self.B.flags.writeable = s['basis_writeable']

    def _route(self, keys, x, tau):
        scores = sorted(((float(k @ x), i) for i, k in keys), reverse=True)
        if not scores or scores[0][0] < tau: return 'UNMATCHED', None
        if len(scores) > 1 and scores[0][0] - scores[1][0] < self.config['margin']:
            return 'AMBIGUOUS', None
        return 'MATCH', scores[0][1]

    def query(self, history, trigger):
        h, t = self.vector(self.L(self.vector(history))), self.vector(trigger)
        status, cid = self._route([(i, c['key']) for i, c in self.contexts.items()], h, self.config['tau_context'])
        if status != 'MATCH': return dict(status=status+'_CONTEXT', hypotheses=[])
        c = self.contexts[cid]
        status, tid = self._route([(i, s['key']) for i, s in c['stores'].items()], t, self.config['tau_trigger'])
        if status != 'MATCH': return dict(status=status+'_TRIGGER', context_id=cid, hypotheses=[])
        store = c['stores'][tid]
        modes = store['modes']
        weights = np.array([m['count'] for m in modes], dtype=float)
        if len(weights): weights /= weights.sum()
        return dict(status='MATCH', context_id=cid, trigger_id=tid,
            forecast_type='MULTIMODAL' if len(modes)>1 else 'UNIMODAL' if modes else 'TENTATIVE',
            parametric=self.L(t+c['R']@t), hypotheses=[m['centroid'].copy() for m in modes],
            probabilities=weights.copy())

    def _sweep(self):
        for c in self.contexts.values():
            for tid, s in list(c['stores'].items()):
                s['candidates'] = [m for m in s['candidates'] if self.step-m['last_seen'] <= self.config['ttl']]
                if not s['modes'] and not s['candidates']: del c['stores'][tid]
        for cid, c in list(self.contexts.items()):
            if not c['persistent'] and self.step-c['last_seen'] > self.config['ttl']:
                del self.contexts[cid]

    def train(self, history, trigger, target):
        h, t, y = self.vector(self.L(self.vector(history))), self.vector(trigger), self.vector(target)
        before = self.query(history, trigger)  # causal forecast before receiving target
        self.step += 1
        self._sweep()  # global training clock, including rejected observations
        status, cid = self._route([(i,c['key']) for i,c in self.contexts.items()], h, self.config['tau_context'])
        if status == 'AMBIGUOUS': return dict(status='AMBIGUOUS_CONTEXT', before=before)
        if status == 'UNMATCHED':
            if len(self.contexts) >= self.config['max_contexts']:
                pool=[(c['agreements'],c['hits'],c['last_seen'],i) for i,c in self.contexts.items() if not c['persistent']]
                if not pool: return dict(status='CONTEXT_CAPACITY_REJECTED', before=before)
                del self.contexts[min(pool)[-1]]
            cid=self.next_context; self.next_context+=1
            self.contexts[cid]=dict(key=h.copy(), R=np.zeros((len(h),len(h))), persistent=False,
                agreements=0,hits=0,last_seen=self.step,stores={})
        c=self.contexts[cid]
        status, tid=self._route([(i,s['key']) for i,s in c['stores'].items()],t,self.config['tau_trigger'])
        if status == 'AMBIGUOUS': return dict(status='AMBIGUOUS_TRIGGER',before=before)
        if status == 'UNMATCHED':
            if len(c['stores']) >= self.config['max_triggers']:
                pool=[(s['last_seen'],i) for i,s in c['stores'].items() if not s['modes']]
                if not pool: return dict(status='TRIGGER_CAPACITY_REJECTED',before=before)
                del c['stores'][min(pool)[1]]
            tid=self.next_trigger; self.next_trigger+=1
            c['stores'][tid]=dict(key=t.copy(),modes=[],candidates=[],last_seen=self.step)
        s=c['stores'][tid]; s['last_seen']=self.step
        c['hits']+=1; c['last_seen']=self.step
        pred=self.L(t+c['R']@t)
        event=self._observe(s,y)
        if event in ('CLUSTERED','CLUSTER_PROMOTED') and len(s['modes']) == 1:
            c['R']+=self.config['lr']*np.outer(y-pred,t)
            if float(pred@y)>=.8: c['agreements']+=1
        if c['agreements']>=self.config['evidence']: c['persistent']=True
        self.check_bounds()
        return dict(status='TRAINED',event=event,context_id=cid,trigger_id=tid,before=before)

    def _observe(self,s,y):
        for name in ('modes','candidates'):
            records=s[name]
            if records:
                idx=int(np.argmax([m['centroid']@y for m in records])); m=records[idx]
                if 1-float(m['centroid']@y)<=self.config['tol']:
                    m['sum']+=y; m['count']+=1; m['centroid']=self.vector(m['sum']); m['last_seen']=self.step
                    if name=='modes': return 'CLUSTERED'
                    if m['count']>=self.config['evidence']:
                        if len(s['modes'])>=self.config['max_modes']: return 'OUTCOME_CAPACITY_REJECTED'
                        s['modes'].append(records.pop(idx)); return 'CLUSTER_PROMOTED'
                    return 'CANDIDATE_ACCUMULATING'
        if len(s['candidates'])>=self.config['max_candidates']:
            idx=min(range(len(s['candidates'])),key=lambda i:(s['candidates'][i]['count'],s['candidates'][i]['last_seen']))
            s['candidates'].pop(idx)
        m=dict(id=self.next_mode,sum=y.copy(),centroid=y.copy(),count=1,last_seen=self.step)
        self.next_mode+=1
        if self.config['evidence']==1:
            if len(s['modes'])>=self.config['max_modes']: return 'OUTCOME_CAPACITY_REJECTED'
            s['modes'].append(m); return 'CLUSTER_PROMOTED'
        s['candidates'].append(m); return 'SPAWNED_CANDIDATE'

    def merge(self,a,b,max_operator_dist=.1):
        """Conservative merge: exact trigger/mode correspondence, no untested union."""
        if a==b or a not in self.contexts or b not in self.contexts:
            return dict(status='NO_OP',reason='Distinct live contexts required')
        ca,cb=self.contexts[a],self.contexts[b]
        if ca['key']@cb['key']<.95: return dict(status='NO_OP',reason='History mismatch')
        if np.linalg.norm(ca['R']-cb['R'])>max_operator_dist:
            return dict(status='NO_OP',reason='Operator mismatch')
        # Exact correspondence is intentionally conservative; near-key consolidation remains unsupported.
        pairs=[]; used=set()
        for ta,sa in ca['stores'].items():
            matches=[(tb,sb) for tb,sb in cb['stores'].items() if np.array_equal(sa['key'],sb['key'])]
            if len(matches)!=1 or matches[0][0] in used: return dict(status='NO_OP',reason='Untested trigger space')
            tb,sb=matches[0]; used.add(tb); pairs.append((ta,sa,sb))
        if not pairs or len(used)!=len(cb['stores']): return dict(status='NO_OP',reason='Incomplete overlap')
        staged=copy.deepcopy(ca)
        for tid,sa,sb in pairs:
            if sa['candidates'] or sb['candidates'] or not sa['modes']:
                return dict(status='NO_OP',reason='Unconfirmed evidence')
            if len(sa['modes'])!=len(sb['modes']): return dict(status='NO_OP',reason='Mode mismatch')
            used_modes=set()
            for ma in staged['stores'][tid]['modes']:
                matches=[(i,mb) for i,mb in enumerate(sb['modes']) if np.array_equal(ma['centroid'],mb['centroid'])]
                if len(matches)!=1 or matches[0][0] in used_modes: return dict(status='NO_OP',reason='Mode mismatch')
                i,mb=matches[0]; used_modes.add(i)
                fa=ma['count']/sum(m['count'] for m in sa['modes']); fb=mb['count']/sum(m['count'] for m in sb['modes'])
                if abs(fa-fb)>.1: return dict(status='NO_OP',reason='Frequency mismatch')
                # Keep a's operator instead of averaging; verify it on every removed trigger.
                if np.linalg.norm(self.L(sb['key']+ca['R']@sb['key'])-self.L(sb['key']+cb['R']@sb['key']))>.01:
                    return dict(status='NO_OP',reason='Prediction degradation')
                ma['sum']+=mb['sum']; ma['count']+=mb['count']; ma['centroid']=self.vector(ma['sum'])
                ma['last_seen']=max(ma['last_seen'],mb['last_seen'])
        staged['persistent']=ca['persistent'] or cb['persistent']
        staged['agreements']+=cb['agreements']; staged['hits']+=cb['hits']
        staged['last_seen']=max(ca['last_seen'],cb['last_seen'])
        self.contexts[a]=staged; del self.contexts[b]
        self.check_bounds()
        return dict(status='MERGED',surviving_context_id=a,removed_context_id=b)

    def check_bounds(self):
        assert len(self.contexts)<=self.config['max_contexts']
        for c in self.contexts.values():
            assert len(c['stores'])<=self.config['max_triggers']
            for s in c['stores'].values():
                assert len(s['modes'])<=self.config['max_modes']
                assert len(s['candidates'])<=self.config['max_candidates']
