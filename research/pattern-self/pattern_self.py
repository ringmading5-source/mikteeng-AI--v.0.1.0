"""Standard-library transition learner. No domain answer rules or fitted ML library."""
from collections import Counter, defaultdict
from math import log

class PatternSelf:
    def __init__(self, max_context=4):
        if max_context < 1: raise ValueError('max_context must be positive')
        self.max_context = max_context
        self.patterns = defaultdict(Counter)
        self.observations = 0
        self.checked_predictions = Counter()

    def observe(self, sequence):
        """Sequences are separate episodes; input tokens must be hashable."""
        sequence = list(sequence)
        for i, outcome in enumerate(sequence):
            for length in range(min(i, self.max_context) + 1):
                context = tuple(sequence[i-length:i]) if length else ()
                self.patterns[context][outcome] += 1
            self.observations += 1
        return self

    def predict(self, context):
        context = list(context)
        # Architecture choice: longest observed suffix, with explicit fallback.
        for length in range(min(len(context), self.max_context), -1, -1):
            key = tuple(context[-length:]) if length else ()
            counts = self.patterns.get(key)
            if counts:
                total = sum(counts.values())
                probabilities = {token: n/total for token, n in counts.items()}
                best = max(probabilities, key=probabilities.get)
                return {'prediction': best, 'distribution': probabilities,
                        'support': total, 'context': key,
                        'frequency': probabilities[best],
                        'entropy': -sum(p*log(p) for p in probabilities.values())}
        return {'prediction': None, 'distribution': {}, 'support': 0,
                'context': (), 'frequency': 0.0, 'entropy': None}

    def feedback(self, context, actual):
        """Record a checked result; update only transitions ending in actual."""
        context = list(context)
        predicted = self.predict(context)['prediction']
        self.checked_predictions[(predicted, actual)] += 1
        for length in range(min(len(context), self.max_context) + 1):
            key = tuple(context[-length:]) if length else ()
            self.patterns[key][actual] += 1
        self.observations += 1
        return self

    def generate(self, context, steps=1):
        result=[]; history=list(context)
        for _ in range(steps):
            token=self.predict(history)['prediction']
            if token is None: break
            result.append(token);history.append(token)
        return result

class NumericPatternSelf:
    """Explicit difference representation; learner discovers observed changes."""
    def __init__(self, max_context=4):
        self.self = PatternSelf(max_context)
    @staticmethod
    def changes(sequence):
        return [b-a for a,b in zip(sequence, sequence[1:])]
    def observe(self, sequence):
        self.self.observe(self.changes(list(sequence)));return self
    def predict(self, sequence):
        sequence=list(sequence)
        if len(sequence)<2: return None
        change=self.self.predict(self.changes(sequence))['prediction']
        return None if change is None else sequence[-1]+change
    def feedback(self, sequence, actual):
        sequence=list(sequence)
        if len(sequence)<2: raise ValueError('Need at least two observations')
        self.self.feedback(self.changes(sequence),actual-sequence[-1]);return self
