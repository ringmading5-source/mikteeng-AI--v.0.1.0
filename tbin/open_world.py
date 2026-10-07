"""Open-world crystallization from TBIN v0.7-v0.9 experiments."""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence
from .core import BraidEncoder, medoid, word_distance, transition_signature, transition_distance

@dataclass
class Candidate:
    observations: List[tuple] = field(default_factory=list)
    transitions: List[tuple] = field(default_factory=list)
    times: List[int] = field(default_factory=list)

class OpenWorldTBIN:
    """Incremental prototype with predictive-transition-gated crystallization.

    A state crystallizes only after recurrence, global structural compactness,
    transition compactness, and temporal persistence. This is experimental and
    does not establish semantic understanding.
    """
    def __init__(self, known=None, encoder=None, accept=0.30, cluster_radius=0.30,
                 transition_radius=0.18, crystallize_count=7, compactness=0.22,
                 min_age=10):
        self.encoder = encoder or BraidEncoder()
        self.known: Dict[str, tuple] = dict(known or {})
        self.crystals: Dict[str, tuple] = {}
        self.candidates: List[Candidate] = []
        self.accept = accept
        self.cluster_radius = cluster_radius
        self.transition_radius = transition_radius
        self.crystallize_count = crystallize_count
        self.compactness = compactness
        self.min_age = min_age
        self.tick = 0
        self.next_id = 1

    def persistent(self):
        return {**self.known, **self.crystals}

    def nearest(self, word):
        pool = self.persistent()
        if not pool:
            return (1.0, None)
        return min((word_distance(word, proto), name) for name, proto in pool.items())

    def ingest(self, values: Sequence[float]):
        self.tick += 1
        word = self.encoder.encode(values)
        trans = transition_signature(word)
        distance, name = self.nearest(word)
        if name is not None and distance <= self.accept:
            return {"status":"RECOGNIZED", "state":name, "distance":distance}

        best = None
        for i, candidate in enumerate(self.candidates):
            center = medoid(candidate.observations, word_distance)
            tcenter = medoid(candidate.transitions, transition_distance)
            dw = word_distance(word, center)
            dt = transition_distance(trans, tcenter)
            if dw <= self.cluster_radius and dt <= self.transition_radius:
                score = dw + dt
                if best is None or score < best[0]:
                    best = (score, i)

        if best is None:
            self.candidates.append(Candidate([word], [trans], [self.tick]))
            return {"status":"PROVISIONAL"}

        candidate = self.candidates[best[1]]
        candidate.observations.append(word)
        candidate.transitions.append(trans)
        candidate.times.append(self.tick)
        center = medoid(candidate.observations, word_distance)
        tcenter = medoid(candidate.transitions, transition_distance)
        spread = sum(word_distance(center, x) for x in candidate.observations)/len(candidate.observations)
        tspread = sum(transition_distance(tcenter, x) for x in candidate.transitions)/len(candidate.transitions)
        age = candidate.times[-1] - candidate.times[0]

        if (len(candidate.observations) >= self.crystallize_count
                and spread <= self.compactness
                and tspread <= self.transition_radius
                and age >= self.min_age):
            state = f"CRYSTAL_{self.next_id}"
            self.next_id += 1
            self.crystals[state] = center
            self.candidates.pop(best[1])
            return {"status":"CRYSTALLIZED", "state":state,
                    "spread":spread, "transition_error":tspread, "age":age}
        return {"status":"PROVISIONAL", "spread":spread,
                "transition_error":tspread, "age":age}
