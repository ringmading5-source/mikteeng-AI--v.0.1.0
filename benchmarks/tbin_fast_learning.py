"""TBIN fast-learning benchmark.

Measures:
1. examples-to-crystallization for recurring novel structure,
2. CPU time per observation,
3. old-state retention after learning,
4. false crystallization under random noise.

This is a synthetic engineering benchmark, not evidence of real-language learning speed.
"""
import random
import statistics
import time

from tbin import BraidEncoder, OpenWorldTBIN

SEED=1207
rng=random.Random(SEED)
encoder=BraidEncoder(window=3, basin=18, strands=11)

def latent():
    return [rng.uniform(-1,1) for _ in range(10)]

def world(z, noise=2.0):
    return [
        max(0,min(255,128+52*z[i % len(z)]+rng.gauss(0,noise)))
        for i in range(36)
    ]

def representative(z, n=30):
    words=[encoder.encode(world(z)) for _ in range(n)]
    def d(a,b):
        L=max(len(a),len(b),1)
        k=min(len(a),len(b))
        return (sum(a[i]!=b[i] for i in range(k))+abs(len(a)-len(b)))/L
    return min(words,key=lambda a:sum(d(a,b) for b in words))

known_latents=[latent() for _ in range(5)]
known={f"KNOWN_{i}":representative(z) for i,z in enumerate(known_latents)}
model=OpenWorldTBIN(known=known,encoder=encoder)

novel=[latent() for _ in range(8)]
examples_to_crystal=[]
times=[]

for z in novel:
    found=None
    for exposure in range(1,61):
        t0=time.perf_counter_ns()
        result=model.ingest(world(z))
        times.append(time.perf_counter_ns()-t0)
        # Interleave a known state so temporal persistence is real.
        model.ingest(world(known_latents[exposure % len(known_latents)]))
        if result["status"]=="CRYSTALLIZED":
            found=exposure
            break
        if result["status"]=="RECOGNIZED" and str(result.get("state","")).startswith("CRYSTAL_"):
            found=exposure
            break
    examples_to_crystal.append(found)

# Retention: nearest persistent state should still be original known state.
retained=0
trials=500
for i in range(trials):
    k=i % len(known_latents)
    word=encoder.encode(world(known_latents[k]))
    _,name=model.nearest(word)
    retained += name==f"KNOWN_{k}"

# Noise test uses a fresh learner with same known states.
noise_model=OpenWorldTBIN(known=known,encoder=encoder)
false_events=0
for _ in range(2000):
    noise=[rng.uniform(0,255) for _ in range(36)]
    r=noise_model.ingest(noise)
    false_events += r["status"]=="CRYSTALLIZED"

learned=[x for x in examples_to_crystal if x is not None]
print("TBIN FAST-LEARNING BENCHMARK (synthetic)")
print("seed:",SEED)
print("novel patterns:",len(novel))
print("crystallized:",len(learned),"/",len(novel))
print("examples to crystallization:",examples_to_crystal)
if learned:
    print("median examples:",statistics.median(learned))
    print("mean examples:",round(statistics.mean(learned),3))
print("mean CPU time/novel observation (ms):",round(statistics.mean(times)/1e6,6))
print("p95 CPU time (ms):",round(sorted(times)[int(.95*(len(times)-1))]/1e6,6))
print("old-state retention:",round(retained/trials,4))
print("false noise crystals:",false_events,"/ 2000")
print("persistent learned crystals:",len(model.crystals))
print("\nCAUTION: synthetic benchmark only. Real-speech sample efficiency remains unmeasured.")
