"""Real-speech TBIN falsification harness.

Dataset layout:
data/tbin_speech/<concept>/<speaker>/*.wav

Example:
data/tbin_speech/water/speaker01/001.wav
data/tbin_speech/water/speaker02/001.wav
data/tbin_speech/fire/speaker01/001.wav

No semantic labels are used inside an utterance representation; directory labels
are used only to evaluate whether structural distances reflect concept identity.
"""
import argparse
from collections import defaultdict
from pathlib import Path

from tbin.audio import acoustic_trajectory
from tbin.invariants import descriptor_encode, relational_features, stable_identity, jaccard_distance

def load(root):
    items=[]
    for wav in sorted(Path(root).glob("*/*/*.wav")):
        concept=wav.parent.parent.name
        speaker=wav.parent.name
        trajectory=acoustic_trajectory(wav)
        if trajectory:
            features=relational_features(descriptor_encode(trajectory))
            items.append((concept,speaker,wav,features))
    return items

def evaluate(items):
    if not items:
        raise SystemExit("No WAV files found. Expected <concept>/<speaker>/*.wav")
    grouped=defaultdict(list)
    for concept,speaker,path,features in items:
        grouped[(concept,speaker)].append(features)

    signatures={
        key:stable_identity(views, support=.60)
        for key,views in grouped.items()
    }

    same=[]; different=[]
    keys=list(signatures)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            d=jaccard_distance(signatures[a],signatures[b])
            if a[0]==b[0] and a[1]!=b[1]:
                same.append(d)
            elif a[0]!=b[0]:
                different.append(d)

    print(f"utterances: {len(items)}")
    print(f"concepts: {len(set(x[0] for x in items))}")
    print(f"speakers: {len(set(x[1] for x in items))}")
    if not same:
        print("same-concept cross-speaker distance: unavailable (need >=2 speakers/concept)")
    else:
        print(f"mean same-concept cross-speaker distance: {sum(same)/len(same):.4f}")
    if not different:
        print("different-concept distance: unavailable (need >=2 concepts)")
    else:
        print(f"mean different-concept distance: {sum(different)/len(different):.4f}")
    if same and different:
        margin=sum(different)/len(different)-sum(same)/len(same)
        print(f"separation margin: {margin:.4f}")
        print("PASS (positive separation)" if margin>0 else "FAIL (no positive separation)")
    print("\nInterpretation: this tests structural separation only; it is not speech recognition.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",nargs="?",default="data/tbin_speech")
    args=p.parse_args()
    evaluate(load(args.root))
