"""Validate the single RSPM checkpoint; no legacy learner or build-time training."""
from pathlib import Path
from mikteeng_rspm import load
path=Path(__file__).resolve().parents[2]/'models/synthetic_vector_demo.json'
model=load(path)
print(f'Mikteeng RSPM checkpoint ready, dimension {model.config["dim"]}')
