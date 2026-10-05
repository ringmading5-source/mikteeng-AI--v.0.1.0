"""Mikteeng RSPM: one fixed-basis context-local vector learner."""
from .engine import BoundedRSPM, assert_exact
from .checkpoint import load, save
MikteengRSPM = BoundedRSPM
__version__ = '0.2.0'
__all__ = ['MikteengRSPM', 'BoundedRSPM', 'assert_exact', 'load', 'save']
