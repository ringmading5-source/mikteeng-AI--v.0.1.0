"""Import compatibility for the renamed package; legacy learners were retired."""
from mikteeng_rspm import MikteengRSPM, BoundedRSPM, load, save, __version__
__all__ = ['MikteengRSPM', 'BoundedRSPM', 'load', 'save']
