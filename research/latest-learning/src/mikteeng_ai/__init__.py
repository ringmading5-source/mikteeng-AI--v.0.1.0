"""mikteeng AI: experimental learned prediction components."""
__version__ = '0.1.0'
from .api import MikteengAI, RoleSession
__all__ = ['MikteengAI', 'RoleSession', '__version__']

from .modalities import ModalityEncoder, MultimodalLearner, TokenStream
__all__ += ["ModalityEncoder", "MultimodalLearner", "TokenStream"]

from .meaning import MeaningLearner
__all__ += ["MeaningLearner"]

from .hierarchy import HierarchicalLearner
__all__ += ["HierarchicalLearner"]

from .self_model import PredictedSelf
__all__ += ["PredictedSelf"]

from .adaptive import AdaptiveSelf
__all__ += ["AdaptiveSelf"]

from .prediction_patterns import PredictionPatternLearner, PredictionPatternSelf
__all__ += ["PredictionPatternLearner", "PredictionPatternSelf"]

from .vector_nodes import BytePacket, PatternNodeSpace
__all__ += ["BytePacket", "PatternNodeSpace"]

from .byte_tokenizer import MultimodalByteTokenizer, MultimodalPacket, BytePatternHierarchy
__all__ += ["MultimodalByteTokenizer", "MultimodalPacket", "BytePatternHierarchy"]

from .byte_prediction import ByteHierarchyPredictor
__all__ += ["ByteHierarchyPredictor"]

from .speech_prediction import AcousticEncoder, SpeechPredictionModel
__all__ += ["AcousticEncoder", "SpeechPredictionModel"]
