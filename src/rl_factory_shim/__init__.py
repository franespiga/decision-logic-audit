# Namespace package shim so SB3 can unpickle CustomCNN as rl_factory.utils.feature_extractors.CustomCNN
from .feature_extractors import CustomCNN

__all__ = ["CustomCNN"]
