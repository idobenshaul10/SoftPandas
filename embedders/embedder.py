from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from typing import Callable
import numpy as np


class Embedder(ABC):
    def __init__(self, model_name: str, metric: Callable[[np.array, np.array], float], threshold: float):
        self.model_name = model_name
        self.metric = metric
        self.threshold = threshold

    @abstractmethod
    def encode(self, text):
        pass


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name, metric, threshold):
        super().__init__(model_name, metric, threshold)
        self.model = SentenceTransformer(self.model_name)

    def encode(self, text):
        return self.model.encode(text)
