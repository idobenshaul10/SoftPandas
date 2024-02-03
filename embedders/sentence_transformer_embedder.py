from sentence_transformers import SentenceTransformer
from embedders.embedder import Embedder
import numpy as np
from typing import Callable


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, metric: Callable[[np.array, np.array], float],
                 threshold: float, device: str = None):
        super().__init__(model_name, metric, threshold, device)
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, data):
        if not self.check_validity(data):
            return np.nan
        return self.model.encode(data)
