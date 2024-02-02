from sentence_transformers import SentenceTransformer
from embedders.embedder import Embedder
import numpy as np

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name, metric, threshold):
        super().__init__(model_name, metric, threshold)
        self.model = SentenceTransformer(self.model_name)

    def encode(self, data):
        if not self.check_validity(data):
            return np.nan
        return self.model.encode(data)