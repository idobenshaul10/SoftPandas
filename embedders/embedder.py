from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer, CrossEncoder, util


class Embedder(ABC):
    @abstractmethod
    def encode(self, text):
        pass


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def encode(self, text):
        return self.model.encode(text)
