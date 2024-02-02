from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


class Embedder(ABC):
    def __init__(self, model_name: str, metric: Callable[[np.array, np.array], float], threshold: float):
        self.model_name = model_name
        self.metric = metric
        self.threshold = threshold

    def check_validity(self, text):
        if not isinstance(text, str):
            print(f"Expected a string, got {type(text)}")
            return False
        return True

    @abstractmethod
    def encode(self, text):
        pass



