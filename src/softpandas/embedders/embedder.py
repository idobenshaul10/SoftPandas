from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import torch
from softpandas.core.utils import get_device


class Embedder(ABC):
    def __init__(self, model_name: str, metric: Callable[[np.array, np.array], float], threshold: float,
                 device: str = None, **kwargs):
        self.model_name = model_name
        self.metric = metric
        self.threshold = threshold
        if device:
            self.device = torch.device(device)
        else:
            self.device = get_device()
        self.kwargs = kwargs

    def check_validity(self, data):
        if not isinstance(data, str):
            print(f"Expected a string, got {type(data)}")
            return False
        return True

    @abstractmethod
    def encode(self, data):
        pass



