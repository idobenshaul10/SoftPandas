import open_clip
import torch
from softpandas.core.utils import is_url, load_image_from_url
from softpandas.embedders.embedder import Embedder
from typing import Callable
import numpy as np
from PIL import Image


class OpenClipEmbedder(Embedder):
    def __init__(self, model_name: str, metric: Callable[[np.array, np.array], float],
                 threshold: float, pretrained: str, device: str = None):
        super().__init__(model_name, metric, threshold, device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name,
                                                                               pretrained=pretrained,
                                                                               device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.embedding_size = self.model.text_projection.shape[1]

    def encode(self, data):
        if not self.check_validity(data):
            return np.nan

        with torch.no_grad():
            if is_url(data) or type(data) == Image:
                if is_url(data):
                    data = load_image_from_url(data)
                data = self.preprocess(data).unsqueeze(0).to(self.device)
                embs = self.model.encode_image(data)

            else:
                data = self.tokenizer([data]).to(self.device)
                embs = self.model.encode_text(data)

        embs = embs.cpu().numpy().squeeze()
        return embs
