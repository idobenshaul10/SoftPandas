import open_clip
import torch

from core.utils import is_url, load_image_from_url
from embedders.embedder import Embedder
import numpy as np
import re
from PIL import Image
import requests
from io import BytesIO


class OpenClipEmbedder(Embedder):
    def __init__(self, model_name, metric, threshold, pretrained):
        super().__init__(model_name, metric, threshold)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name,
                                                                               pretrained=pretrained,
                                                                               device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode(self, data):
        if not self.check_validity(data):
            return np.nan

        with torch.no_grad():
            if is_url(data) or type(data) == Image:
                if is_url(data):
                    data = load_image_from_url(data)
                data = self.preprocess(data).unsqueeze(0)#.to(self.device)
                embs = self.model.encode_image(data)

            else:
                data = self.tokenizer([data])#.to(self.device)
                embs = self.model.encode_text(data)

        embs = embs.cpu().numpy().squeeze()
        return embs
