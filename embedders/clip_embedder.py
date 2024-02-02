import open_clip
import torch
from embedders.embedder import Embedder
import numpy as np
import re
from PIL import Image
import requests
from io import BytesIO


def is_url(string):
    return re.match(r'https?://', string) is not None


def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image


class OpenClipEmbedder(Embedder):
    def __init__(self, model_name, metric, threshold):
        super().__init__(model_name, metric, threshold)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name,
                                                                               pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode(self, data):
        if not self.check_validity(data):
            return np.nan

        if is_url(data) or type(data) == Image:
            if is_url(data):
                data = load_image_from_url(data)
            data = self.preprocess(data).unsqueeze(0)
            embs = self.model.encode_image(data)

        else:
            data = self.tokenizer([data])
            embs = self.model.encode_text(data)

        embs /= embs.norm(dim=-1, keepdim=True)
        return embs