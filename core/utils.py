import torch
import re
from PIL import Image
import requests
from io import BytesIO


def get_device():
    if torch.cuda.is_available():
        res = "cuda"
    elif torch.backends.mps.is_available():
        res = "mps"
    else:
        res = "cpu"
    return torch.device(res)


def is_url(string):
    return re.match(r'https?://', string) is not None


def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image
