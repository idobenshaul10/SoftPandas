import torch
import re
from PIL import Image
import requests
from io import BytesIO
import base64


def get_device():
    if torch.cuda.is_available():
        res = "cuda"
    elif torch.backends.mps.is_available():
        res = "mps"
    else:
        res = "cpu"
    return torch.device(res)


def is_url(string):
    if not isinstance(string, str):
        return False
    return re.match(r'https?://', string) is not None


def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image


def get_thumbnail(image, size=(100, 100)):
    return image.resize(size)


def image_base64(im):
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}" style="max-width: 100px; max-height: 100px;">'
