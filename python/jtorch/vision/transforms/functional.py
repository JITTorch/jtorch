import jittor as jt
from jittor.transform import crop
from jtorch.vision.transforms import resize

def resized_crop(
    img, top, left, height, width, size, interpolation, antialias=False):
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img