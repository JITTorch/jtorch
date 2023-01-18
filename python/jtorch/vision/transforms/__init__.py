from jittor.transform import *

def _compute_resized_output_size(image_size, size, max_size=None):
    if len(size) == 1:
        h, w = image_size
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = size if isinstance(size, int) else size[0]

        new_short, new_long = requested_new_short, int(requested_new_short * long / short)

        if max_size is not None:
            if max_size <= requested_new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short / new_long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    else:  # specified both h and w
        new_w, new_h = size[1], size[0]
    return [new_h, new_w]

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = (np.array(img).transpose((2,0,1)) \
                - self.mean*np.float32(255.)) \
                * (np.float32(1./255.)/self.std)
        else:
            if not isinstance(self.mean, jt.Var):
                self.mean = jt.array(self.mean).unsqueeze(-1).unsqueeze(-1)
                self.std = jt.array(self.std).unsqueeze(-1).unsqueeze(-1)
            if len(img.shape) != len(self.mean.shape):
                self.mean = self.mean.unsqueeze(-1).unsqueeze(-1)
                self.std = self.std.unsqueeze(-1).unsqueeze(-1)
            img = (img - self.mean) / self.std
        return img

def resize(img, size, interpolation=Image.BILINEAR):
    if not ((isinstance(size, list) or isinstance(size, tuple)) and len(size) == 2):
        raise TypeError(f"Got inappropriate size arg: {size}")
    return img.resize(tuple(size[::-1]), interpolation)

class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR, max_size=None):
        if isinstance(size, int): size = [size]
        self.size = size
        self.mode = interpolation
        self.max_size=max_size
    
    def __call__(self, img:Image.Image):
        w, h = img.size
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        size = _compute_resized_output_size((h, w), self.size, self.max_size)
        return resize(img, size, self.mode)