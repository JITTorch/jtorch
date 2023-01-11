from jittor.transform import *

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
            img = (img - self.mean) / self.std
        return img
