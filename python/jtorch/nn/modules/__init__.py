import jittor as jt


class Module(jt.Module):
    def __init__(self, *args, **kw) -> None:
        super(Module, self).__init__(*args, **kw)

    def __call__(self, *args, **kw):
        return self.forward(*args, **kw)

    def to(self, device):
        ''' do nothing but return its self'''
        return self

    def cuda(self, device=None):
        jt.flags.use_device = 1

    def load_state_dict(self, state_dict, strict=None):
        return super().load_state_dict(state_dict)

    @property
    def training(self):
        if not hasattr(self, "is_train"):
            self.is_train = True
        return self.is_train
    @training.setter
    def training(self, value):
        self.is_train = value
