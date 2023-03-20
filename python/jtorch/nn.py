import jtorch
import jittor as jt

from jtorch import make_module, Tensor, ModuleMisc

for k,v in jt.nn.__dict__.items():
    if isinstance(v, type) and issubclass(v, jt.Module):
        globals()[k] = make_module(v)
    elif callable(v):
        globals()[k] = v

class Module(ModuleMisc, jt.Module):
    def __call__(self, *args, **kw):
        return self.forward(*args, **kw)

    def execute(self, *args, **kw):
        return self.forward(*args, **kw)

    def to(self, device):
        ''' do nothing but return its self'''
        return self

def Parameter(x:Tensor, requires_grad:bool=True) -> Tensor:
    x = x.clone()
    x.requires_grad = requires_grad
    x.retains_grad = requires_grad
    return x


class Flatten(Module):
    ''' Flattens the contiguous range of dimensions in a Var.
    :param start_dim: the first dimension to be flattened. Defaults: 1.
    :type start_dim: int
    :param end_dim: the last dimension to be flattened. Defaults: -1.
    :type end_dim: int
    '''
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x) -> jt.Var:
        return x.flatten(self.start_dim, self.end_dim)