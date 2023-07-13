import jtorch
import jittor as jt

from jtorch import make_module, Tensor, ModuleMisc, wrapper

for k,v in jt.nn.__dict__.items():
    if callable(v):
        globals()[k] = wrapper(v)

for k,v in jt.nn.__dict__.items():
    if isinstance(v, type) and issubclass(v, jt.Module):
        globals()[k] = make_module(v)

class Module(ModuleMisc, jt.Module):
    
    def __call__(self, *args, **kw):
        return self.forward(*args, **kw)

    def execute(self, *args, **kw):
        return self.forward(*args, **kw)


    

def Parameter(x:Tensor, requires_grad:bool=True) -> Tensor:
    x = x.clone()
    x.requires_grad = requires_grad
    x.retains_grad = requires_grad
    return x

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return jt.nn.embedding(input, weight)

def dropout(x, p=0.5, training=False):
    return jt.nn.dropout(x, p, training)


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

_BatchNorm = None

from . import utils