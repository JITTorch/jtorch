import jtorch
import jittor as jt

from jtorch import make_module, Tensor

for k,v in jt.nn.__dict__.items():
    if callable(v):
        globals()[k] = v

for k,v in jt.nn.__dict__.items():
    if isinstance(v, type) and issubclass(v, jt.Module):
        globals()[k] = make_module(v)

from jtorch.nn.modules import Module

def Parameter(x:Tensor, requires_grad:bool=True) -> Tensor:
    x = x.clone()
    x.requires_grad = requires_grad
    x.retains_grad = requires_grad
    return x

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return jt.nn.embedding(input, weight)

def dropout(x, p=0.5, training=False):
    return jt.nn.dropout(x, p, training)
