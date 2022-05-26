import jtorch
import jittor as jt

from jtorch import make_module, Tensor

for k,v in jt.nn.__dict__.items():
    if isinstance(v, type) and issubclass(v, jt.Module):
        globals()[k] = make_module(v)

class Module(jt.Module):
    def __call__(self, *args, **kw):
        return self.forward(*args, **kw)

def Parameter(x:Tensor, requires_grad:bool=True) -> Tensor:
    x = x.clone()
    x.requires_grad = requires_grad
    x.retain_grad = requires_grad
    return x
