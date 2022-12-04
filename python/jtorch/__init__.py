import os
os.environ["FIX_TORCH_ERROR"] = "0"

import jittor as jt
from jittor import *

import jtorch.compiler

import jtorch_core
from jtorch_core import *

def wrapper(func):
    def inner(*args, **kw):
        requires_grad = False
        dtype = None
        if "requires_grad" in kw:
            requires_grad = kw["requires_grad"]
            del kw["requires_grad"]
        if "dtype" in kw:
            dtype = kw["dtype"]
            del kw["dtype"]
        if "device" in kw:
            del kw["device"]
        ret = func(*args, **kw)
        ret.requires_grad = requires_grad
        if dtype is not None:
            ret.astype(dtype)
        return ret
    return inner
        

import inspect
_wrapper_keys = set(["shape", "start", "size"])
for k,v in list(globals().items()):
    if callable(v) and not isinstance(v, type):
        try:
            spec = inspect.getfullargspec(v)
            args_name = spec[0]
            if len(args_name) and args_name[0] in _wrapper_keys:
                globals()[k] = wrapper(v)
            elif spec.varargs in _wrapper_keys:
                globals()[k] = wrapper(v)
        except:
            pass

Tensor = Var

Tensor.backward = lambda x: jtorch_core.backward(x)
Tensor.grad = property(grad_get, grad_set, grad_del)
Tensor.retains_grad = property(retain_grad_get, retain_grad_set)
def retain_grad(x:Tensor, value:bool=True):
    x.retains_grad = value
    return value
Tensor.retain_grad = retain_grad

Tensor.to = lambda self, device: self
Tensor.ndimension = lambda self: self.ndim

def argmax(x: Var, dim=None, keepdim: bool = False):
    return jt.argmax(x, dim, keepdim)[0]
Tensor.argmax = argmax

def tensor_type(x: Var, dtype=None, **kwargs):
    if dtype:
        return x.astype(dtype)
    else:
        return x.dtype
Tensor.type = tensor_type

from . import autograd
from .autograd import *

tensor = wrapper(array)

def mod_zero_grad(self):
    for p in self.parameters():
        p.grad = None
Module.zero_grad = mod_zero_grad

def make_module(cls):
    class TMod(cls):
        def __init__(self, *args, **kw):
            with jt.flag_scope(th_mode=0):
                super().__init__(*args, **kw)
            for k,v in self.__dict__.items():
                if not k.startswith("_") and isinstance(v, Var) \
                    and v.requires_grad:
                    v.retain_grad()
        def __call__(self, *args, **kw):
            return self.execute(*args, **kw)
        def forward(self, *args, **kw):
            return self.execute(*args, **kw)
    return TMod

import jtorch.cuda
import jtorch.nn
from jtorch.nn import Module, Parameter
import jtorch.optim

from jtorch.utils.dtype import Dtype, get_string_dtype

def frombuffer(buffer: bytearray, 
              *, 
              dtype: Dtype, 
              count: int = -1, 
              offset: int = 0, 
              requires_grad: bool = True) -> Tensor:
    dtype = get_string_dtype(dtype)
    tensor = jt.array(np.frombuffer(buffer, dtype, count=count, offset=offset))
    if requires_grad and tensor.dtype.is_float():
        tensor.requires_grad = True
    return tensor
