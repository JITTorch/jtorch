import os
os.environ["FIX_TORCH_ERROR"] = "0"

import jittor as jt
from jittor import *
int = type(1)
float = type(1.0)
bool = type(True)

import jtorch.compiler

import jtorch_core
from jtorch_core import *

def wrapper(func):
    def inner(*args, **kw):
        requires_grad = None
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
        if requires_grad is not None:
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

Tensor.backward = lambda x, create_graph=False: jtorch_core.backward(x)
Tensor.grad = property(grad_get, grad_set, grad_del)
Tensor.retains_grad = property(retain_grad_get, retain_grad_set)
def retain_grad(x:Tensor, value:bool=True):
    x.retains_grad = value
    return value
Tensor.retain_grad = retain_grad

Tensor.device = None
Tensor.to = lambda self, device, non_blocking=False: self
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

import jtorch.cuda as cuda
import jtorch.nn
import jtorch._six
from jtorch.nn import Module, Parameter
import jtorch.optim
import jtorch.distributed as distributed
import jtorch.jit as jit


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

def conflict_wrapper(origin_func, new_func):
    def wrapper(*args, **kw):
        if jt.flags.th_mode:
            return new_func(*args, **kw)
        else:
            return origin_func(*args, **kw)
    return wrapper

def min(*args, **kw):
    dim = None
    if len(args) >= 2 and isinstance(args[1], int):
        dim = args[1]
    elif "dim" in kw and isinstance(kw["dim"], int):
        dim = kw["dim"]
    if dim is not None:
        k, v = jt.argmin(*args, **kw)
        return v, k
    else:
        return jt.min(*args, **kw)
Tensor.min = conflict_wrapper(jt.min, min)

def max(*args, **kw):
    dim = None
    if "dim" in kw:
        x = kw["dim"]
    if len(args) >= 2 and isinstance(args[1], int):
        dim = args[1]
    elif "dim" in kw and isinstance(kw["dim"], int):
        dim = kw["dim"]
    if dim is not None:
        k, v = jt.argmax(*args, **kw)
        return v, k
    else:
        return jt.max(*args, **kw)
Tensor.max = conflict_wrapper(jt.max, max)

Tensor.mul_ = jt.multiply_

def argsort(*args, **kw):
    k, v = jt.argsort(*args, **kw)
    return k
Tensor.argsort = conflict_wrapper(jt.argsort, argsort)

def clamp_(*args, **kw):
    new_kw = {}
    if "min" in kw:
        new_kw["min_v"] = kw["min"]
    if "max" in kw:
        new_kw["max_v"] = kw["max"]
    return jt.clamp_(*args,**new_kw) 
Tensor.clamp_ = conflict_wrapper(jt.clamp_, clamp_)
Tensor.eq = jt.Var.equal
cat = jt.concat

def manual_seed(seed):
    jt.set_global_seed(seed)

def load_(*args, **kw):
    return jt.load(args[0])

load = conflict_wrapper(jt.load, load_)

def empty(shape, dtype, device=None):
    return jt.empty(shape, dtype)

def rand(shape, dtype, device=None):
    return jt.rand(shape, dtype)

def normal_(x, mean=0, std=1):
    return (x - mean) / std

def __floor(x):
    return x.floor

def div_(x, y):
    if isinstance(y, jt.Var):
        return jt.divide(x, y)
    return x / y

Tensor.floor_ = __floor
Tensor.normal_ = normal_
Tensor.div = div_