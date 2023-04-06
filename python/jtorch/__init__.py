import os
os.environ["FIX_TORCH_ERROR"] = "0"

import jittor as jt
from jittor import *
org_int = int = type(1)
org_float = float = type(1.0)
org_bool = bool = type(True)

import jtorch.compiler

import jtorch_core
from jtorch_core import *


def wrapper(func):
    has_dtype = False
    if hasattr(func, "__code__"):
        has_dtype = "dtype" in func.__code__.co_varnames
    def inner(*args, **kw):
        requires_grad = None
        dtype = None
        if "requires_grad" in kw:
            requires_grad = kw["requires_grad"]
            del kw["requires_grad"]
        if not has_dtype and "dtype" in kw:
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

def empty(*size, dtype=jt.float32, device=None, requires_grad=False):
    if len(size) == 1 and not isinstance(size[0], org_int):
        size = size[0]
    return jt.empty(size, dtype)

Tensor = Var

Tensor.backward = lambda x: jtorch_core.backward(x)
Tensor.grad = property(grad_get, grad_set, grad_del)
Tensor.retains_grad = property(retain_grad_get, retain_grad_set)
def retain_grad(x:Tensor, value:bool=True):
    x.retains_grad = value
    return value
Tensor.retain_grad = retain_grad

Tensor.to = lambda self, device: self
Tensor.dim = lambda self: self.ndim
Tensor.ndimension = lambda self: self.ndim
Tensor.nelement = lambda self: self.numel()
Tensor.cuda = lambda self: self
def device_get(x:Tensor):
    return device("cpu") if not jt.has_cuda or not jt.flags.use_cuda else device("cuda")
Tensor.device = property(device_get)

# Need to modify jittor code
def get_data(x:Tensor):
    if jt.flags.th_mode:
        return x
    else:
        return x.get_data()

Tensor.data = property(get_data, Tensor.set_data)
Tensor.copy_ = lambda a, b: a.assign(b)

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

class ModuleMisc:
    def parameters(self):
        return iter(super().parameters())

    def load_state_dict(self, state_dict, strict=False):
        return super().load_state_dict(state_dict)

    def to(self, device):
        ''' do nothing but return its self'''
        return self

def make_module(cls):
    class TMod(ModuleMisc, cls):
        def __init__(self, *args, **kw):
            dtype = None
            if "dtype" in kw:
                dtype = kw["dtype"]
                del kw["dtype"]
            with jt.flag_scope(th_mode=0):
                super().__init__(*args, **kw)
            for k,v in self.__dict__.items():
                if not k.startswith("_") and isinstance(v, Var) \
                    and v.requires_grad:
                    v.retain_grad()
                if dtype is not None and isinstance(v, Var):
                    v.assign(v.cast(dtype))
        def __call__(self, *args, **kw):
            return self.execute(*args, **kw)
        def forward(self, *args, **kw):
            return self.execute(*args, **kw)
        
        @property
        def training(self):
            if not hasattr(self, "is_train"):
                self.is_train = True
            return self.is_train
        @training.setter
        def training(self, value):
            self.is_train = value

    TMod.__name__ = cls.__name__
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

def conflict_wrapper(origin_func, new_func):
    def wrapper(*args, **kw):
        if jt.flags.th_mode:
            return new_func(*args, **kw)
        else:
            return origin_func(*args, **kw)
    return wrapper

def min(*args, **kw):
    dim = None
    if len(args) >= 2 and isinstance(args[1], org_int):
        dim = args[1]
    elif "dim" in kw and isinstance(kw["dim"], org_int):
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
    if len(args) >= 2 and isinstance(args[1], org_int):
        dim = args[1]
    elif "dim" in kw and isinstance(kw["dim"], org_int):
        dim = kw["dim"]
    if dim is not None:
        k, v = jt.argmax(*args, **kw)
        return v, k
    else:
        return jt.max(*args, **kw)
Tensor.max = conflict_wrapper(jt.max, max)

def argsort(*args, **kw):
    k, v = jt.argsort(*args, **kw)
    return k
Tensor.argsort = conflict_wrapper(jt.argsort, argsort)

LongTensor = jt.int64
FloatTensor = jt.float
HalfTensor = jt.float16
BoolTensor = jt.bool

class JDType:
    def __init__(self, func, str):
        self.func = func
        self.str = str
        self.__name__ = str.split(".")[-1]
    def __call__(self, *args, **kw):
        return self.func(*args, **kw)
    def __str__(self):
        return self.str

int8 = JDType(jt.int8, "torch.int8")
int16 = JDType(jt.int16, "torch.int16")
int = int32 = JDType(jt.int32, "torch.int32")
long = int64 = JDType(jt.int64, "torch.int64")

half = float16 = JDType(jt.float16, "torch.float16")
float = float32 = JDType(jt.float32, "torch.float32")
double = float64 = JDType(jt.float64, "torch.float64")
bfloat16 = "bfloat16" # TODO

def load(path, map_location="cpu"):
    return jt.load(path)

def is_tensor(x):
    return isinstance(x, Tensor)

manual_seed = jt.set_global_seed
jt.flags.amp_level = 3

def zeros(size, dtype=None, device=None):
    if dtype is None:
        dtype = jt.float32
    return jt.zeros(size, dtype=dtype)

def full(fill_value, size, dtype=None, device=None):
    if dtype is None:
        dtype = jt.float32
    return jt.full(size, fill_value, dtype=dtype)

def unique_consecutive(inputs, return_inverse=False, return_counts=False, dim=None):
    """
    Eliminate consecutive duplicate values

    For example
    >>> x = jt.array([1, 1, 2, 2, 3, 1, 1, 2])
    >>> output, counts = unique_consecutive(x, return_counts=True)
    >>> output
    Var([1, 2, 3, 1, 2])
    >>> counts
    Var([2, 2, 1, 2, 1])
    """
    if dim is None:
        inputs = inputs.flatten()
        dim = 0

    y = inputs[(slice(None),) * dim + (slice(1, None),)] != inputs[(slice(None),) * dim + (slice(None, -1),)]
    y = jt.concat([jt.ones((1,) * dim, dtype=bool), y], dim=dim)
    ret = inputs[y]
    if return_inverse:
        raise NotImplementedError("return_inverse is not implemented yet")
    if return_counts:
        """ Eliminate consecutive duplicate values
        return the every element count of consecutive duplicate values appears in the inputs Var.
        For example
        >>> x = jt.array([1, 1, 2, 2, 3, 1, 1, 2])
        >>> output, counts = unique_consecutive(x, return_counts=True)
        >>> output
        Var([1, 2, 3, 1, 2])
        >>> counts
        Var([2, 2, 1, 2, 1])
        """
        counts = jt.zeros(ret.shape, dtype=jt.int32)
        cnt = 0
        for i in inputs:
            if i == ret[cnt]:
                counts[cnt] += 1
            else:
                cnt += 1
                counts[cnt] += 1
        return ret, counts
    return ret

def sum(x, dim=None, keepdim=False):
    if dim is None:
        return jt.sum(x)
    else:
        return jt.sum(x, dim=dim, keepdims=keepdim)