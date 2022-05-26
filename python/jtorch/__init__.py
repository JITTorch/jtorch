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

Var.backward = lambda x: jtorch_core.backward(x)
Var.grad = property(grad_get, grad_set, grad_del)
Var.retain_grad = property(retain_grad_get, retain_grad_set)

from . import autograd
from .autograd import *

Tensor = Var
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
                    v.retain_grad = True
        def __call__(self, *args, **kw):
            return self.execute(*args, **kw)
        def forward(self, *args, **kw):
            return self.execute(*args, **kw)
    return TMod

import jtorch.nn
from jtorch.nn import Module, Parameter
import jtorch.optim
