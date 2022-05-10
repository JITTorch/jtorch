import os
os.environ["FIX_TORCH_ERROR"] = "0"

import jittor as jt
from jittor import *

import jtorch.compiler

from jtorch_core import *

def linspace(start, end, steps, dtype="float32", device=None):
    return jt.linspace(start, end, steps).astype(dtype)

def randn(*args, device=None, **kw):
    return jt.randn(*args, **kw)
