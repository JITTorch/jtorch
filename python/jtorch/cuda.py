import jittor as jt
import jtorch

def is_available():
    return jt.has_cuda

def device_count():
    return int(jt.has_cuda)

def set_device(device=None):
    pass

def get_rng_state(device=None):
    pass

def current_device():
    return jtorch.device("cuda")

def mem_get_info(i):
    return ("75GB",)


class Generator:
    def __init__(self):
        pass

    def set_state(self, state):
        self.state = state

default_generators = [Generator()]
_lazy_call = lambda func: func()
device = None

LongTensor = jt.int64
FloatTensor = jt.float
HalfTensor = jt.float16
BoolTensor = jt.bool

manual_seed = jt.set_global_seed
manual_seed_all = jt.set_global_seed

class Event:
    pass

class Stream:
    pass

from typing import Any

from .gradscaler import GradScaler
# class GradScaler:
    
#     def scale(self,outputs):
#         return outputs
    
#     def unscale_(self,optimizer):
#         pass

#     def get_scale(self,):
#         return 1.0
    
#     def step(self,optimizer,*args,**kwargs):
#         optimizer.step(*args,**kwargs)

#     def update(self,new_scale=None):
#         pass

class autocast:
    def __init__(self,**kwargs):
        pass 

    def __enter__(self,):
        pass 

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        pass


