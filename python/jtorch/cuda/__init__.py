import jittor as jt
import jtorch.cuda.amp as amp

def is_available():
    return jt.has_cuda

def max_memory_allocated():
    # no effect?
    return jt.MemInfo().total_cuda_ram