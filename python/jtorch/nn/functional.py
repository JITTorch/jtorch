import jtorch
import jittor as jt
from jtorch.nn import Module

relu = jt.nn.relu
relu6 = jt.nn.relu6
elu = jt.nn.elu

def leaky_relu(x, scale):
    return jt.nn.leaky_relu(x, scale)

def celu(x):
    raise NotImplementedError

def selu(x):
    raise NotImplementedError

log_softmax = jt.nn.log_softmax