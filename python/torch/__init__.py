import os
os.environ["FIX_TORCH_ERROR"] = "0"

import jittor as jt
import jtorch
from jtorch import *
__version__ = "2.0.0"

import sys
def load_mod(name):
    exec("import "+name)
    return eval(name)

autograd = sys.modules["torch.autograd"] = load_mod("jtorch.autograd")
sys.modules["torch.cuda"] = load_mod("jtorch.cuda")
sys.modules["torch.nn"] = load_mod("jtorch.nn")
sys.modules["torch.nn.functional"] = load_mod("jtorch.nn")
sys.modules["torch.nn.modules"] = load_mod("jtorch.nn")
jtorch.nn.module = jtorch.nn
sys.modules["torch.nn.parameter"] = load_mod("jtorch.nn")
jtorch.nn.parameter = jtorch.nn
sys.modules["torch.nn.utils"] = load_mod("jtorch.nn")
jtorch.nn.functional = jtorch.nn
sys.modules["torch.utils"] = load_mod("jtorch.utils")
sys.modules["torch._utils"] = load_mod("jtorch.utils")
sys.modules["torch.utils.data"] = load_mod("jtorch.utils.data")
sys.modules["torch.utils.data.sampler"] = load_mod("jtorch.utils.data")
jtorch.utils.data.sampler = jtorch.utils.data
sys.modules["torch.utils.checkpoint"] = load_mod("jtorch.utils.checkpoint")

distributed = sys.modules["torch.distributed"] = load_mod("jtorch.distributed")
sys.modules["torch.nn.parallel"] = load_mod("jtorch.distributed")
sys.modules["torch.nn.parallel.distributed"] = load_mod("jtorch.distributed")
_C = sys.modules["torch._C"] = load_mod("jtorch.misc")
_six = sys.modules["torch._six"] = load_mod("jtorch.misc")
jit = sys.modules["torch.jit"] = load_mod("jtorch.misc")

sys.modules["torchvision"] = load_mod("jtorch.vision")
sys.modules["torchvision.datasets"] = load_mod("jtorch.vision.datasets")
sys.modules["torchvision.transforms"] = load_mod("jtorch.vision.transforms")
sys.modules["torchvision.transforms.functional"] = load_mod("jtorch.vision.transforms")
jtorch.vision.transforms.functional = jtorch.vision.transforms
