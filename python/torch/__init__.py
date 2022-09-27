import os
os.environ["FIX_TORCH_ERROR"] = "0"

import jittor as jt
import jtorch
from jtorch import *

import sys
def load_mod(name):
    exec("import "+name)
    return eval(name)

sys.modules["torch.nn"] = load_mod("jtorch.nn")
sys.modules["torch.utils"] = load_mod("jtorch.utils")
sys.modules["torch.utils.data"] = load_mod("jtorch.utils.data")
sys.modules["torchvision"] = load_mod("jtorch.vision")
sys.modules["torchvision.datasets"] = load_mod("jtorch.vision.datasets")
sys.modules["torchvision.transforms"] = load_mod("jtorch.vision.transforms")
