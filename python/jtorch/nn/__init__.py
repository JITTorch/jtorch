import jtorch
import jittor as jt
from jittor.nn import matmul_transpose
from jtorch import make_module, Tensor, ModuleMisc

from typing import Iterable, Iterator, Mapping, Optional, Tuple, Dict
from collections import OrderedDict, abc as container_abcs
from jittor import normalize

for k,v in jt.nn.__dict__.items():
    if callable(v):
        globals()[k] = v

for k,v in jt.nn.__dict__.items():
    if isinstance(v, type) and issubclass(v, jt.Module):
        globals()[k] = make_module(v)

class Module(ModuleMisc, jt.Module):
    def __call__(self, *args, **kw):
        return self.forward(*args, **kw)

    def execute(self, *args, **kw):
        return self.forward(*args, **kw)

    def add_module(self, name, module):
        self.__setattr__(name, module)

def Parameter(x:Tensor, requires_grad:bool=True) -> Tensor:
    x = x.clone()
    x.requires_grad = requires_grad
    x.retains_grad = requires_grad
    return x

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return jt.nn.embedding(input, weight)

def dropout(x, p=0.5, training=False):
    return jt.nn.dropout(x, p, training)


class Flatten(Module):
    ''' Flattens the contiguous range of dimensions in a Var.
    :param start_dim: the first dimension to be flattened. Defaults: 1.
    :type start_dim: int
    :param end_dim: the last dimension to be flattened. Defaults: -1.
    :type end_dim: int
    '''
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x) -> jt.Var:
        return x.flatten(self.start_dim, self.end_dim)

_BatchNorm = None


class ModuleDict(Module):
    # _modules: Dict[str, Module]  # type: ignore[assignment]

    def __init__(self, modules: Optional[Mapping[str, Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict.
        """
        self._modules.clear()

    def pop(self, key: str) -> Module:
        r"""Remove key from the ModuleDict and return its module.
        Args:
            key (str): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v


    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys.
        """
        return self._modules.keys()


    def items(self) -> Iterable[Tuple[str, Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs.
        """
        return self._modules.items()

    def values(self) -> Iterable[Module]:
        r"""Return an iterable of the ModuleDict values.
        """
        return self._modules.values()

    def update(self, modules: Mapping[str, Module]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.
        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.
        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, (OrderedDict, ModuleDict, container_abcs.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            # modules here can be a list with two items
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError("ModuleDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:
                    raise ValueError("ModuleDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(m)) +
                                     "; 2 is required")
                # modules can be Mapping (what it's typed at), or a list: [(name1, module1), (name2, module2)]
                # that's too cumbersome to type correctly with overloads, so we add an ignore here
                self[m[0]] = m[1]  # type: ignore[assignment]

    # remove forward alltogether to fallback on Module's _forward_unimplemented

def ReLU(inplace=False):
    return jt.nn.ReLU()

def linear(x, w, b=None):
    x = matmul_transpose(x, w)
    if b is not None:
        return x + b
    return x

class CosineSimilarity(Module):
    def __init__(self, dim=1, eps=1e-8):
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        w12 = (x1 * x2).sum(self.dim)
        w1 = (x1 * x1).sum(self.dim)
        w2 = (x2 * x2).sum(self.dim)
        return w12 / (w1 * w2).sqrt().clamp(min_v=self.eps)