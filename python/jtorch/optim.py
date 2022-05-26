import jittor as jt
from jittor.optim import *

class Optimizer(jt.optim.Optimizer):
    def pre_step(self, loss, retain_graph=False):
        jt.flags.node_order = 1
        params_has_grad = []
        for pg in self.param_groups:
            pg["grads"] = [ 0 if p.grad is None else p.grad
                for p in pg["params"] ]
            for p in pg["params"]:
                if p.requires_grad:
                    params_has_grad.append(p)
        jt.sync(params_has_grad)

    def zero_grad(self):
        for pg in self.param_groups:
            pg["grads"] = [ None for p in pg["params"] ]
            for p in pg["params"]: p.grad = None

    def post_step(self):
        jt.flags.node_order = 0

for k,v in jt.optim.__dict__.items():
    if isinstance(v, type) and issubclass(v, jt.optim.Optimizer) and \
        not v is jt.optim.Optimizer:
        class OptimWrap(v, Optimizer):
            pass
        globals()[k] = OptimWrap
