import torch
import jittor as jt
from jittor import optim

class autocast():
    r"""
    See :class:`torch.autocast`.
    ``torch.cuda.amp.autocast(args...)`` is equivalent to ``torch.autocast("cuda", args...)``
    """
    def __init__(self):
        pass

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[override]
        return

    def __call__(self, func):
        return func

class GradScaler(object):
    def __init__(self,
                 init_scale=2.**16,
                 growth_factor=2.0,
                 backoff_factor=0.5,
                 growth_interval=2000,
                 enabled=True):
        # TODO: get jittor amp status?
        # if enabled and amp_definitely_not_available():
        #     warnings.warn("torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.")
        #     self._enabled = False
        # else:
        #     self._enabled = enabled
        self._enabled = enabled
        if self._enabled:
            assert growth_factor > 1.0, "The growth factor must be > 1.0."
            assert backoff_factor < 1.0, "The backoff factor must be < 1.0."

            self._init_scale = init_scale
            # self._scale will be lazily initialized during the first call to scale()
            self._scale = None
            self._growth_factor = growth_factor
            self._backoff_factor = backoff_factor
            self._growth_interval = growth_interval
            self._init_growth_tracker = 0
            # self._growth_tracker will be lazily initialized during the first call to scale()
            self._growth_tracker = None
            # self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def unscale_(self, optimizer):
        return
    
    def step(self, optimizer):
        optimizer.n_step += 1
        optimizer.step()
        return
    
    def update(self):
        return
    
    def scale(self, loss):
        return loss
    
    def state_dict(self):
        return None
