from milwsi.utils.registry import OPTIMIZER_REGISTRY
from torch.optim import Adam, SGD
from .lookahead import Lookahead
from .radam import RAdam


OPTIMIZER_REGISTRY.register(Adam)
OPTIMIZER_REGISTRY.register(SGD)
OPTIMIZER_REGISTRY.register(Lookahead)
OPTIMIZER_REGISTRY.register(RAdam)


def build_optimizer(name, **kwargs):
    return OPTIMIZER_REGISTRY.get(name)(**kwargs)

