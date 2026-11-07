from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps

from torch import nn

from ..core.manager import CacheManager


@contextmanager
def patch_forward(
    targets: dict[str, nn.Module],
    manager: CacheManager,
    step_fn: Callable[[], int],
    context_fn: Callable[[], str | None] | None = None,
    *,
    derivative_start_step: int | None = None,
):
    """Patch forwards temporarily. Reset the manager before each generation.

    Each module must appear once; each key/context is called once per step.
    Restoration preserves instance attributes, including on exceptions.
    """
    if len({id(module) for module in targets.values()}) != len(targets):
        raise ValueError("a module cannot have multiple cache keys")
    originals = []
    try:
        for key, module in targets.items():
            original = module.forward
            originals.append((module, "forward" in vars(module), vars(module).get("forward")))

            @wraps(original)
            def wrapped(*args, _key=key, _original=original, **kwargs):
                context = context_fn() if context_fn is not None else None
                return manager.wrap(
                    _key, step_fn(), lambda: _original(*args, **kwargs), context,
                    derivative_start_step=derivative_start_step,
                )

            module.forward = wrapped
        yield
    finally:
        for module, had_override, override in reversed(originals):
            if had_override:
                module.forward = override
            else:
                del module.forward
