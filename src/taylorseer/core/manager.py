from collections import defaultdict
from collections.abc import Callable

from ..config import TaylorSeerConfig
from ..utils import Feature
from .cache import TaylorSeerCache
from .scheduler import ActivationScheduler


class CacheManager:
    """TaylorSeer cache manager, which combines:

    - `TaylorSeerCache`: Store approximated derivatives at the latest activation step
        via finite differences for prediction using a Taylor expansion.
    - `ActivationScheduler`: Decide which step should be activated (full-compute).

    And only comprise one function `wrap(key, step, compute_fn, context)`,
    which provides a computed (activation) or estimated (predict) feature tensor.
    """

    def __init__(self, config: TaylorSeerConfig):
        self.config = config
        self.scheduler = ActivationScheduler(config)
        self.caches = defaultdict[tuple[str, str | None], TaylorSeerCache](
            lambda: TaylorSeerCache(config.order)
        )

        self._last_calls: dict[tuple[str, str | None], int] = {}

    def reset(self) -> None:
        """Start a new request; do not share a manager across concurrent requests."""
        for cache in self.caches.values():
            cache.reset()
        self.caches.clear()
        self._last_calls.clear()

    def wrap(
        self,
        key: str,
        step: int,
        compute_fn: Callable[[], Feature],
        context: str | None = None,
        *,
        derivative_start_step: int | None = None,
    ) -> Feature:
        """Run `compute_fn` or forecast with cached approximated derivates using Taylor expansion.

        Args:
            key (str): An unique identifier for each cacheable unit (e.g., `(layer_index, module_name)`)
            step (int): Current denoising step.
            compute_fn (Callable[[], Feature]): Zero-argument callable performing the real forward
                computation. Only invoked on activation or full-compute steps.
            context (str | None): Identifies which independent conditioning branch (e.g.
                `"cond"`/`"uncond"` under classifier-free guidance) `key` is being called for.
                Each `(key, context)` pair keeps its own `TaylorSeerCache`, so branches that run
                at the same `step` don't get mixed into one Taylor series.
            derivative_start_step: First step allowed to build derivatives. Earlier
                activations refresh only order zero. Defaults to warmup_steps - 1
                (bounded by zero), matching the official first_enhance gate.

        Returns:
            Feature: The computed or predicted feature.
        """
        if derivative_start_step is None:
            derivative_start_step = max(self.config.warmup_steps - 1, 0)
        cache = self.caches[(key, context)]
        previous = self._last_calls.get((key, context))
        if previous is not None and step <= previous:
            raise ValueError(
                "each key/context must be called once per increasing step; reset for a new request"
            )
        if cache.last_step is None or self.scheduler.should_activate(step):
            feature = compute_fn()
            cache.update(
                feature, step, update_derivatives=step >= derivative_start_step
            )
        else:
            feature = cache.predict(step)
        self._last_calls[(key, context)] = step
        return feature
