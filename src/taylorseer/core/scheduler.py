from ..config import TaylorSeerConfig


class ActivationScheduler:
    """Fixed-interval full-vs-predict decision strategy.

    - Warmup steps: Steps in `[0, warmup_steps)` are always full-compute steps.
    - Normal steps: Count each `interval` from the last warmup activation.
    - Step zero always activates, including when `warmup_steps=0`.
    """

    def __init__(self, config: TaylorSeerConfig):
        self.config = config

    def should_activate(self, step: int) -> bool:
        """Decide whether `step` should be a full-compute or activation step.

        Args:
            step (int): Step index in denoising loop (0-based).

        Returns:
            bool: True if `step` should trigger a full compute (call `TaylorSeerCache.update`),
                False if it should be predicted instead (call `TaylorSeerCache.predict`).
        """
        # Count the interval from the last warmup activation; always seed step 0.
        anchor = max(self.config.warmup_steps - 1, 0)
        return step <= anchor or (step - anchor) % self.config.interval == 0
