import torch

from ..utils import Feature, map_feature


class TaylorSeerCache:
    """Inference-only derivative history for one module and conditioning branch.

    Steps are increasing denoising-loop indices, not diffusion timesteps.
    Snapshots own their storage to protect history from in-place modifications.
    Derivatives retain the input dtype as in the official implementation;
    prediction follows the official dtype and arithmetic order.
    """

    def __init__(self, order: int):
        self.order = order
        self.cache: dict[int, Feature] = {}
        self.last_step: int | None = None

    def reset(self) -> None:
        """Release history before a new generation request."""
        self.cache.clear()
        self.last_step = None

    @torch.no_grad()
    def update(self, feature: Feature, step: int, *, update_derivatives: bool = True) -> None:
        """Refresh history; disabling derivatives discards all higher orders."""
        if self.last_step is not None and step <= self.last_step:
            raise ValueError("update steps must strictly increase; reset for a new request")
        snapshot = map_feature(lambda x: x.detach().clone(), feature)
        if self.cache:
            map_feature(lambda new, old: new, snapshot, self.cache[0])
        new_cache = {0: snapshot}
        if self.last_step is not None and update_derivatives:
            distance = step - self.last_step
            for i in range(1, self.order + 1):
                if i - 1 not in self.cache:
                    break
                new_cache[i] = map_feature(
                    lambda new, old: (new - old) / distance,
                    new_cache[i - 1], self.cache[i - 1],
                )
        self.cache = new_cache
        self.last_step = step

    @staticmethod
    def _check_dtype(tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_floating_point():
            raise TypeError("cached features must have a floating point dtype")
        return tensor

    @torch.no_grad()
    def predict(self, step: int) -> Feature:
        """Evaluate the Taylor polynomial at or after the last update."""
        if self.last_step is None:
            raise RuntimeError("cannot predict before the first cache update")
        if step < self.last_step:
            raise ValueError("prediction step must not precede the last update")
        distance = step - self.last_step
        orig = self.cache[0]

        import math
        pred = map_feature(lambda x: x * 1.0, orig)
        for i in range(1, len(self.cache)):
            pred = map_feature(
                lambda p, c: p + (1 / math.factorial(i)) * c * (distance ** i),
                pred, self.cache[i],
            )
        return pred

