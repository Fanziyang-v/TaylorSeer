"""Diffusers DiT: cache attention and MLP outputs before adaptive gates."""
from contextlib import contextmanager
from .base import BaseAdapter
from ..integration.patch import patch_forward


class DiTAdapter(BaseAdapter):
    def cacheable_modules(self, pipeline):
        return {f"block.{i}.{name}": getattr(block, name)
                for i, block in enumerate(pipeline.transformer.transformer_blocks)
                for name in ("attn1", "ff")}

    def step_fn(self, pipeline):
        state = {"step": -1}
        return lambda: state["step"]

    @contextmanager
    def patch(self, pipeline, manager):
        state = {"step": -1}

        def begin(module, args):
            state["step"] += 1

        manager.reset()
        handle = pipeline.transformer.register_forward_pre_hook(begin)
        try:
            with patch_forward(self.cacheable_modules(pipeline), manager,
                               lambda: state["step"]):
                yield
        finally:
            handle.remove()
            manager.reset()
