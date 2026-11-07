"""FLUX.1 adaptation using the official pre-gate cache boundaries."""

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from functools import wraps

import torch
from torch import nn

from ..core import CacheManager
from ..integration.patch import patch_forward
from .base import BaseAdapter


class FLUXAdapter(BaseAdapter):
    """Adapt Diffusers FluxPipeline (separate text/image single-block inputs).

    A patch scope is one generation request. Standard guidance and true CFG
    (cond followed by uncond) are supported. Training, compiled execution and
    custom pipelines with more than two transformer calls per step are outside
    this adapter's contract.
    """

    def cacheable_modules(self, pipeline) -> dict[str, nn.Module]:
        targets = {}
        for i, block in enumerate(pipeline.transformer.transformer_blocks):
            targets[f"double.{i}.attn"] = block.attn
            targets[f"double.{i}.ff"] = block.ff
            targets[f"double.{i}.ff_context"] = block.ff_context
        for i, block in enumerate(pipeline.transformer.single_transformer_blocks):
            # The custom forward below caches this block's pre-gate projection.
            targets[f"single.{i}.total"] = block
        return targets

    def step_fn(self, pipeline) -> Callable[[], int]:
        last = None
        step = -1

        def resolve():
            nonlocal last, step
            timestep = pipeline.current_timestep
            if timestep is None:
                raise RuntimeError(
                    "FLUXAdapter requires an active FluxPipeline denoising loop"
                )
            # FluxPipeline assigns one timestep object per loop iteration. This
            # avoids a GPU-to-CPU scalar sync and is shared by both CFG branches.
            if timestep is not last:
                step += 1
                last = timestep
            return step

        return resolve

    def context_fn(self, pipeline) -> Callable[[], str | None]:
        """Resolve once per transformer invocation, rather than per submodule."""
        last = None
        branch = -1

        def resolve():
            nonlocal last, branch
            timestep = pipeline.current_timestep
            if timestep is not last:
                last, branch = timestep, 0
            else:
                branch += 1
            if branch > 1:
                raise RuntimeError(
                    "expected at most cond/uncond transformer calls per step"
                )
            return ("cond", "uncond")[branch]

        return resolve

    @contextmanager
    def patch(self, pipeline, manager: CacheManager):
        """Install one request's cache and restore all forwards/hooks on exit."""
        transformer = pipeline.transformer
        if transformer.training:
            raise ValueError("FLUXAdapter is inference-only; call transformer.eval()")
        if getattr(transformer, "is_cache_enabled", False):
            raise ValueError(
                "disable the transformer's existing cache before using FLUXAdapter"
            )
        targets = self.cacheable_modules(pipeline)
        resolve_step, resolve_context = self.step_fn(pipeline), self.context_fn(
            pipeline
        )
        state = {"step": -1, "context": None}

        def begin_call(module, args):
            state["step"] = resolve_step()
            state["context"] = resolve_context()

        manager.reset()
        try:
            with ExitStack() as stack:
                handle = transformer.register_forward_pre_hook(begin_call)
                stack.callback(handle.remove)
                doubles = {k: v for k, v in targets.items() if k.startswith("double.")}
                stack.enter_context(
                    patch_forward(
                        doubles,
                        manager,
                        lambda: state["step"],
                        lambda: state["context"],
                    )
                )
                for i, block in enumerate(transformer.single_transformer_blocks):
                    had_override = "forward" in vars(block)
                    override = vars(block).get("forward")

                    def restore(
                        block=block, had_override=had_override, override=override
                    ):
                        if had_override:
                            block.forward = override
                        else:
                            del block.forward

                    stack.callback(restore)
                    block.forward = self._single_forward(
                        block,
                        f"single.{i}.total",
                        manager,
                        state,
                    )
                yield
        finally:
            manager.reset()

    @staticmethod
    def _single_forward(block, key, manager, state):
        @wraps(block.forward)
        def forward(
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):
            text_length = encoder_hidden_states.shape[1]
            residual = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            normalized, gate = block.norm(residual, emb=temb)

            def compute():
                mlp = block.act_mlp(block.proj_mlp(normalized))
                attention = block.attn(
                    hidden_states=normalized,
                    image_rotary_emb=image_rotary_emb,
                    **(joint_attention_kwargs or {}),
                )
                return block.proj_out(torch.cat([attention, mlp], dim=-1))

            feature = manager.wrap(key, state["step"], compute, state["context"])
            output = residual + gate.unsqueeze(1) * feature
            if output.dtype == torch.float16:
                output = output.clip(-65504, 65504)
            return output[:, :text_length], output[:, text_length:]

        return forward
