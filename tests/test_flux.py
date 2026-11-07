from types import SimpleNamespace

import pytest
import torch
from diffusers import FluxTransformer2DModel

from taylorseer.adapters import FLUXAdapter
from taylorseer.config import TaylorSeerConfig
from taylorseer.core import CacheManager


def make_pipeline():
    torch.manual_seed(7)
    model = FluxTransformer2DModel(
        in_channels=4, out_channels=4, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=12,
        pooled_projection_dim=8, axes_dims_rope=(2, 2, 4),
    ).eval()
    return SimpleNamespace(transformer=model, current_timestep=None)


def inputs():
    return dict(hidden_states=torch.randn(1, 3, 4),
                encoder_hidden_states=torch.randn(1, 2, 12),
                pooled_projections=torch.randn(1, 8), timestep=torch.tensor([0.9]),
                img_ids=torch.zeros(3, 3), txt_ids=torch.zeros(2, 3), return_dict=False)


@torch.no_grad()
def test_full_compute_matches_diffusers_and_restores():
    pipe = make_pipeline()
    args = inputs()
    expected = pipe.transformer(**args)[0]
    adapter = FLUXAdapter()
    manager = CacheManager(TaylorSeerConfig(interval=1))
    assert len(adapter.cacheable_modules(pipe)) == 4
    with adapter.patch(pipe, manager):
        pipe.current_timestep = torch.tensor(900)
        actual = pipe.transformer(**args)[0]
        torch.testing.assert_close(actual, expected)
        assert isinstance(manager.caches[('double.0.attn', 'cond')].cache[0], tuple)
    assert not manager.caches
    for block in [*pipe.transformer.transformer_blocks, *pipe.transformer.single_transformer_blocks]:
        assert 'forward' not in vars(block)
    assert not pipe.transformer._forward_pre_hooks
    torch.testing.assert_close(pipe.transformer(**args)[0], expected)


@torch.no_grad()
def test_skips_work_preserves_current_gate_residual_and_cfg():
    pipe = make_pipeline()
    manager = CacheManager(TaylorSeerConfig(order=1, interval=4, warmup_steps=1))
    single = pipe.transformer.single_transformer_blocks[0]
    calls = {'double': 0, 'single': 0}
    handles = []
    for name, module in [('double', pipe.transformer.transformer_blocks[0].attn),
                         ('single', single.attn)]:
        original = module.forward
        def counted(*args, _original=original, _name=name, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)
        module.forward = counted
    observed = {}

    def capture(module, args, kwargs):
        observed.update(kwargs)

    handles.append(single.register_forward_pre_hook(capture, with_kwargs=True))
    try:
        with FLUXAdapter().patch(pipe, manager):
            for step in range(2):
                pipe.current_timestep = torch.tensor(900 - step * 100)
                for branch in ['cond', 'uncond']:
                    args = inputs()
                    pipe.transformer(**args)
                    if step == 1:
                        residual = torch.cat([observed['encoder_hidden_states'], observed['hidden_states']], dim=1)
                        _, gate = single.norm(residual, emb=observed['temb'])
                        feature = manager.caches[('single.0.total', branch)].predict(step)
                        expected = residual + gate.unsqueeze(1) * feature
                        # Invoke the replacement directly to check the pre-gate boundary,
                        # using a separate manager that has the same cached history.
                        reference_manager = CacheManager(manager.config)
                        reference_manager.caches[('single.0.total', branch)] = manager.caches[('single.0.total', branch)]
                        forward = FLUXAdapter._single_forward(single, 'single.0.total', reference_manager,
                                                             {'step': step, 'context': branch})
                        text, image = forward(**observed)
                        torch.testing.assert_close(torch.cat([text, image], dim=1), expected)
            assert calls == {'double': 2, 'single': 2}
            assert len(manager.caches) == 8
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def test_exception_cleanup_and_branch_limit():
    pipe = make_pipeline()
    manager = CacheManager(TaylorSeerConfig())
    with pytest.raises(RuntimeError, match='at most'):
        with FLUXAdapter().patch(pipe, manager):
            pipe.current_timestep = torch.tensor(900)
            for _ in range(3):
                pipe.transformer(**inputs())
    assert not manager.caches
    assert not pipe.transformer._forward_pre_hooks
    assert 'forward' not in vars(pipe.transformer.single_transformer_blocks[0])
