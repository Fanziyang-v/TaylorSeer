from unittest.mock import patch
import torch
from diffusers import DiTTransformer2DModel
from types import SimpleNamespace
from taylorseer import DiTAdapter, CacheManager, TaylorSeerConfig


def test_dit_full_and_forecast_restore():
    torch.manual_seed(0)
    model = DiTTransformer2DModel(num_attention_heads=2, attention_head_dim=8,
        in_channels=4, out_channels=8, num_layers=1, sample_size=8,
        patch_size=2, num_embeds_ada_norm=10).eval()
    pipe = SimpleNamespace(transformer=model)
    x = torch.randn(2, 4, 8, 8)
    kwargs = dict(timestep=torch.tensor([4, 4]), class_labels=torch.tensor([1, 2]))
    with torch.no_grad():
        expected = model(x, **kwargs).sample
        manager = CacheManager(TaylorSeerConfig(interval=1))
        with DiTAdapter().patch(pipe, manager):
            torch.testing.assert_close(model(x, **kwargs).sample, expected)
        attention = model.transformer_blocks[0].attn1
        original = attention.forward
        calls = []
        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        attention.forward = counted
        manager = CacheManager(TaylorSeerConfig(interval=4, warmup_steps=1))
        with DiTAdapter().patch(pipe, manager):
            for _ in range(3):
                assert torch.isfinite(model(x, **kwargs).sample).all()
        attention.forward = original
        assert len(calls) == 1
        assert not manager.caches
        torch.testing.assert_close(model(x, **kwargs).sample, expected)
