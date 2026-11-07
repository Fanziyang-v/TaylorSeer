import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from taylorseer.config import TaylorSeerConfig
from taylorseer.core import ActivationScheduler, CacheManager, TaylorSeerCache
from taylorseer.integration.patch import patch_forward
from taylorseer.utils import map_feature


@pytest.mark.parametrize('warmup', [0, 1, 3])
@pytest.mark.parametrize('order', [0, 1, 3])
def test_official_parity(warmup, order):
    path = Path(__file__).resolve().parents[1] / 'TaylorSeer-ori/TaylorSeers-Diffusers/taylorseer_flux/taylorseer_utils/__init__.py'
    spec = importlib.util.spec_from_file_location('official_taylor', path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    config = TaylorSeerConfig(order=order, interval=4, warmup_steps=warmup)
    manager = CacheManager(config)
    factors = {0: torch.tensor([0.])}
    state = {'step': 0, 'activated_steps': [0], 'stream': 's', 'layer': 0, 'module': 'm'}
    data = {'max_order': order, 'first_enhance': warmup,
            'cache': {-1: {'s': {0: {'m': factors}}}}}
    counter = 0
    for step in range(18):
        state['step'] = step
        full = step == 0 or step < warmup or counter == 3
        value = torch.tensor([step ** 2 + 2.0])
        if full:
            counter = 0
            state['activated_steps'].append(step)
            if step == 0:
                data['cache'][-1]['s'][0]['m'] = {0: value}
            else:
                official.derivative_approximation(data, state, value)
            expected = value
        else:
            counter += 1
            expected = official.taylor_formula(data, state)
        assert manager.scheduler.should_activate(step) == full
        actual = manager.wrap('m', step, lambda: value)
        torch.testing.assert_close(actual, expected)
        for i, derivative in data['cache'][-1]['s'][0]['m'].items():
            torch.testing.assert_close(manager.caches[('m', None)].cache[i], derivative)


def test_snapshot_and_tuple_prediction():
    cache = TaylorSeerCache(1)
    x = torch.tensor([1.], requires_grad=True)
    cache.update((x, x * 2), 0)
    with torch.no_grad():
        x.add_(100)
    cache.update((torch.tensor([3.]), torch.tensor([6.])), 2)
    result = cache.predict(3)
    torch.testing.assert_close(result[0], torch.tensor([4.]))
    torch.testing.assert_close(result[1], torch.tensor([8.]))
    assert not cache.cache[0][0].requires_grad
    result[0].zero_()
    assert cache.cache[0][0].item() == 3


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_precision(dtype):
    cache = TaylorSeerCache(1)
    cache.update(torch.tensor([1.], dtype=dtype), 0)
    cache.update(torch.tensor([3.], dtype=dtype), 2)
    assert cache.predict(3).dtype == dtype
    assert cache.predict(3).item() == 4


def test_invalid_state_and_reset():
    cache = TaylorSeerCache(2)
    with pytest.raises(RuntimeError):
        cache.predict(0)
    cache.update(torch.ones(1), 0)
    with pytest.raises(ValueError):
        cache.update(torch.ones(1), 0)
    cache.update(torch.ones(1), 1)
    cache.update(torch.ones(1), 2, update_derivatives=False)
    assert list(cache.cache) == [0]
    with pytest.raises(ValueError):
        cache.predict(1)
    assert cache.last_step == 2
    cache.reset()
    assert cache.last_step is None and not cache.cache


def test_manager_cold_start_branches_and_reset():
    manager = CacheManager(TaylorSeerConfig())
    for context, value in [('cond', 1.), ('uncond', 100.)]:
        manager.wrap('late', 3, lambda: torch.tensor([value]), context)
        result = manager.wrap('late', 4, lambda: pytest.fail('must predict'), context)
        assert result.item() == value
    with pytest.raises(ValueError):
        manager.wrap('late', 4, lambda: torch.ones(1), 'cond')
    manager.reset()
    assert not manager.caches
    assert manager.wrap('late', 0, lambda: torch.tensor([5.]), 'cond').item() == 5


def test_patch_restores_on_exception_and_keeps_keys_separate():
    a, b = nn.Identity(), nn.Identity()
    original = lambda x: x + 10
    b.forward = original
    manager = CacheManager(TaylorSeerConfig(warmup_steps=1))
    step = 0
    with pytest.raises(RuntimeError, match='test'):
        with patch_forward({'a': a, 'b': b}, manager, lambda: step):
            assert a(torch.tensor([1.])).item() == 1
            assert b(torch.tensor([1.])).item() == 11
            step = 1
            assert a(torch.tensor([50.])).item() == 1
            assert b(torch.tensor([50.])).item() == 11
            raise RuntimeError('test')
    assert 'forward' not in vars(a)
    assert b.forward is original
    with pytest.raises(ValueError):
        with patch_forward({'a': a, 'alias': a}, manager, lambda: 0):
            pass
