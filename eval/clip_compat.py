"""Keep TorchMetrics' CLIP feature interface compatible with Transformers 5."""
from functools import wraps

import torch


def adapt_clip_features(model):
    """Adapt only this model instance; preserve projected feature tensors."""
    def adapt(method):
        @wraps(method)
        def features(*args, **kwargs):
            result = method(*args, **kwargs)
            if isinstance(result, torch.Tensor):
                return result
            # Transformers 5 CLIP applies the projection before returning this.
            return result.pooler_output
        return features

    for name in ('get_image_features', 'get_text_features'):
        setattr(model, name, adapt(getattr(model, name)))
    return model
