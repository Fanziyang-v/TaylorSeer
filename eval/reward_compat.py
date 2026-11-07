"""Bridge ImageReward 1.5's vendored BERT to Transformers 5.

Only the legacy ImageReward model uses these helpers; FLUX keeps the current
Transformers implementation. No weights or scoring equations are changed.
"""
import torch
from torch import nn


def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *inputs):
    if chunk_size <= 0:
        return forward_fn(*inputs)
    length = inputs[0].shape[chunk_dim]
    if any(x.shape[chunk_dim] != length for x in inputs) or length % chunk_size:
        raise ValueError('incompatible chunk dimensions')
    chunks = [x.chunk(length // chunk_size, dim=chunk_dim) for x in inputs]
    return torch.cat([forward_fn(*parts) for parts in zip(*chunks)], dim=chunk_dim)


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    heads = set(heads) - already_pruned_heads
    mask = torch.ones(n_heads, head_size)
    for head in heads:
        mask[head - sum(previous < head for previous in already_pruned_heads)] = 0
    return heads, torch.arange(mask.numel())[mask.flatten().bool()].long()


def prune_linear_layer(layer, index, dim=0):
    index = index.to(layer.weight.device)
    weight = layer.weight.index_select(dim, index).detach().clone()
    shape = list(layer.weight.shape)
    shape[dim] = len(index)
    result = nn.Linear(shape[1], shape[0], bias=layer.bias is not None).to(layer.weight)
    with torch.no_grad():
        result.weight.copy_(weight)
        if layer.bias is not None:
            result.bias.copy_(layer.bias if dim == 1 else layer.bias.index_select(0, index))
    return result


def invert_attention_mask(self, mask):
    if mask.dim() == 2:
        mask = mask[:, None, None, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    return (1.0 - mask.to(self.dtype)) * torch.finfo(self.dtype).min


def get_head_mask(self, mask, layers, is_attention_chunked=False):
    if mask is None:
        return [None] * layers
    if mask.dim() == 1:
        mask = mask[None, None, :, None, None].expand(layers, -1, -1, -1, -1)
    elif mask.dim() == 2:
        mask = mask[:, None, :, None, None]
    mask = mask.to(self.dtype)
    return mask.unsqueeze(-1) if is_attention_chunked else mask


def init_reward_tokenizer():
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids('[ENC]')
    return tokenizer


def import_image_reward():
    import transformers.modeling_utils as modeling
    # ImageReward imports these removed names directly at module import time.
    names = ('apply_chunking_to_forward', 'find_pruneable_heads_and_indices', 'prune_linear_layer')
    added = [name for name in names if not hasattr(modeling, name)]
    try:
        for name in added:
            setattr(modeling, name, globals()[name])
        import ImageReward as reward
    finally:
        for name in added:
            delattr(modeling, name)
    from ImageReward.models.BLIP.med import BertPreTrainedModel
    for name in ('invert_attention_mask', 'get_head_mask'):
        if not hasattr(BertPreTrainedModel, name):
            setattr(BertPreTrainedModel, name, globals()[name])
    from ImageReward.models.BLIP.med import BertModel
    # The reward encoder has no LM head or tied output weights. Keep the
    # original BERT initialization rather than Transformers 5 post_init logic.
    BertModel.init_weights = lambda self: self.apply(self._init_weights)
    from ImageReward.models.BLIP import blip, blip_pretrain
    # blip_pretrain imports the function by value, so update both references.
    blip.init_tokenizer = init_reward_tokenizer
    blip_pretrain.init_tokenizer = init_reward_tokenizer
    return reward
