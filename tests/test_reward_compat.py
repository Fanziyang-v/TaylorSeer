"""Offline CPU checks for ImageReward's legacy text encoder."""
import importlib.util
import pytest
import torch
from eval.reward_compat import apply_chunking_to_forward, prune_linear_layer, find_pruneable_heads_and_indices


def test_chunking_and_pruning():
    x=torch.randn(2,6,4)
    torch.testing.assert_close(apply_chunking_to_forward(lambda v:v*2,2,1,x),x*2)
    layer=torch.nn.Linear(4,6)
    heads,index=find_pruneable_heads_and_indices([1],3,2,set())
    assert heads=={1} and index.tolist()==[0,1,4,5]
    torch.testing.assert_close(prune_linear_layer(layer,index)(x),layer(x)[...,index])


@pytest.mark.skipif(importlib.util.find_spec('ImageReward') is None,reason='evaluation extra not installed')
def test_legacy_bert_cross_attention_cpu():
    import transformers.modeling_utils as modeling
    original=hasattr(modeling,'apply_chunking_to_forward')
    from eval.reward_compat import import_image_reward
    import_image_reward()
    assert hasattr(modeling,'apply_chunking_to_forward')==original
    from ImageReward.models.BLIP.med import BertConfig, BertModel
    config=BertConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,
                      num_attention_heads=2,intermediate_size=32,add_cross_attention=True)
    config.encoder_width=16
    model=BertModel(config,add_pooling_layer=False).eval()
    with torch.no_grad():
        result=model(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3),
            encoder_hidden_states=torch.randn(1,2,16),encoder_attention_mask=torch.ones(1,2),return_dict=True)
    assert result.last_hidden_state.shape==(1,3,16)
    assert torch.isfinite(result.last_hidden_state).all()
