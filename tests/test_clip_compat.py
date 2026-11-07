"""Offline checks with a real, small CLIP model and TorchMetrics scoring."""
import torch
from transformers import CLIPConfig, CLIPModel

from eval.clip_compat import adapt_clip_features


def test_clip_score_projected_features():
    from torchmetrics.functional.multimodal.clip_score import _clip_score_update

    config = CLIPConfig(
        text_config=dict(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         max_position_embeddings=8, eos_token_id=2),
        vision_config=dict(hidden_size=24, intermediate_size=32,
                           num_hidden_layers=1, num_attention_heads=2,
                           image_size=8, patch_size=4),
        projection_dim=8,
    )
    model = CLIPModel(config).eval()
    pixels = torch.randn(1, 3, 8, 8)
    ids = torch.tensor([[1, 3, 2]])
    mask = torch.ones_like(ids)

    class Processor:
        def __call__(self, images=None, text=None, **kwargs):
            return {'pixel_values': pixels} if images is not None else {
                'input_ids': ids, 'attention_mask': mask}

    with torch.inference_mode():
        expected = model(pixel_values=pixels, input_ids=ids, attention_mask=mask)
        expected_score = 100 * (expected.image_embeds * expected.text_embeds).sum(-1)
        adapt_clip_features(model)
        assert model.get_image_features(pixels).shape == (1, 8)
        assert model.get_text_features(ids, mask).shape == (1, 8)
        score, count = _clip_score_update(
            torch.zeros(1, 3, 8, 8, dtype=torch.uint8), ['test'], model, Processor())
    assert count == 1
    torch.testing.assert_close(score, expected_score)


def test_legacy_tensor_features():
    class Model:
        def get_image_features(self, value):
            return value

        get_text_features = get_image_features

    model = adapt_clip_features(Model())
    value = torch.randn(1, 8)
    assert model.get_image_features(value) is value
    assert model.get_text_features(value) is value
