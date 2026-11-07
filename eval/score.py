"""Paired DrawBench metrics, one independent worker per device."""

import argparse
import json
import os
from pathlib import Path
from .common import paired_runs, save, fingerprint, ensure_config

CLIP_MODEL = "openai/clip-vit-base-patch16"
REWARD_MODEL = "ImageReward-v1.0"


class Metrics:
    def __init__(self, device, clip_model=CLIP_MODEL, reward_model=REWARD_MODEL):
        from .reward_compat import import_image_reward
        RM = import_image_reward()
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
        from torchmetrics.multimodal import CLIPScore
        from .clip_compat import adapt_clip_features

        self.device = device
        self.clip = CLIPScore(model_name_or_path=clip_model).to(device).eval()
        adapt_clip_features(self.clip.model)
        self.reward = RM.load(reward_model, device=device).eval()
        self.lpips = (
            LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
            .to(device)
            .eval()
        )

    def compute(self, prompt, baseline, candidate):
        import torch
        from torchvision.transforms.functional import pil_to_tensor
        from torchmetrics.functional.image import (
            peak_signal_noise_ratio,
            structural_similarity_index_measure,
        )

        if baseline.size != candidate.size:
            raise ValueError("paired image sizes differ")
        with torch.inference_mode():
            integers = [
                pil_to_tensor(image).unsqueeze(0).to(self.device)
                for image in (baseline, candidate)
            ]
            a, b = [image.float() / 255 for image in integers]
            row = dict(
                lpips=float(self.lpips(b, a)),
                ssim=float(structural_similarity_index_measure(b, a, data_range=1.0)),
                psnr=float(peak_signal_noise_ratio(b, a, data_range=1.0)),
            )
            self.lpips.reset()
            for label, image, tensor in zip(
                ("baseline", "taylorseer"), (baseline, candidate), integers
            ):
                row[label + "_clipscore"] = float(self.clip(tensor, [prompt]))
                self.clip.reset()
                row[label + "_imagereward"] = float(self.reward.score(prompt, image))
        return {
            key: "inf" if value == float("inf") else value for key, value in row.items()
        }


def score_pairs(pairs, output, metrics, rank=0, world=1):
    from PIL import Image

    for a, b in pairs[rank::world]:
        path = Path(output) / "records" / f"{a['sample_id']:08d}.json"
        identity = dict(
            sample_id=a["sample_id"],
            seed=a["seed"],
            metadata=a["metadata"],
            baseline_sha256=fingerprint(a["image"]),
            candidate_sha256=fingerprint(b["image"]),
        )
        if path.exists():
            existing = json.loads(path.read_text())
            if any(existing.get(key) != value for key, value in identity.items()):
                raise ValueError("score input changed; use a new output directory")
            continue
        with Image.open(a["image"]) as ia, Image.open(b["image"]) as ib:
            scores = metrics.compute(
                a["metadata"]["prompt"], ia.convert("RGB"), ib.convert("RGB")
            )
        save(path, dict(identity, **scores))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", help="Defaults to local CUDA device")
    p.add_argument("--clip-model", default=CLIP_MODEL)
    p.add_argument("--reward-model", default=REWARD_MODEL)
    a = p.parse_args()
    bc, tc, pairs = paired_runs(a.baseline, a.candidate)
    rank, world = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    device = a.device or "cuda:" + os.getenv("LOCAL_RANK", "0")
    config = dict(
        baseline=bc,
        candidate=tc,
        clip_model=a.clip_model,
        reward_model=a.reward_model,
        lpips_net="alex",
        expected=len(pairs),
    )
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    ensure_config(root / "config.json", config)
    # Each rank has its own config; report verifies all workers agree.
    config_path = root / f"config.rank{rank}.json"
    for previous in root.glob("config.rank*.json"):
        if json.loads(previous.read_text()) != config:
            raise ValueError("metric configuration changed")
    save(config_path, config)
    if not pairs[rank::world]:
        return
    score_pairs(pairs, root, Metrics(device, a.clip_model, a.reward_model), rank, world)


if __name__ == "__main__":
    main()
