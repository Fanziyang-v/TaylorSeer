"""Shared image inference CLI. Model loading is excluded from timing."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import torch
from taylorseer import TaylorSeerConfig, CacheManager, DiTAdapter, FLUXAdapter


def main(kind):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--interval", type=int, default=4 if kind == "dit" else 6)
    parser.add_argument("--warmup", type=int, default=2 if kind == "dit" else 3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=f"{kind}.png")
    parser.add_argument("--prompt", default="A cat sitting on a windowsill")
    parser.add_argument("--class-label", type=int, default=207)
    parser.add_argument("--guidance", type=float, default=4 if kind == "dit" else 3.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    from diffusers import DiTPipeline, FluxPipeline

    pipeline_type = DiTPipeline if kind == "dit" else FluxPipeline
    dtype = torch.float32 if kind == "dit" or args.device == "cpu" else torch.bfloat16
    pipe = pipeline_type.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    pipe.transformer.eval()
    adapter = DiTAdapter() if kind == "dit" else FLUXAdapter()
    manager = CacheManager(TaylorSeerConfig(args.order, args.interval, args.warmup))
    kwargs = (
        dict(class_labels=[args.class_label])
        if kind == "dit"
        else dict(prompt=args.prompt, height=args.height, width=args.width)
    )

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    with (
        torch.inference_mode(),
        nullcontext() if args.baseline else adapter.patch(pipe, manager),
    ):
        sync()
        start = time.perf_counter()
        image = pipe(
            **kwargs,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator(device=args.device).manual_seed(args.seed),
        ).images[0]
        sync()
        elapsed = time.perf_counter() - start
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    output.with_suffix(".json").write_text(
        json.dumps(dict(vars(args), seconds=elapsed), indent=2)
    )
