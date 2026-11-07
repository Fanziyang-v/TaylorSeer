"""One independent FLUX replica per torchrun process."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
from .common import COUNTS, load, save, fingerprint, tasks, ensure_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=COUNTS, default="drawbench")
    p.add_argument(
        "--data",
        default=str(Path(__file__).parent / "drawbench/drawbench.jsonl"),
        help="JSONL with prompt and original metadata",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--method", choices=["baseline", "taylorseer"], required=True)
    p.add_argument("--output", required=True)
    for name, default in [
        ("seed", 42),
        ("steps", 50),
        ("height", 1024),
        ("width", 1024),
        ("order", 1),
        ("interval", 6),
        ("warmup", 3),
    ]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--samples", type=int)
    p.add_argument("--limit", type=int, help="First N prompts for smoke tests")
    args = p.parse_args()
    import torch
    from diffusers import FluxPipeline
    from taylorseer import FLUXAdapter, CacheManager, TaylorSeerConfig

    rank, world = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    device = "cuda:" + os.getenv("LOCAL_RANK", "0")
    torch.cuda.set_device(device)
    prompts = load(args.data)
    if args.limit is not None:
        if args.limit < 1:
            p.error("limit must be positive")
        prompts = prompts[: args.limit]
    count = COUNTS[args.dataset] if args.samples is None else args.samples
    if count < 1:
        p.error("samples must be positive")
    root = Path(args.output)
    config = dict(
        vars(args),
        data_sha256=fingerprint(args.data),
        count=count,
        expected=len(prompts) * count,
        dtype="bfloat16",
    )
    config.pop("output")
    root.mkdir(parents=True, exist_ok=True)
    ensure_config(root / "config.json", config)
    if not list(tasks(prompts, count, args.seed, rank, world)):
        return
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(
        device
    )
    pipe.transformer.eval()
    manager = CacheManager(TaylorSeerConfig(args.order, args.interval, args.warmup))

    def generate(task):
        scope = (
            nullcontext()
            if args.method == "baseline"
            else FLUXAdapter().patch(pipe, manager)
        )
        with torch.inference_mode(), scope:
            return pipe(
                prompt=task["metadata"]["prompt"],
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=torch.Generator(device=device).manual_seed(task["seed"]),
            ).images[0]

    local = list(tasks(prompts, count, args.seed, rank, world))
    if local:
        generate(local[0])
    for task in local:
        index = task["sample_id"]
        image = root / f"{task['prompt_id']:05d}/samples/{task['sample_index']:04d}.png"
        record = root / f"records/{index:08d}.json"
        if image.is_file() and record.is_file():
            continue
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = generate(task)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        image.parent.mkdir(parents=True, exist_ok=True)
        result.save(image)
        save(image.parent.parent / "metadata.jsonl", task["metadata"])
        save(record, dict(task, image=str(image.resolve()), seconds=seconds, rank=rank))


if __name__ == "__main__":
    main()
