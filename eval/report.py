"""Merge complete metric records; average samples, never worker means."""

import argparse
import json
import math
from pathlib import Path
from .common import records, save

FIELDS = (
    "lpips",
    "ssim",
    "psnr",
    "baseline_clipscore",
    "taylorseer_clipscore",
    "baseline_imagereward",
    "taylorseer_imagereward",
)


def summarize(rows):
    result = {}
    for key in FIELDS:
        values = [float(row[key]) for row in rows]
        if any(math.isnan(v) or v == -math.inf for v in values):
            raise ValueError("invalid score")
        if key != "psnr" and any(not math.isfinite(v) for v in values):
            raise ValueError("only PSNR may be infinite")
        result[key] = "inf" if math.inf in values else sum(values) / len(values)
    result["psnr_infinite_samples"] = sum(row["psnr"] == "inf" for row in rows)
    for key in ("clipscore", "imagereward"):
        result[key + "_delta"] = result["taylorseer_" + key] - result["baseline_" + key]
    return result


def build_report(root):
    root = Path(root)
    configs = [json.loads(p.read_text()) for p in root.glob("config.rank*.json")]
    if not configs or any(c != configs[0] for c in configs):
        raise ValueError("missing or incompatible metric configurations")
    rows = records(root)
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(range(configs[0]["expected"])):
        raise ValueError("incomplete or duplicate metric results")
    categories = sorted({r["metadata"]["category"] for r in rows})
    return dict(
        config=configs[0],
        samples=len(rows),
        overall=summarize(rows),
        categories={
            c: dict(
                samples=sum(r["metadata"]["category"] == c for r in rows),
                **summarize([r for r in rows if r["metadata"]["category"] == c]),
            )
            for c in categories
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", required=True)
    p.add_argument("--output", default="report.json")
    a = p.parse_args()
    save(a.output, build_report(a.scores))


if __name__ == "__main__":
    main()
