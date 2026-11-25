"""Command line for the CXR report evidence pipeline."""

import argparse
from collections import Counter
from pathlib import Path

from . import audit, data, openi, paths, reports


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def count(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer.")
    return number


def fraction(value):
    number = float(value)
    if not 0 < number <= 0.2:
        raise argparse.ArgumentTypeError("Expected a radius fraction in (0, 0.2].")
    return number


def check(args):
    if not args.manifest.is_file():
        raise SystemExit(f"Manifest is missing: {args.manifest}. Run prepare-openi or prepare successfully first.")
    studies = data.load_manifest(args.manifest)
    missing = [row["image"] for row in studies if not (args.images / row["image"]).is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} image files; first missing: {missing[0]}.")
    for split in data.SPLITS:
        row = next((row for row in studies if row["split"] == split), None)
        if row:
            image = data.image_at(args.images, row["image"])
            print(split, row["view"], image.size)
    print(dict(Counter(row["split"] for row in studies)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, handler, summary, files=(), **defaults):
        sub = commands.add_parser(name, help=summary, description=summary)
        for field in files:
            sub.add_argument("--" + field, type=Path, required=field not in defaults, default=defaults.get(field))
        sub.set_defaults(handler=handler)
        return sub

    run = paths.results_run()
    inputs = {"manifest": paths.MANIFEST, "images": paths.IMAGES}

    sub = command("prepare", data.prepare, "Build a patient-disjoint manifest and exact download URLs.",
                  ("metadata", "splits", "reports", "output", "urls"))
    sub.add_argument("--train", type=count, default=32)
    sub.add_argument("--validation", type=count, default=8)
    sub.add_argument("--test", type=count, default=8)
    sub.add_argument("--seed", type=int, default=42)
    sub = command("prepare-openi", openi.prepare_openi, "Pair Open-i XML reports with labeled frontal PNG images.",
                  ("reports", "images", "frontal-list", "output"),
                  reports=paths.DATA / "NLMCXR_reports.tgz", images=paths.IMAGES,
                  **{"frontal-list": paths.DATA / "view_frontal.txt", "output": None})
    sub.add_argument("--train", type=count)
    sub.add_argument("--validation", type=count)
    sub.add_argument("--test", type=count)
    sub.add_argument("--seed", type=int, default=42)
    command("check", check, "Check selected image paths and open an image from each split.", ("manifest", "images"), **inputs)

    sub = command("train", reports.train, "Train a new chest X-ray report LoRA adapter.",
                  ("manifest", "images", "model", "output"), **inputs, model=paths.QWEN, output=paths.LOCAL_TRAINING)
    sub.add_argument("--max-train", type=positive)
    sub.add_argument("--max-val", type=positive)
    sub.add_argument("--steps", type=positive)
    sub.add_argument("--epochs", type=positive, default=1)
    sub.add_argument("--select-best", action="store_true", help="Evaluate each epoch and export the lowest-validation-loss adapter.")
    sub.add_argument("--early-stopping-patience", type=positive,
                     help="Stop after this many epochs without lower validation loss. Enables best-adapter selection.")
    sub.add_argument("--accumulate", type=positive, default=8)
    sub.add_argument("--seed", type=int, default=42)
    sub.add_argument("--image-edge", type=positive, default=768)
    sub.add_argument("--use-indication", action="store_true")

    sub = command("generate", reports.generate, "Generate paired image and text report comparisons.",
                  ("manifest", "images", "model", "output"), **inputs, model=paths.QWEN,
                  output=paths.LOCAL_RUN / "reports.json")
    sub.add_argument("--adapter", type=Path)
    sub.add_argument("--variants", nargs="+", choices=data.VARIANTS, default=list(data.VARIANTS))
    sub.add_argument("--split", choices=(*data.SPLITS, "audit", "all"), default="test")
    sub.add_argument("--limit", type=positive)
    sub.add_argument("--max-tokens", type=positive, default=256)
    sub.add_argument("--image-edge", type=positive, default=768)
    sub.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    sub.add_argument("--use-indication", action="store_true")

    sub = command("metrics", reports.metrics, "Compare report predictions with reference Findings.",
                  ("manifest", "reports", "output"), manifest=paths.MANIFEST, reports=run / "reports.json", output=None)
    sub.add_argument("--variants", nargs="+", choices=data.VARIANTS, help="Score a selected subset of the configured variants.")
    sub.add_argument("--allow-partial", action="store_true", help="Score only cases containing every selected variant.")
    sub.add_argument("--radgraph", action="store_true")

    sub = command("ground", audit.ground, "Localize Qwen findings with MAIRA-2 in its own environment.",
                  ("manifest", "images", "reports", "model", "output"), **inputs, reports=run / "reports.json",
                  model=paths.MAIRA, output=None)
    sub.add_argument("--max-findings", type=positive, default=8)
    sub.add_argument("--device", choices=("cuda", "cpu"), default="cuda")

    sub = command("probe", audit.probe, "Run a shared visual verifier on original, target and control images.",
                  ("manifest", "images", "reports", "grounding", "model", "output"), **inputs,
                  reports=run / "reports.json", model=paths.QWEN, grounding=None, output=None)
    sub.add_argument("--image-edge", type=positive, default=768)
    sub.add_argument("--radius-fraction", type=fraction, default=0.025)
    sub.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    sub.add_argument("--alignment", action="store_true")
    command("summarize-probes", audit.summarize_probes, "Summarize coverage and target-control differences without labels.",
            ("probes", "output"), probes=run / "probes.json", output=None)
    command("review", audit.write_review, "Export a blind CSV template for manual visual support labels.", ("probes", "output"))
    command("fit", audit.fit, "Fit on validation, evaluate on test and export scores.",
            ("probes", "review", "estimator", "predictions", "output"))
    sub = command("knowledge", reports.answer_knowledge, "Generate an answer from supplied local passages.",
                  ("request", "model", "output"))
    sub.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.command == "prepare-openi":
        using_local_manifest = args.output is None
        if using_local_manifest:
            args.output = paths.MANIFEST
            if args.output.exists():
                parser.error("The local manifest already exists. Run check, or specify a different --output.")
        sizes = (1600, 100, 100) if using_local_manifest else (16, 4, 4)
        for name, size in zip(data.SPLITS, sizes, strict=True):
            if getattr(args, name) is None:
                setattr(args, name, size)
    if args.command == "generate":
        if len(set(args.variants)) != len(args.variants):
            parser.error("Variants must be unique.")
        if args.adapter is None and any(variant.startswith("lora_") for variant in args.variants):
            trained = paths.LOCAL_TRAINING / "adapter"
            args.adapter = trained if (trained / "adapter_model.safetensors").is_file() else paths.FORMAL_ADAPTER
            if not (args.adapter / "adapter_model.safetensors").is_file():
                parser.error("No local report adapter found. Run train or specify --adapter.")
    if args.command == "metrics" and args.output is None:
        args.output = args.reports.with_name("report-metrics-local.json")
    if args.command == "ground" and args.output is None:
        args.output = args.reports.with_name("ground.json")
    if args.command == "probe":
        args.grounding = args.grounding or args.reports.with_name("ground.json")
        args.output = args.output or args.reports.with_name("probes.json")
    if args.command == "summarize-probes" and args.output is None:
        args.output = args.probes.with_name("probe-summary.json")
    args.handler(args)


if __name__ == "__main__":
    main()
