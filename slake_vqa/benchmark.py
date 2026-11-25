"""Evaluate base Qwen and optional LoRA on the same SLAKE questions."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
from statistics import mean

from .dataset import Slake
from .paths import CHECKPOINTS, DATA, FORMAL_ADAPTER, RESULTS, default_model
from evidence.storage import digest_image, file_digest, write_json
from evidence.vqa.model import VQAPredictor, exact_match, prompt_hash, token_f1


def load_wrong_images(path, rows, slake):
    if path is None:
        return {}
    questions = {row.qid: row for row in rows}
    modalities = {row.image_id: row.modality for row in rows}
    mapping = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for entry in csv.DictReader(f):
            qid, other = entry["qid"].strip(), entry["wrong_image_id"].strip()
            row = questions.get(qid)
            if row is None or qid in mapping or not row.modality or modalities.get(other) != row.modality:
                raise ValueError(f"Invalid same-split/same-modality wrong image for {qid}: {other}")
            if entry.get("reviewed", "").strip().casefold() != "yes":
                raise ValueError(f"Wrong image for {qid} needs reviewed=yes after verifying it cannot support the answer.")
            if digest_image(slake.image(row.image_id)) == digest_image(slake.image(other)):
                raise ValueError(f"Wrong image for {qid} has identical pixels.")
            mapping[qid] = other
    if not mapping:
        raise ValueError("Wrong-image manifest is empty.")
    return mapping


def metrics(cases, conditions):
    return {name: {"n": len(cases),
                   "em": mean(exact_match(row["predictions"][name], row["answer"]) for row in cases),
                   "token_f1": mean(token_f1(row["predictions"][name], row["answer"]) for row in cases)}
            for name in conditions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--with-lora", action="store_true", help="Use the local or archived SLAKE adapter.")
    parser.add_argument("--wrong-images", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", type=Path, default=RESULTS / "benchmark")
    args = parser.parse_args()
    if args.with_lora and args.adapter is None:
        trained = CHECKPOINTS / "slake-lora" / "adapter"
        args.adapter = trained if (trained / "adapter_model.safetensors").is_file() else FORMAL_ADAPTER
        if not (args.adapter / "adapter_model.safetensors").is_file():
            parser.error("No SLAKE adapter found. Run python -m slake_vqa train or specify --adapter.")
    if args.limit < 1 or args.max_new_tokens < 1:
        parser.error("--limit and --max-new-tokens must be positive.")
    if args.adapter and not (args.adapter / "adapter_model.safetensors").is_file():
        parser.error("--adapter must point to a saved PEFT adapter.")
    args.output.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    with Slake(args.root) as slake:
        questions = slake.questions(args.split)
        wrong = load_wrong_images(args.wrong_images, questions, slake)
        questions = [row for row in questions if not wrong or row.qid in wrong]
        random.Random(42).shuffle(questions)
        questions = questions[:args.limit]
        conditions = ("image", "text_only", "wrong_image") if wrong else ("image", "text_only")
        variants = {"base": None}
        if args.adapter:
            variants["lora"] = args.adapter
        inputs = {"model": args.model, "split": args.split, "prompt_sha256": prompt_hash(), "seed": 42,
                  "device": args.device, "max_new_tokens": args.max_new_tokens, "qids": [row.qid for row in questions],
                  "data_sha256": {name: file_digest(args.root / name) for name in (f"{args.split}.json", "imgs.zip")},
                  "wrong_images": wrong, "adapter_sha256": file_digest(args.adapter / "adapter_model.safetensors") if args.adapter else None}
        config_path = args.output / "run.json"
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != inputs:
            raise ValueError("Output belongs to a different run. Choose a new --output directory.")
        write_json(config_path, inputs)
        for name, adapter in variants.items():
            case_path = args.output / f"{name}_cases.json"
            cases = json.loads(case_path.read_text(encoding="utf-8")) if case_path.exists() else []
            if [row["qid"] for row in cases] != inputs["qids"][:len(cases)]:
                raise ValueError(f"Unexpected results in {case_path}")
            if len(cases) < len(questions):
                model = VQAPredictor(args.model, adapter=adapter, device=args.device)
                for index, row in enumerate(questions[len(cases):], start=len(cases) + 1):
                    images = {"image": slake.image(row.image_id), "text_only": None}
                    if wrong:
                        images["wrong_image"] = slake.image(wrong[row.qid])
                    preds = {mode: model.answer(row.question, image, max_new_tokens=args.max_new_tokens) for mode, image in images.items()}
                    source = {"wrong_image_id": wrong.get(row.qid), "image_sha256": digest_image(images["image"])}
                    cases.append({**asdict(row), **source, "predictions": preds})
                    write_json(case_path, cases)
                    print(f"{name} {index}/{len(questions)} {row.qid}", flush=True)
                del model
            all_metrics[name] = metrics(cases, conditions)
    if "lora" in all_metrics:
        all_metrics["lora_minus_base_em_pp"] = {key: 100 * (all_metrics["lora"][key]["em"] - all_metrics["base"][key]["em"])
                                                for key in conditions}
    write_json(args.output / "summary.json", {"dataset": "BoKelvin/SLAKE", "split": args.split,
               "model": args.model, "device": args.device, "max_new_tokens": args.max_new_tokens,
               "question_count": len(questions), "unique_images": len({row.image_id for row in questions}),
               "metrics": all_metrics})


if __name__ == "__main__":
    main()
