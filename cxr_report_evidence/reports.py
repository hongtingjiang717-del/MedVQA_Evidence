"""Train a report adapter and generate paired Qwen3-VL outputs."""

import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evidence.storage import digest_image, file_digest, write_json
from evidence.vqa.model import LORA_TARGETS, VQAPredictor

from .data import VARIANTS, image_at, load_manifest, read_json, samples


def prompt(study, use_indication):
    instruction = (
        "Write the Findings section of an English frontal chest radiograph research report. "
        "Write one visually assessable finding per line when possible. Preserve uncertainty, "
        "negation and laterality. Do not invent findings or compare with a prior examination. "
        "If the inputs are insufficient to describe the image, say so. Output only findings."
    )
    if use_indication and study.get("indication"):
        instruction += f"\nClinical indication: {study['indication']}"
    return instruction


def resized(image, edge):
    image = image.copy()
    image.thumbnail((edge, edge))
    return image


def findings(text):
    text = re.sub(r"(?im)^\s*(?:#{1,6}\s*)?(?:findings|impression)\s*:\s*", "", text)
    parts = re.split(r"[\r\n;]+|(?<=[.!?])\s+(?=[A-Z])", text)
    return list(dict.fromkeys(part for value in parts
                              if (part := re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s+", "", value).strip())))


class ReportCollator:
    def __init__(self, processor, images, use_indication, edge):
        self.processor, self.images, self.use_indication, self.edge = processor, images, use_indication, edge

    def __call__(self, examples):
        study, = examples
        image = resized(image_at(self.images, study["image"]), self.edge)
        user = [{"role": "user", "content": [{"type": "image", "image": image},
                                            {"type": "text", "text": prompt(study, self.use_indication)}]}]
        reply = {"role": "assistant", "content": [{"type": "text", "text": "\n".join(findings(study["reference"]))}]}
        kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt")
        batch = self.processor.apply_chat_template(user + [reply], add_generation_prompt=False, **kwargs)
        prefix = self.processor.apply_chat_template(user, add_generation_prompt=True, **kwargs)["input_ids"]
        length = prefix.shape[-1]
        if length >= batch["input_ids"].shape[-1] or not batch["input_ids"][:, :length].equal(prefix):
            raise ValueError(f"Assistant token span mismatch in study {study['study_id']}.")
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][:, :length] = -100
        return batch


def train(args):
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError(f"Training stopped before starting: {args.output} is not an empty output directory. "
                         "Choose a fresh --output directory for this training run.")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise ValueError("Report LoRA training requires CUDA.")
    rows = load_manifest(args.manifest)
    train_set = samples(rows, "train", args.max_train)
    validation = samples(rows, "validation", args.max_val)
    set_seed(args.seed)
    bf16 = torch.cuda.is_bf16_supported()
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), dtype=torch.bfloat16 if bf16 else torch.float32,
        attn_implementation="sdpa", local_files_only=True,
    )
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=.05,
                                            target_modules=LORA_TARGETS))
    model.config.use_cache = False
    model.print_trainable_parameters()
    patience = args.early_stopping_patience
    select_best = args.select_best or patience is not None
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)] if patience is not None else None
    hp = TrainingArguments(output_dir=str(args.output), per_device_train_batch_size=1,
                           per_device_eval_batch_size=1, gradient_accumulation_steps=args.accumulate,
                           max_steps=args.steps if args.steps is not None else -1, num_train_epochs=args.epochs,
                           learning_rate=1e-4, bf16=bf16, fp16=not bf16, remove_unused_columns=False,
                           gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                           eval_strategy="epoch" if select_best else "no",
                           load_best_model_at_end=select_best, metric_for_best_model="eval_loss" if select_best else None,
                           prediction_loss_only=True, logging_steps=5, save_strategy="epoch", save_total_limit=2,
                           report_to="none", seed=args.seed)
    trainer = Trainer(model=model, args=hp, train_dataset=train_set, eval_dataset=validation,
                      data_collator=ReportCollator(processor, args.images, args.use_indication, args.image_edge),
                      callbacks=callbacks)
    result = trainer.train()
    epoch_validation = [{"step": entry["step"], "epoch": entry["epoch"],
                         "eval_loss": entry["eval_loss"] if math.isfinite(entry["eval_loss"]) else None}
                        for entry in trainer.state.log_history if "eval_loss" in entry]
    if select_best and (not trainer.state.best_model_checkpoint or not any(
            entry["eval_loss"] is not None for entry in epoch_validation)):
        raise RuntimeError("No checkpoint with finite epoch validation loss was selected.")
    eval_metrics = trainer.evaluate()
    if not all(math.isfinite(value) for value in (result.training_loss, eval_metrics["eval_loss"])):
        raise RuntimeError("Training or validation loss is non-finite.")
    trainer.save_model(str(args.output / "adapter"))
    write_json(args.output / "training_summary.json", {"model": str(args.model), "seed": args.seed,
               "manifest_sha256": file_digest(args.manifest), "image_edge": args.image_edge,
               "use_indication": args.use_indication, "train_studies": len(train_set),
               "validation_studies": len(validation), "train_metrics": result.metrics,
               "validation_metrics": eval_metrics, "global_step": result.global_step,
               "epochs_requested": args.epochs, "max_steps_requested": args.steps,
               "epochs_completed": trainer.state.epoch, "planned_steps": trainer.state.max_steps,
               "early_stopping_patience": patience,
               "stopped_early": patience is not None and result.global_step < trainer.state.max_steps,
               "gradient_accumulation_steps": args.accumulate, "epoch_validation": epoch_validation,
               "best_model_checkpoint": trainer.state.best_model_checkpoint,
               "best_global_step": trainer.state.best_global_step})


def resume(output, config, key):
    rows = {}
    if output.is_file():
        previous = read_json(output)
        old_config = previous["config"]
        if old_config != config:
            changed = sorted(key for key in old_config.keys() | config.keys()
                             if key not in old_config or key not in config or old_config[key] != config[key])
            raise ValueError(f"{output} has a different configuration ({', '.join(changed)}). "
                             "Use the original configuration to resume or choose a new output path.")
        rows = {row[key]: row for row in previous["rows"]}
    return rows


def save(output, config, rows):
    write_json(output, {"config": config, "rows": list(rows.values())})


def generate(args):
    import gc

    rows = samples(load_manifest(args.manifest), args.split, args.limit)
    if any(name.startswith("lora_") for name in args.variants) and args.adapter is None:
        raise ValueError("Supply a report adapter for the LoRA variants.")
    adapter_hash = file_digest(args.adapter / "adapter_model.safetensors") if args.adapter else None
    config = {"manifest_sha256": file_digest(args.manifest), "base_config_sha256": file_digest(args.model / "config.json"),
              "adapter_sha256": adapter_hash, "variants": args.variants, "use_indication": args.use_indication,
              "max_tokens": args.max_tokens, "image_edge": args.image_edge, "split": args.split, "limit": args.limit,
              "device": args.device}
    saved = resume(args.output, config, "study_id")
    for group in ("base", "lora"):
        variants = [name for name in args.variants if name.startswith(group + "_")]
        if not variants or all(all(name in saved.get(study["study_id"], {}).get("reports", {}) for name in variants)
                               for study in rows):
            continue
        model = VQAPredictor(str(args.model), adapter=args.adapter if group == "lora" else None, device=args.device)
        for study in rows:
            ident = study["study_id"]
            image = image_at(args.images, study["image"])
            digest = digest_image(image)
            record = saved.setdefault(ident, {"study_id": ident, "split": study["split"], "image_sha256": digest, "reports": {}})
            if record["image_sha256"] != digest:
                raise ValueError(f"Image changed since the prior generation: {ident}.")
            for name in variants:
                if name in record["reports"]:
                    continue
                visual = resized(image, args.image_edge) if name.endswith("_image") else None
                report, _ = model.complete(prompt(study, args.use_indication), visual, max_new_tokens=args.max_tokens)
                record["reports"][name] = report
                save(args.output, config, saved)
                print(f"{ident} {name}", flush=True)
        del model
        gc.collect()
        if args.device == "cuda":
            import torch

            torch.cuda.empty_cache()


def metrics(args):
    studies = load_manifest(args.manifest)
    reports = read_json(args.reports)
    config = reports["config"]
    if config["manifest_sha256"] != file_digest(args.manifest):
        raise ValueError("The report predictions belong to another manifest.")
    expected = {study["study_id"]: study for study in samples(studies, config["split"], config["limit"])}
    observed = {row["study_id"]: row for row in reports["rows"]}
    variants = getattr(args, "variants", None) or config["variants"]
    if len(set(variants)) != len(variants) or set(variants) - set(config["variants"]):
        raise ValueError("Select unique report variants from the generation configuration.")
    if len(observed) != len(reports["rows"]) or observed.keys() - expected.keys():
        raise ValueError("Unexpected or duplicate study IDs in the generated reports.")
    paired = [row for row in reports["rows"] if set(variants).issubset(row["reports"])]
    if not paired or (len(paired) != len(expected) and not getattr(args, "allow_partial", False)):
        raise ValueError(f"Only {len(paired)}/{len(expected)} planned studies have every selected report variant. "
                         "Finish report generation with the original configuration before comparing metrics.")

    try:
        from rouge_score.rouge_scorer import RougeScorer
    except ModuleNotFoundError as error:
        if error.name != "rouge_score":
            raise
        raise SystemExit(f'ROUGE-L requires rouge-score. Run "{sys.executable}" -m pip install "rouge-score==0.1.2", '
                         "then rerun metrics on the existing reports.json.") from None

    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    grouped = defaultdict(list)
    for row in paired:
        study = expected[row["study_id"]]
        for variant in variants:
            grouped[study["split"], variant].append((study["reference"], row["reports"][variant]))
    if args.radgraph:
        from radgraph import F1RadGraph

        graph_metric = F1RadGraph(reward_level="partial", model_type="radgraph-xl")
    output = defaultdict(dict)
    for (split, variant), rows in grouped.items():
        entry = {"reports": len(rows), "rouge_l_f1": mean(scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in rows)}
        if args.radgraph:
            refs, hyps = zip(*rows)
            score, *_ = graph_metric(refs=list(refs), hyps=list(hyps))
            entry["radgraph_f1_partial"] = float(score)
        output[split][variant] = entry
    write_json(args.output, {"reports_sha256": file_digest(args.reports),
               "coverage": {"planned_studies": len(expected), "paired_studies": len(paired), "variants": variants},
               "by_split": output})


def answer_knowledge(args):
    request = read_json(args.request)
    passages = request["passages"]
    excerpts = "\n".join(f"[{i}] {hit['text']}" for i, hit in enumerate(passages, 1))
    prompt = ("Answer the educational medical-imaging question using the supplied excerpts. "
              "Cite them with [number]. If the excerpts do not answer the question, say so. "
              "Do not interpret a patient's image.\n" + excerpts + "\nQuestion: " + request["question"])
    text, _ = VQAPredictor(str(args.model), device=args.device).complete(prompt, max_new_tokens=192)
    citations = re.findall(r"\[(\d+)\]", text)
    write_json(args.output, {"answer": text if citations and all(1 <= int(i) <= len(passages) for i in citations) else None})
