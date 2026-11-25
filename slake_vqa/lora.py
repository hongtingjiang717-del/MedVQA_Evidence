"""Train a Qwen3-VL LoRA adapter on official SLAKE train questions."""

import argparse
from pathlib import Path

from .dataset import Slake
from .paths import CHECKPOINTS, DATA, default_model
from evidence.storage import write_json
from evidence.vqa.model import LORA_TARGETS, VQAPredictor


class SupervisedImageAnswers:
    def __init__(self, processor, slake):
        self.processor = processor
        self.slake = slake

    def __call__(self, examples):
        if len(examples) != 1:
            raise ValueError("Use per-device batch size 1 and gradient accumulation.")
        row = examples[0]
        messages = VQAPredictor._messages(row.question, self.slake.image(row.image_id))
        full = messages + [{"role": "assistant", "content": [{"type": "text", "text": row.answer}]}]
        kwargs = {"tokenize": True, "return_dict": True, "return_tensors": "pt"}
        batch = self.processor.apply_chat_template(full, add_generation_prompt=False, **kwargs)
        prefix = self.processor.apply_chat_template(messages, add_generation_prompt=True, **kwargs)["input_ids"]
        length = prefix.shape[-1]
        if length >= batch["input_ids"].shape[-1] or not batch["input_ids"][:, :length].equal(prefix):
            raise ValueError(f"Cannot align the assistant token span for SLAKE {row.qid}.")
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][:, :length] = -100
        return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA)
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--output", type=Path, default=CHECKPOINTS / "slake-lora")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int, default=120)
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1)
    args = parser.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        parser.error("LoRA training requires a CUDA PyTorch environment.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty --output directory for a new run.")
    if args.max_train is not None and args.max_train < 1 or args.max_val < 1:
        parser.error("Training and validation limits must be positive.")
    set_seed(42)
    bf16 = torch.cuda.is_bf16_supported()
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=4 * 32 * 32, max_pixels=256 * 32 * 32)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16 if bf16 else torch.float32, attn_implementation="sdpa",
    )
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16,
                                           target_modules=LORA_TARGETS, lora_dropout=0.05))
    model.config.use_cache = False
    model.print_trainable_parameters()
    hp = TrainingArguments(output_dir=str(args.output), per_device_train_batch_size=1,
                           per_device_eval_batch_size=1, gradient_accumulation_steps=8,
                           max_steps=args.steps, num_train_epochs=args.epochs, learning_rate=1e-4,
                           bf16=bf16, fp16=not bf16, remove_unused_columns=False,
                           gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                           prediction_loss_only=True, logging_steps=10, save_strategy="epoch",
                           save_total_limit=2, report_to="none", seed=42)
    with Slake(args.root) as slake:
        train, val = slake.questions("train"), slake.questions("validation")
        if {row.image_id for row in train} & {row.image_id for row in val}:
            raise ValueError("Training and validation contain the same image IDs.")
        train, val = train[:args.max_train], val[:args.max_val]
        trainer = Trainer(model=model, args=hp, train_dataset=train, eval_dataset=val,
                          data_collator=SupervisedImageAnswers(processor, slake))
        result = trainer.train()
        validation = trainer.evaluate()
        trainer.save_model(str(args.output / "adapter"))
    write_json(args.output / "training_summary.json", {"model": args.model, "train_questions": len(train),
               "validation_questions": len(val), "train_metrics": result.metrics, "validation_metrics": validation})


if __name__ == "__main__":
    main()
