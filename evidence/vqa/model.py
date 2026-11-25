"""Qwen3-VL inference and short-answer evaluation metrics."""
from __future__ import annotations

import hashlib
import re
import string
from collections import Counter

DATASET_ID = "mdwiratathya/SLAKE-vqa-english"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
LORA_TARGETS = r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"


def normalize_answer(text: str) -> str:
    """英文答案统一格式，用于透明可复现的自动评估。"""
    text = str(text).lower().strip()
    text = re.sub(r"^answer\s*:\s*", "", text)
    text = text.splitlines()[0].strip() if text else ""
    text = "".join(" " if c in string.punctuation else c for c in text)
    return " ".join(text.split())


def exact_match(pred: str, target: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(target))


def token_f1(pred: str, target: str) -> float:
    """对简短英文答案的词级重叠评估，不等于临床语义正确率。"""
    p, t = normalize_answer(pred).split(), normalize_answer(target).split()
    if not p or not t:
        return float(p == t)
    common = sum((Counter(p) & Counter(t)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(t)
    return 2 * precision * recall / (precision + recall)


def prompt_for(question: str) -> str:
    return (
        "You are answering a research benchmark of medical image questions. "
        "Reply with only the short answer in English, without an explanation. "
        "Do not assume a diagnosis not supported by the input. "
        f"Question: {question}\nANSWER:"
    )


def load_data():
    """第一次调用联网下载约60MB压缩的英文SLAKE QA数据；下载失败会明确报错。"""
    from datasets import load_dataset
    return load_dataset(DATASET_ID)


def prompt_hash():
    return hashlib.sha256(prompt_for("<QUESTION>").encode("utf-8")).hexdigest()


class VQAPredictor:
    def __init__(self, model_id: str = MODEL_ID, *, adapter=None, device="cuda"):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("缺少 PyTorch / Transformers。请先安装requirements。") from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("未检测到CUDA GPU。真实模型模式建议AutoDL RTX4090。")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.float16 if device == "cuda" else torch.bfloat16,
            device_map="auto" if device == "cuda" else {"": "cpu"},
            attn_implementation="sdpa",
        ).eval()
        if adapter is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter)).eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @staticmethod
    def _messages(question: str, image=None):
        content = []
        if image is not None:
            content.append({"type": "image", "image": image.convert("RGB")})
        content.append({"type": "text", "text": prompt_for(question)})
        return [{"role": "user", "content": content}]

    def complete(self, prompt: str, image=None, max_new_tokens: int = 48, *, confidence=False):
        """Return generated text and optional geometric mean token probability."""
        content = []
        if image is not None:
            content.append({"type": "image", "image": image.convert("RGB")})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      return_dict_in_generate=confidence, output_scores=confidence)
            sequence = out.sequences if confidence else out
            tokens = sequence[0, inputs["input_ids"].shape[-1]:]
            score = None
            if confidence and len(tokens):
                eos = self.model.generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos])
                scores = [logits[0].float().log_softmax(0)[token].item()
                          for logits, token in zip(out.scores, tokens) if token.item() not in eos_ids]
                if scores:
                    import math

                    score = math.exp(sum(scores) / len(scores))
        return self.processor.decode(tokens, skip_special_tokens=True).strip(), score

    def predict(self, question, image=None, max_new_tokens=48, *, confidence=False):
        if not str(question).strip():
            return "", None
        result, score = self.complete(prompt_for(question), image, max_new_tokens, confidence=confidence)
        result = re.sub(r"^ANSWER\s*:\s*", "", result, flags=re.I)
        return (result.splitlines()[0].strip() if result else ""), score

    def answer(self, question: str, image=None, max_new_tokens: int = 48) -> str:
        """Return a short benchmark answer, with no image for text-only variants."""
        return self.predict(question, image, max_new_tokens)[0]
