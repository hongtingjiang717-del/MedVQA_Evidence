"""医学视觉问答项目核心模块：真实开源模型推理，没有伪造的检测或分割功能。"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional

DATASET_ID = "mdwiratathya/SLAKE-vqa-english"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


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


class VQAPredictor:
    def __init__(self, model_id: str = MODEL_ID):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("缺少 PyTorch / Transformers。请先安装requirements。") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到CUDA GPU。真实模型模式建议AutoDL RTX4090。")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()

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

    def answer(self, question: str, image=None, max_new_tokens: int = 48) -> str:
        """模型推理；image=None时是严格的文字消融实验。"""
        if not str(question).strip():
            return ""
        messages = self._messages(question, image)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[-1]:]
        result = self.processor.decode(trimmed, skip_special_tokens=True).strip()
        result = re.sub(r"^ANSWER\s*:\s*", "", result, flags=re.I)
        return result.splitlines()[0].strip()
