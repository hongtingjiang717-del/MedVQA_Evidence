"""Exercise the local app and counterfactual image path without model weights."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageChops, ImageDraw

from evidence.perturbation import matched_mask, perturb

from .audit import probe
from .paths import RESULTS
from evidence.storage import write_json


def verify_regions():
    class ContrastReader:
        def predict(self, question, image, max_new_tokens=48, confidence=False):
            patch = image.convert("L").crop((18, 18, 36, 36))
            lo, hi = patch.getextrema()
            return ("visible" if hi - lo > 100 else "missing"), 0.8

        def answer(self, question, image, max_new_tokens=48):
            return self.predict(question, image, max_new_tokens=max_new_tokens)[0]

    mask = Image.new("L", (96, 96))
    ImageDraw.Draw(mask).ellipse((12, 12, 42, 42), fill=255)
    image = Image.new("RGB", mask.size, "gray")
    draw = ImageDraw.Draw(image)
    for x in range(15, 40, 2):
        draw.line((x, 18, x, 38), fill="white")
    control = matched_mask(mask)
    target_area = mask.histogram()[255]
    if control is None or control.histogram()[255] != target_area or ImageChops.multiply(mask, control).getbbox():
        raise AssertionError("Control should match shape/area and avoid the target.")
    diff = ImageChops.difference(image, perturb(image, mask, 8)).convert("L")
    if ImageChops.multiply(diff, ImageChops.invert(mask)).getbbox():
        raise AssertionError("Mask perturbation changed pixels outside the target.")
    result = probe(ContrastReader(), image, "Is the pattern visible?", mask, radius=8)
    if result["target_specific_change"] != 1:
        raise AssertionError("Synthetic reader should only respond to the target perturbation.")
    return {"synthetic_target_pixels": target_area, "control_pixels": control.histogram()[255],
            "target_specific_change": result["target_specific_change"], "variants": result["variants"]}


def verify_service(root):
    from fastapi.testclient import TestClient
    from .web import create_app

    def client():
        return TestClient(create_app(db=root / "qdrant", state_db=root / "state.sqlite3"))

    with client() as api:
        assert api.get("/").status_code == 200
        assert api.get("/health").json()["status"] == "ready"
        headers = {"X-Profile-Token": api.post("/profiles").json()["profile_token"]}
        second = {"X-Profile-Token": api.post("/profiles").json()["profile_token"]}
        assert api.put("/preferences", headers=headers, json={"style": "detailed"}).status_code == 200
        question = {"question": "How does a CT scanner reconstruct a slice?"}
        assert api.post("/ask", json=question).status_code == 401
        response = api.post("/ask", headers=headers, json=question)
        response.raise_for_status()
        knowledge = response.json()
        assert knowledge["intent"] == "knowledge" and "ct-principle" in {hit["id"] for hit in knowledge["passages"]}
        other = api.post("/ask", headers=second, json={**question, "thread_id": knowledge["thread_id"]}).json()
        assert other["style"] == "concise"
        mixed = {"question": "How does CT work?", "intent": "mixed", "image_id": "xmlab0"}
        response = api.post("/ask", headers=headers, json=mixed)
        response.raise_for_status()
        mixed_result = response.json()
        assert mixed_result["intent"] == "mixed" and mixed_result["status"] and mixed_result["passages"]
        visual = api.post("/ask", headers=headers, json={**mixed, "intent": "visual"}).json()
        assert visual["knowledge_answer"] is None and visual["status"]
    with client() as api:
        recovered = api.post("/ask", headers=headers, json=question).json()
        assert recovered["style"] == "detailed" and recovered["thread_id"] != knowledge["thread_id"]
    return {"health": "ready", "unauthenticated_request_rejected": True, "html": 200,
            "knowledge_citation_ids": [hit["id"] for hit in knowledge["passages"]], "mixed_route_returned_sources": True,
            "visual_disabled_message": visual["status"][0], "separate_profile_preference": other["style"],
            "preference_after_restart": recovered["style"]}


def verify_lora(processor_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from .dataset import Slake
    from evidence.vqa.model import LORA_TARGETS
    from .lora import SupervisedImageAnswers

    torch.set_num_threads(2)
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True,
                                              min_pixels=4 * 32 * 32, max_pixels=24 * 32 * 32)
    config = Qwen3VLConfig(text_config={"hidden_size": 64, "head_dim": 32, "num_attention_heads": 2,
                                       "num_key_value_heads": 2, "num_hidden_layers": 1, "intermediate_size": 128,
                                       "vocab_size": 151936, "use_cache": False,
                                       "rope_scaling": {"rope_type": "default", "mrope_section": [4, 6, 6]}},
                           vision_config={"depth": 1, "hidden_size": 32, "num_heads": 2, "intermediate_size": 64,
                                          "out_hidden_size": 64, "deepstack_visual_indexes": []})
    model = get_peft_model(Qwen3VLForConditionalGeneration(config),
                           LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=LORA_TARGETS)).train()
    with Slake() as slake:
        row = slake.questions("train")[0]
        batch = SupervisedImageAnswers(processor, slake)([row])
    result = model(**batch)
    result.loss.backward()
    trainable = [(name, weight) for name, weight in model.named_parameters() if weight.requires_grad]
    assert trainable and all("lora_" in name for name, _ in trainable)
    assert any(weight.grad is not None and weight.grad.abs().sum() > 0 for _, weight in trainable)
    optimizer = torch.optim.AdamW([weight for _, weight in trainable])
    optimizer.step()
    return {"weights": "tiny random Qwen3-VL architecture", "dataset_train_qid": row.qid,
            "assistant_tokens": int((batch["labels"] != -100).sum().item()),
            "trainable_parameters": sum(weight.numel() for _, weight in trainable),
            "forward_loss_finite": bool(torch.isfinite(result.loss)), "optimizer_steps": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "smoke_summary.json")
    parser.add_argument("--processor", type=Path, help="Optional local Qwen3-VL tokenizer and processor for a tiny CPU LoRA smoke test.")
    args = parser.parse_args()
    with TemporaryDirectory(dir=".") as temporary:
        result = {"region_test": verify_regions(), "http_test": verify_service(Path(temporary))}
    if args.processor:
        result["lora_training_smoke"] = verify_lora(args.processor)
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
