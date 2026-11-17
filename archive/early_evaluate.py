"""三组严格消融：原图、仅问题文字、错误图像；持续保存，断线不丢已经完成的结果。"""
import argparse
import csv
import json
import random
from pathlib import Path

from medvqa import MODEL_ID, VQAPredictor, exact_match, load_data, normalize_answer, token_f1


MODES = ("image", "text_only", "wrong_image")


def save_summary(results):
    from statistics import mean
    summary = {}
    for mode in MODES:
        group = [r for r in results if r["mode"] == mode]
        if not group:
            continue
        summary[mode] = {
            "n": len(group),
            "exact_match": round(mean(float(r["exact_match"]) for r in group), 4),
            "token_f1": round(mean(float(r["token_f1"]) for r in group), 4),
        }
    images = {r["index"]: r["prediction"] for r in results if r["mode"] == "image"}
    wrong = {r["index"]: r["prediction"] for r in results if r["mode"] == "wrong_image"}
    joint = images.keys() & wrong.keys()
    if joint:
        summary["counterfactual"] = {
            "n": len(joint),
            "prediction_change_rate": round(
                sum(normalize_answer(images[i]) != normalize_answer(wrong[i]) for i in joint) / len(joint), 4
            ),
            "note": "改变率不是模型置信度或诊断准确率。",
        }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--output", default="results/experiments.jsonl")
    p.add_argument("--model", default=MODEL_ID)
    args = p.parse_args()
    dataset = load_data()[args.split]
    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    if args.limit > 0:
        indices = indices[:args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    previous = []
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                previous.append(record)
                if record.get("split") == args.split and record.get("model") == args.model:
                    completed.add((record["split"], record["index"], record["mode"], record["model"]))
            except (json.JSONDecodeError, KeyError):
                continue
    print("加载公开模型真实权重：", args.model)
    model = VQAPredictor(args.model)
    # 错误图像来自同一个split，但与原图不相同。
    for k, idx in enumerate(indices, 1):
        item = dataset[idx]
        image, question, gold = item["image"], str(item["question"]), str(item["answer"])
        for mode in args.modes:
            key = (args.split, idx, mode, args.model)
            if key in completed:
                continue
            if mode == "text_only":
                inp_image = None
            elif mode == "wrong_image":
                other = (idx + len(dataset) // 2 + rng.randrange(1, max(2, len(dataset) // 4))) % len(dataset)
                inp_image = dataset[other]["image"]
                # 过滤相同图像的重复问答：必要时退而使用空白图像。
                if inp_image.tobytes() == image.tobytes():
                    from PIL import Image
                    inp_image = Image.new("RGB", image.size, color="gray")
            else:
                inp_image = image
            prediction = model.answer(question, inp_image)
            record = {
                "split": args.split, "index": idx, "mode": mode, "model": args.model,
                "question": question, "ground_truth": gold, "prediction": prediction,
                "exact_match": exact_match(prediction, gold), "token_f1": token_f1(prediction, gold),
            }
            with output.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            previous.append(record)
            completed.add(key)
            print(f"[{k}/{len(indices)}] {mode}: {question[:42]} | 答案={gold!r} | 模型={prediction!r}")
    # 汇总限定为本次数据集、模型、随机抽样索引，避免混用旧结果。
    view = [r for r in previous if r.get("split") == args.split and r.get("model") == args.model and r.get("index") in indices and r.get("mode") in args.modes]
    summary = save_summary(view)
    summary.update({"split": args.split, "limit": len(indices), "seed": args.seed, "model": args.model})
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n汇总结果：", json.dumps(summary, ensure_ascii=False, indent=2))
    print("结果保存至", output, "和", summary_path)


if __name__ == "__main__":
    main()
