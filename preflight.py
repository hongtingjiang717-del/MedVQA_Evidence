"""第一道闸门：检查数据和显卡，不下载大模型、不训练。"""
import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from medvqa import DATASET_ID, MODEL_ID, load_data


def fingerprint(image):
    """将原图像统一编码后计算哈希，可检测相同图片跨集合泄漏。"""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def audit(dataset, exhaustive=False):
    print(f"数据源: {DATASET_ID}")
    hashes = {}
    summary = {}
    for split in dataset:
        rows = dataset[split]
        count = Counter()
        failures = []
        for i in range(len(rows)):
            item = rows[i]
            if not item.get("question") or item.get("answer") is None:
                failures.append(i)
            if exhaustive:
                count[fingerprint(item["image"])] += 1
        summary[split] = {"samples": len(rows), "bad_labels": len(failures)}
        if exhaustive:
            hashes[split] = set(count)
            summary[split]["unique_image_hashes"] = len(count)
        print(split, summary[split])
        if len(rows):
            ex = rows[0]
            print("  示例:", ex["question"], "→", ex["answer"], "图像尺寸", ex["image"].size)
    if exhaustive:
        names = list(hashes)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                common = hashes[a] & hashes[b]
                print(f"图像哈希交叉 {a} ∩ {b}: {len(common)}")
                if common:
                    print("警告：不同数据分割出现相同图像，需排除跨集图像泄漏！")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exhaustive", action="store_true", help="完整按图像SHA256跨集合审计，耗时较长")
    args = p.parse_args()
    print("核实数据集：", DATASET_ID)
    print("核实模型权重：", MODEL_ID)
    try:
        import torch
        print("PyTorch：", torch.__version__, "CUDA：", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU：", torch.cuda.get_device_name(0))
    except ImportError:
        print("未安装torch。")
    print("开始检查公开数据（首次运行需要联网）...")
    ds = load_data()
    summary = audit(ds, exhaustive=args.exhaustive)
    Path("results").mkdir(exist_ok=True)
    Path("results/preflight.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
