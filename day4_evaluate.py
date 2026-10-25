#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 4: 可续跑的医学视觉依赖正式验证集分析。

特点：
* 只运行原图(image)与仅文本(text_only)两组核心对照；不再使用无法追溯的旧错误图组。
* 和 evaluate.py 使用相同的 random.Random(seed).shuffle 取样方式，验证后重用已经
  跑完的50条结果，不会覆盖旧结果。
* 每条预测落盘，断线后继续；输入、原始标签、图像哈希核验防止混用。
* 聚类Bootstrap以唯一图像为单位；输出包含配对转移和逐题人工复核表。

Usage:
  python day4_evaluate.py --limit 150
  python day4_evaluate.py --limit 150 --report-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from medvqa import DATASET_ID, MODEL_ID, exact_match, load_data, normalize_answer, prompt_for, token_f1

MODES = ("image", "text_only")


def digest_image(im):
    """计算RGB图像字节的SHA256，包含宽高，便于相同影像聚类。"""
    rgb = im.convert("RGB")
    h = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode())
    h.update(rgb.tobytes())
    return h.hexdigest()


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第{line_no}行JSON无效（疑似中断写入）：{e}") from e
    return rows


def safe_append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def validate_record(row, *, item, idx, mode, split, model, img_hash):
    """只复用和当前SLAKE数据、提示词、模型相符的既往结果。"""
    checks = {
        "index": idx,
        "mode": mode,
        "split": split,
        "model": model,
        "question": str(item["question"]),
        "ground_truth": str(item["answer"]),
    }
    for k, v in checks.items():
        if row.get(k) != v:
            raise RuntimeError(f"记录不一致，拒绝复用: index={idx}, mode={mode}, {k}: {row.get(k)!r} != {v!r}")
    if row.get("image_sha256") and row["image_sha256"] != img_hash:
        raise RuntimeError(f"图像哈希不一致，拒绝复用: index={idx}")
    if "prediction" not in row:
        raise RuntimeError(f"记录缺少prediction: index={idx}")
    if row.get("prompt_template_sha256") and row["prompt_template_sha256"] != prompt_hash():
        raise RuntimeError("推理提示模板和本次运行不同，拒绝复用。")


def prompt_hash():
    return hashlib.sha256(prompt_for("<QUESTION>").encode("utf-8")).hexdigest()


def make_row(src, *, item, idx, mode, split, model, img_hash, origin):
    pred = str(src["prediction"])
    gold = str(item["answer"])
    return {
        "schema": "medvqa_day4_v1",
        "dataset": DATASET_ID,
        "split": split,
        "index": idx,
        "mode": mode,
        "model": model,
        "question": str(item["question"]),
        "ground_truth": gold,
        "prediction": pred,
        "exact_match": exact_match(pred, gold),
        "token_f1": token_f1(pred, gold),
        "image_sha256": img_hash,
        "prompt_template_sha256": prompt_hash(),
        "decoding": "do_sample=False,max_new_tokens=48",
        "origin": origin,
    }


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    rank = (len(values) - 1) * q
    low, high = int(rank), min(int(rank) + 1, len(values) - 1)
    return values[low] * (high - rank) + values[high] * (rank - low)


def aggregate(pairs, *, seed, iters=3000):
    result = {}
    for mode in MODES:
        v = [p[mode] for p in pairs]
        result[mode] = {
            "n_questions": len(v),
            "exact_match": round(mean(float(r["exact_match"]) for r in v), 4),
            "token_f1": round(mean(float(r["token_f1"]) for r in v), 4),
        }
    different = [p for p in pairs if float(p["image"]["exact_match"]) != float(p["text_only"]["exact_match"])]
    helped = sum(p["image"]["exact_match"] > p["text_only"]["exact_match"] for p in different)
    hurt = len(different) - helped
    result["paired_transitions"] = {
        "image_correct_text_wrong": helped,
        "image_wrong_text_correct": hurt,
        "both_same_em": len(pairs) - len(different),
        "answer_changed": sum(normalize_answer(p["image"]["prediction"]) != normalize_answer(p["text_only"]["prediction"]) for p in pairs),
    }
    groups = defaultdict(list)
    for p in pairs:
        groups[p["image"]["image_sha256"]].append(p)
    clusters = list(groups.values())
    base_em = mean(float(p["image"]["exact_match"]) - float(p["text_only"]["exact_match"]) for p in pairs)
    base_f1 = mean(float(p["image"]["token_f1"]) - float(p["text_only"]["token_f1"]) for p in pairs)
    rng = random.Random(seed + 2026)
    em_samples = []
    f1_samples = []
    for _ in range(iters):
        selected = [clusters[rng.randrange(len(clusters))] for __ in clusters]
        n = sum(len(group) for group in selected)
        em_samples.append(sum(float(p["image"]["exact_match"]) - float(p["text_only"]["exact_match"]) for group in selected for p in group) / n)
        f1_samples.append(sum(float(p["image"]["token_f1"]) - float(p["text_only"]["token_f1"]) for group in selected for p in group) / n)
    result["paired_effect"] = {
        "n_questions": len(pairs),
        "n_unique_images": len(groups),
        "em_diff_percentage_points": round(base_em * 100, 2),
        "em_cluster_bootstrap_95ci_pp": [round(percentile(em_samples, 0.025) * 100, 2), round(percentile(em_samples, 0.975) * 100, 2)],
        "token_f1_diff": round(base_f1, 4),
        "token_f1_cluster_bootstrap_95ci": [round(percentile(f1_samples, 0.025), 4), round(percentile(f1_samples, 0.975), 4)],
        "method_note": "按同一医学影像成簇Bootstrap。区间为探索性不确定性估计，不代表外部医院泛化能力。",
    }
    return result


def report(rows, *, indices, output, split, model, seed, all_valid=True):
    selected = set(indices)
    mapping = {}
    for r in rows:
        key = (r.get("index"), r.get("mode"))
        if r.get("split") != split or r.get("model") != model or r.get("index") not in selected or r.get("mode") not in MODES:
            continue
        if key in mapping:
            raise RuntimeError(f"结果文件重复：{key}；请先备份并检查，不能自动静默去重。")
        mapping[key] = r
    paired = []
    for idx in indices:
        if all((idx, m) in mapping for m in MODES):
            paired.append({m: mapping[(idx, m)] for m in MODES})
    summary_path = output.with_name(output.stem + "_summary.json")
    csv_path = output.with_name(output.stem + "_cases.csv")
    md_path = output.with_name(output.stem + "_report.md")
    summary = {
        "dataset": DATASET_ID,
        "split": split,
        "model": model,
        "seed": seed,
        "target_questions": len(indices),
        "completed_paired_questions": len(paired),
        "missing_predictions": len(indices) * len(MODES) - len(mapping),
        "status": "complete" if len(paired) == len(indices) else "partial",
        "metrics": aggregate(paired, seed=seed) if paired else {},
        "notes": ["严格EM不等于医学语义准确率；同义词和答案标签须人工复核。", "旧50条结果在验证一致后复用，正式测试集未参与调参。", "并不对语言模型的临床诊断安全性作出声明。"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["index", "image_sha256", "question", "ground_truth", "image_prediction", "text_prediction", "image_em", "text_em", "image_f1", "text_f1", "answer_changed", "label_for_review"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for p in paired:
            a,b = p["image"],p["text_only"]
            flag = "review" if (a["exact_match"] == 0 or b["exact_match"] == 0 or a["prediction"] != b["prediction"]) else "spot_check"
            writer.writerow({"index":a["index"],"image_sha256":a["image_sha256"],"question":a["question"],"ground_truth":a["ground_truth"],"image_prediction":a["prediction"],"text_prediction":b["prediction"],"image_em":a["exact_match"],"text_em":b["exact_match"],"image_f1":a["token_f1"],"text_f1":b["token_f1"],"answer_changed":normalize_answer(a["prediction"]) != normalize_answer(b["prediction"]),"label_for_review":flag})
    if paired:
        mm = summary["metrics"]
        eff=mm["paired_effect"]
        lines=["# Day 4 医学视觉依赖扩大评测（验证集）", "", f"- 已完整配对：{len(paired)} / {len(indices)}条问答", f"- 唯一影像：{eff['n_unique_images']}张", f"- 原图EM：{mm['image']['exact_match']:.1%}", f"- 仅文本EM：{mm['text_only']['exact_match']:.1%}", f"- EM差异：{eff['em_diff_percentage_points']:+.2f}个百分点", f"- 图像聚类Bootstrap 95%CI：{eff['em_cluster_bootstrap_95ci_pp']}", f"- 原图正确/文本错误：{mm['paired_transitions']['image_correct_text_wrong']}条；反向：{mm['paired_transitions']['image_wrong_text_correct']}条", "", "这不是临床评估。不能将字面答案匹配率解释为医疗诊断能力；应人工检查同义词与标签质量。"]
    else:
        lines=["# Day 4 暂无完整配对数据"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n===== Day 4 当前汇总 =====")
    print("已配对",len(paired),"/",len(indices),"；唯一图像", summary.get("metrics", {}).get("paired_effect",{}).get("n_unique_images",0))
    if paired:
        print("原图 EM",summary["metrics"]["image"]["exact_match"]," | 仅文本 EM",summary["metrics"]["text_only"]["exact_match"])
        print("EM差异(百分点)",summary["metrics"]["paired_effect"]["em_diff_percentage_points"])
        print("95%CI",summary["metrics"]["paired_effect"]["em_cluster_bootstrap_95ci_pp"])
    print("完整结果:",summary_path, csv_path, md_path, sep="\n  ")


def main():
    p=argparse.ArgumentParser(description="Day4 扩大SLAKE医学问答的原图 vs 仅文本对照；可复用Day2结果")
    p.add_argument("--limit",type=int,default=150,help="验证集问题条数，推荐150；后续可以加大到200")
    p.add_argument("--seed",type=int,default=42,help="必须保持Day2的固定种子42以复用前50条")
    p.add_argument("--split",choices=["validation"],default="validation",help="只允许验证集，锁定测试集避免调参污染")
    p.add_argument("--model",default=MODEL_ID)
    p.add_argument("--previous",default="results/experiments.jsonl")
    p.add_argument("--output",default="results/day4_validation.jsonl")
    p.add_argument("--report-only",action="store_true",help="只根据已有记录生成当前报告，不启动GPU")
    args=p.parse_args()
    if args.seed != 42:
        raise SystemExit("为了保证前50条样本顺序及旧结果复用一致，当前脚本要求 --seed 42")
    ds=load_data()[args.split]
    indices=list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    if args.limit <=0:
        raise SystemExit("limit必须为正整数，例如150或200")
    indices=indices[:args.limit]
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    existing=read_jsonl(output)
    completed={}
    for row in existing:
        if row.get("split") != args.split or row.get("model") != args.model:
            raise RuntimeError("已有Day4输出含其他模型或数据划分，拒绝继续混写")
        key=(row.get("index"), row.get("mode"))
        if key in completed:
            raise RuntimeError(f"Day4输出存在重复结果：{key}")
        if row.get("mode") not in MODES:
            raise RuntimeError("Day4输出包含意外实验条件")
        completed[key]=row
        idx,mode=key
        if not isinstance(idx,int) or not (0<=idx<len(ds)):
            raise RuntimeError(f"索引无效：{idx}")
        item=ds[idx]
        validate_record(row,item=item,idx=idx,mode=mode,split=args.split,model=args.model,img_hash=digest_image(item["image"]))
    # 严格核对历史结果，仅接收相同样本的原图/纯文本结果。
    old=read_jsonl(Path(args.previous)) if not args.report_only else []
    old_map={}
    for row in old:
        if row.get("split")==args.split and row.get("model")==args.model and row.get("mode") in MODES and row.get("index") in indices:
            key=(row["index"],row["mode"])
            if key in old_map:
                raise RuntimeError(f"原有结果存在重复：{key}；先人工检查")
            old_map[key]=row
    reused=0
    if not args.report_only:
        for idx in indices:
            item=ds[idx]
            img_hash=digest_image(item["image"])
            for mode in MODES:
                key=(idx,mode)
                if key not in completed and key in old_map:
                    row=old_map[key]
                    validate_record(row,item=item,idx=idx,mode=mode,split=args.split,model=args.model,img_hash=img_hash)
                    new=make_row(row,item=item,idx=idx,mode=mode,split=args.split,model=args.model,img_hash=img_hash,origin="reused_day2_checked")
                    safe_append(output,new)
                    completed[key]=new
                    reused+=1
        print(f"复用旧结果 {reused} 条；目前完成 {len(completed)}/{len(indices)*2} 次预测。")
        missing=[(idx,mode) for idx in indices for mode in MODES if (idx,mode) not in completed]
        if missing:
            print(f"还有{len(missing)}次推理需要补跑。开始加载已有Qwen3-VL权重……")
            from medvqa import VQAPredictor
            predictor=VQAPredictor(args.model)
            for t,(idx,mode) in enumerate(missing,1):
                item=ds[idx]
                inp=item["image"] if mode=="image" else None
                prediction=predictor.answer(str(item["question"]),inp)
                src={"prediction":prediction}
                new=make_row(src,item=item,idx=idx,mode=mode,split=args.split,model=args.model,img_hash=digest_image(item["image"]),origin="day4_inference")
                safe_append(output,new)
                completed[(idx,mode)]=new
                if t<=3 or t%10==0 or t==len(missing):
                    print(f"[{t}/{len(missing)}] 第{idx}题 {mode} | 标准={str(item['answer'])!r} 预测={prediction!r}",flush=True)
        else:
            print("推理已全部存在，直接刷新报告（不加载GPU）。")
    report(list(completed.values()),indices=indices,output=output,split=args.split,model=args.model,seed=args.seed)

if __name__=='__main__':
    main()
