#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MedVQA-Evidence Day5: 真实图文对照演示 + 已验证统计摘要。

运行：python day5_app.py
离线预检：python day5_app.py --check

重要：不下载新的模型、不修改原文件 app.py，复用 medvqa.VQAPredictor。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / 'results' / 'day4_validation_summary.json'


def load_summary(path=SUMMARY):
    p = Path(path)
    if not p.is_file():
        return None
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if data.get('status') != 'complete':
        raise ValueError(f'当前结果尚未完整：{p}')
    if not data.get('metrics', {}).get('paired_effect'):
        raise ValueError('结果文件缺少 paired_effect，不显示不完整统计数据')
    return data


def summary_md(data):
    if data is None:
        return ("⚠️ 尚未找到 `results/day4_validation_summary.json`。实时问答仍可运行；"
                "若要展示真实指标，请确认已经执行 Day4，并将结果保留在项目 `results/` 目录。")
    m = data['metrics']
    a, b, e = m['image'], m['text_only'], m['paired_effect']
    counts = m.get('paired_transitions', {})
    low, high = e['em_cluster_bootstrap_95ci_pp']
    n = data['completed_paired_questions']
    return f"""### 已完成的验证集对照实验

**模型：** `{data['model']}`　 **数据集：** `{data['dataset']}`  
**样本：** `{n}`条配对问答、`{e['n_unique_images']}`张唯一医学影像；固定随机种子 `{data.get('seed', '未知')}`

| 指标 | 原图＋问题 | 只有问题 |
|---|---:|---:|
| Exact Match（完全匹配率） | **{a['exact_match']:.2%}** | **{b['exact_match']:.2%}** |
| Token-F1（词级F1） | **{a['token_f1']:.4f}** | **{b['token_f1']:.4f}** |

- **准确率差值：+{e['em_diff_percentage_points']:.2f}个百分点**（按唯一影像聚类的Bootstrap 95%区间：**{low:.2f}～{high:.2f}**个百分点）
- 图像输入使 **{counts.get('image_correct_text_wrong', '?')}** 题由错转对，但使 **{counts.get('image_wrong_text_correct', '?')}** 题由对转错。
- 原图与纯文本回答发生变化：**{counts.get('answer_changed', '?')} / {n}**。

**研究限定：** 此结果来自验证集，属于探索性比较；严格字面匹配不等于医学语义准确率，也不代表独立医院数据上的泛化能力。模型可能产生错误回答，不能作为临床诊断依据。
"""


@lru_cache(maxsize=1)
def get_predictor():
    from medvqa import VQAPredictor
    return VQAPredictor()


def predict(image, question, compare):
    question = (question or '').strip()
    if image is None:
        return '', '', '请先上传一张医学影像（推荐使用公开或获许可的数据）。'
    if not question:
        return '', '', '请输入一个英文医学影像问题。'
    try:
        model = get_predictor()
        answer_image = model.answer(question, image)
        answer_text = model.answer(question, None) if compare else '未启用文本对照'
        return answer_image, answer_text, '推理完成。回答差异不是正确性的证明，请结合影像和专家知识判断。'
    except Exception as exc:
        return '', '', f'推理失败：{type(exc).__name__}: {exc}'



@lru_cache(maxsize=1)
def get_slake_examples():
    """按图像内容去重，从已缓存的SLAKE验证集取10张真实图片和配对问题。"""
    from medvqa import load_data

    ds = load_data()["validation"]
    samples = []
    seen = set()
    for row in ds:
        img = row["image"].convert("RGB")
        fingerprint = hashlib.sha256(
            str(img.size).encode("utf-8") + img.tobytes()
        ).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        samples.append((img, str(row["question"]), str(row["answer"])))
        if len(samples) >= 10:
            break
    if not samples:
        raise RuntimeError("SLAKE验证集没有可用影像")
    return samples


def load_slake_example(number):
    """Gradio按钮：无需用户上传本地图片，直接显示数据集自带的图片。"""
    try:
        samples = get_slake_examples()
        index = int(number) - 1
        if index < 0 or index >= len(samples):
            raise ValueError(f"可选示例范围为1至{len(samples)}")
        img, question, answer = samples[index]
        return img.copy(), question, (f"已加载SLAKE验证集第{index + 1}张不同图像。"
                                      f"标准答案：{answer}。点击『开始真实模型推理』比较模型输出。")
    except Exception as exc:
        return None, "", f"读取SLAKE示例失败：{type(exc).__name__}: {exc}"

def make_demo():
    import gradio as gr
    data = load_summary()
    with gr.Blocks(title='MedVQA-Evidence | 医学视觉依赖评测') as demo:
        gr.Markdown('# MedVQA-Evidence｜医学影像视觉问答与证据依赖评测\n'
                    '基于公开 Qwen3-VL-2B-Instruct 与 SLAKE 数据集的研究演示。'
                    '**模型结果仅用于科研和求职展示，不构成诊断或医疗建议。**')
        with gr.Tab('实时图像问答'):
            gr.Markdown('上传医学影像，输入英文问题，同时比较 **有图像** 与 **仅文本** 两个条件。'
                        '默认启用对照，因此每次点击会顺序推理两次。')
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type='pil', label='医学图像（请勿上传私人患者资料）')
                    q = gr.Textbox(label='医学影像问题（英文）',
                                   value='What modality is used to take this image?', lines=2)
                    compare = gr.Checkbox(value=True, label='启用仅文本对照')
                    with gr.Row():
                        example_number = gr.Dropdown(
                            choices=[str(i) for i in range(1, 11)], value='1',
                            label='SLAKE真实图像示例编号（无需自行上传）')
                        sample_btn = gr.Button('载入SLAKE示例')
                    btn = gr.Button('开始真实模型推理', variant='primary')
                with gr.Column():
                    ans_img = gr.Textbox(label='原图 + 问题', lines=3)
                    ans_txt = gr.Textbox(label='仅文本问题', lines=3)
                    status = gr.Textbox(label='运行状态', lines=3)
            sample_btn.click(
                fn=load_slake_example, inputs=[example_number],
                outputs=[image, q, status])
            btn.click(fn=predict, inputs=[image, q, compare], outputs=[ans_img, ans_txt, status])
            gr.Markdown('**没有医学影像文件也能测试：** 选择示例编号，点击「载入SLAKE示例」，'
                        '图片和对应问题会自动填入。无需本地下载图像。')
            gr.Markdown('**提示：** 第一次点击时会加载已下载的模型权重。'
                        '相同答案不代表图像无用；不同答案也不代表回答正确。')
        with gr.Tab('150条验证集实验结果'):
            gr.Markdown(summary_md(data))
            plot_path = ROOT / 'figures' / 'day5_validation_metrics.png'
            if plot_path.is_file():
                gr.Image(value=str(plot_path), label='基于真实汇总数据绘制的结果图', interactive=False)
            gr.Markdown('图表和表格均读取你已完成的结果文件或基于该文件生成，**不是实时推理的准确率**。')
        with gr.Tab('方法与局限'):
            gr.Markdown('''### 科学问题
向医学视觉语言模型提供影像，相比只提供问题文字，是否能够提高医学影像问答正确率？

### 评测设计
- 数据：SLAKE 英文版本；固定随机种子选择验证集问答。
- 模型：Qwen3-VL-2B-Instruct；无微调；确定性生成设置。
- 核心对照：Image + Question 与 Question Only；150条配对问答来自73张不同图像。
- 指标：Exact Match（完全匹配）、Token-F1（词级F1）。
- 区间估计：按**唯一医学影像**为单位聚类的Bootstrap。

### 不应过度解读
- 字符串差异会使 `Transverse Plane` / `axial` 等医学同义词被判错。
- 部分SLAKE标签表达不够规范，尚需医学语义复核。
- 这是验证集的探索性结果，尚无独立测试集或跨医院验证。
- 未验证临床安全性，无法定位病灶，不具备检测或分割能力。''')
    return demo


def main():
    p = argparse.ArgumentParser(description='MedVQA Day5 成果展示')
    p.add_argument('--check', action='store_true', help='只检查路径和摘要，不加载模型、不启动网页')
    p.add_argument('--port', type=int, default=int(os.environ.get('PORT', '7860')))
    args = p.parse_args()
    data = load_summary()
    if args.check:
        print('项目目录:', ROOT)
        print('摘要文件:', SUMMARY)
        print('摘要状态:', '存在且已完成' if data else '未找到（实时问答仍能工作）')
        if data:
            print('配对样本:', data['completed_paired_questions'])
            print('原图EM:', data['metrics']['image']['exact_match'])
            print('仅文本EM:', data['metrics']['text_only']['exact_match'])
            print('区间:', data['metrics']['paired_effect']['em_cluster_bootstrap_95ci_pp'])
        print('原始模型脚本:', '存在' if (ROOT / 'medvqa.py').is_file() else '缺失！')
        return
    demo = make_demo()
    demo.launch(server_name='0.0.0.0', server_port=args.port, share=False)


if __name__ == '__main__':
    main()
