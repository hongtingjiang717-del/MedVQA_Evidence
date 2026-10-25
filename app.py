"""Gradio应用：基于真实影像的公开模型问答，仅用于研究和演示。"""
from functools import lru_cache

import gradio as gr
from medvqa import VQAPredictor


@lru_cache(maxsize=1)
def get_model():
    return VQAPredictor()


def answer(image, question, compare_text_only):
    if image is None or not question or not question.strip():
        return "请上传医学图像并输入问题。", ""
    try:
        model = get_model()
        with_image = model.answer(question, image)
        without_image = model.answer(question, None) if compare_text_only else "未启用"
        return with_image, without_image
    except Exception as e:
        return f"模型运行失败：{type(e).__name__}: {e}", ""


with gr.Blocks(title="MedVQA-Evidence | 医学视觉问答与消融") as demo:
    gr.Markdown("# 医学影像视觉问答与视觉证据消融\n研究原型：对照‘图像+问题’与‘仅问题文字’的回答差异。**非医疗诊断工具。**")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="上传公开或获许可的医学图像")
            question = gr.Textbox(label="英文问题", value="What modality is used to take this image?")
            compare = gr.Checkbox(label="同时运行仅文字基线（增加一次推理）", value=True)
            run = gr.Button("运行真实模型")
        with gr.Column():
            original = gr.Textbox(label="图像 + 问题 的输出")
            ablation = gr.Textbox(label="仅问题文字 的输出")
    run.click(answer, inputs=[image, question, compare], outputs=[original, ablation])
    gr.Markdown("*模型回答可能错误；两个输出相同不意味着图像无用，两个输出不同也不证明答案正确。真实评估请使用 evaluate.py。*")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
