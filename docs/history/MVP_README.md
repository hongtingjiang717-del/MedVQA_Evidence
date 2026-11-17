> 早期 MVP 设计说明，保留历史状态。当前实验结果和运行命令以根目录 README 为准。

# MedVQA-Evidence：医学多模态视觉问答及反事实视觉依赖评测

**项目状态**：完整的源代码与评估框架；基础逻辑可离线单元测试。**本压缩包不包含模型权重、数据集文件；尚未在用户AutoDL RTX4090上运行真实权重**，因而不能声称已经跑通完整GPU实验、不能承诺任何准确率。

## 核心科研问题

> 医学视觉语言模型回答SLAKE医学影像问题时，是否真正使用了影像内容？在替换为错误图像、移除图像时，回答性能和内容如何变化？

三组严格对照：

- A 原图+英文问题（Image + Question）
- B 仅英文问题（Question-only）；用于识别语言捷径
- C 随机错误图+相同问题（Wrong-image）；用于观察反事实敏感性

评价指标：Exact Match（标准化答案完全匹配）、Token-F1（词汇重叠）、预测改变率；不能将改变率解释为置信度、真实性或临床安全性。**若基础模型能力不足，依然可以产生有意义的模型局限性研究，但不能伪称性能提升。**

## 为什么选它

- 数据集：`mdwiratathya/SLAKE-vqa-english`，约61MB压缩下载；已在Hugging Face确认公开，含图像+英文问题+答案三个字段，训练4919、验证1053、测试1061问答样本。
- 影像：SLAKE原始集包含642张医学影像（CT、MRI、X-ray）。整理版中同一影像会重复关联不同问题，不能把7千多条QA误报为7千多名患者。
- 模型：`Qwen/Qwen3-VL-2B-Instruct`，官方Apache-2.0开放，HF存储约4.27GB，需要单独下载。
- 官方参考：https://github.com/QwenLM/Qwen3-VL
- 原始SLAKE及其语义标注： https://huggingface.co/datasets/BoKelvin/SLAKE
- 原始SLAKE官方网站：https://www.med-vqa.com/slake/
- 应用：Gradio。
- **不需要YOLO、不需要病灶框、也没有伪造定位/诊断能力。**

## 运行环境（AutoDL推荐）

1. 选择带 CUDA PyTorch 的 RTX 4090 实例（常见24GB显存）；Python建议3.10/3.11。
2. `cd medvqa_evidence_mvp`
3. 用 `nvidia-smi` 确认GPU可用，再运行：

   ```bash
   python -m pip install -U "transformers>=4.57.0,<5" "accelerate>=1.2,<3" "datasets>=3,<5" "huggingface_hub>=0.30" "pillow>=10" "pandas>=2" "numpy>=1.26" "gradio>=5,<7" "tqdm>=4.66"
   ```

   如镜像安装包异常，建议优先原PyPI或官方ModelScope渠道；避免在已正常工作的PyTorch CUDA环境中随意升级torch。

4. **数据闸门：先测数据，暂不下载模型。**

   ```bash
   python preflight.py
   python preflight.py --exhaustive
   ```

   前者检查可获取性、字段、样本数；后者按图像SHA256检查训练/验证/测试是否含重合的原图（强烈推荐）。若跨集合出现重合，不应继续用原划分报告性能，必须按图像重分组。

5. **模型闸门：10条验证集问题的真实推理。** 首次自动下载约4.27GB模型文件：

   ```bash
   python evaluate.py --split validation --limit 10 --modes image
   ```

   确认模型能载入、回答后，再运行全部三组对照：

   ```bash
   python evaluate.py --split validation --limit 50 --modes image text_only wrong_image
   ```

   产生 `results/experiments.jsonl`（逐条原始预测）和 `results/experiments_summary.json`（统计值）。重复运行自动跳过同模型、数据集划分、序号与模式已完成的预测。

6. **正式评估：**仅在验证集调好prompt后，再对锁定的测试集一次性评估：

   ```bash
   python evaluate.py --split test --limit 0 --modes image text_only wrong_image --output results/test.jsonl
   ```

   `--limit 0` 代表完整分割，可能需要较长GPU时间。计算临床语义准确率需要人工标注或公正的语义评判，Token-F1不能代替。

7. **网页展示：**

   ```bash
   python app.py
   ```

   打开AutoDL映射到7860端口的Gradio地址；切勿把未授权的患者图像上传到公开服务。

8. **离线单元测试：**

   ```bash
   python -m unittest discover -s tests -v
   ```

## 止损条件（启动当日就执行）

- 数据集无法在目前网络和环境中获得：停止主路线，可选择已核实的 `flaviagiammarino/vqa-rad` 小规模数据，但先做图像级划分检查。
- Qwen权重无法下载、或GPU不能加载并完成10条真实推理：停止开展附加模块，绝不提前建设Agent、框选、报告生成。
- 10条能跑、50条效果弱：先保留评测项目，继续只做数据和错误分析；不承诺准确率达到某一阈值。

## 可选增量，不是核心交付依赖

- 错误类型：按检查部位、问题类型拆分（整理版不含官方类型字段；可从官方`BoKelvin/SLAKE`对照追加）。
- LoRA/QLoRA（低秩微调）：仅在基线跑通后开展；需要谨慎按图像划分，避免同图多问答泄漏。
- KG/RAG（知识图谱检索增强）：原始SLAKE含知识图谱，优先为知识型问题提供可追踪知识证据；不能把测试答案直接注入提示。

## 该项目目前尚未完成的验证

- 本运行容器不能拉取Hugging Face权重/数据，因此GPU推理和完整数据集审计未实际执行。
- 下载速度、AutoDL Python/PyTorch版本、GPU显存峰值、模型在数据集上的最终准确率尚未测得。
- 没有病变检测、分割、像素级定位或诊断报告生成模块，绝不宣传为这些能力。

## 简历写法（仅限结果跑出之后）

『基于Qwen3-VL-2B和SLAKE搭建医学影像视觉问答评测系统；复现开源多模态模型真实推理与Gradio展示，设计图像输入、仅文本输入、反事实图像三组消融，基于图像级泄漏审计与EM/Token-F1分析模型对视觉证据的依赖；最终在XX条样本上得出XX结果。』

定量数字XX必须来自真实评估文件，不能现在填写。
