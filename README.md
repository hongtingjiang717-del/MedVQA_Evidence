# MedVQA-Evidence｜医学影像视觉问答与视觉依赖评测

> **Qwen3-VL-2B-Instruct + SLAKE + 配对消融实验 + Gradio**  
> 研究问题：**医学视觉语言模型在回答医学影像问题时，真实图像是否比仅输入问题文字带来额外预测增益？**

![网页：有图像与仅文本的同题问答](assets/01_medical_vqa_demo.png)

## 1. 项目亮点

- **真实开源权重推理**：使用 `Qwen/Qwen3-VL-2B-Instruct`，基于 PyTorch / Transformers 实现医学影像视觉问答，不依赖外部付费API。
- **严谨的配对对照**：同一道问题同时评估「图像+问题」与「仅问题」；固定随机种子 `42`。
- **防止虚假泛化**：对训练、验证、测试进行图像内容哈希交叉审计；此次用户运行报告交叉计数均为 0（仅涵盖完全一致的图像）。
- **考虑数据相关性**：150条问答来自73张唯一图像，按图像分簇进行Bootstrap不确定性估计。
- **可视化工程交付**：Gradio网页支持SLAKE真实示例、一键加载医学图像、文字对照、统计结果页。

## 2. 真实实验结果（SLAKE **验证集**，非测试集）

| 指标 | 图像 + 问题 | 只有问题 |
|---|---:|---:|
| 问答数 | 150 | 150 |
| 唯一图像数 | 73 | 73 |
| Exact Match（规范化完全匹配率） | **47.33%（71/150）** | **35.33%（53/150）** |
| Token-F1（词级F1） | **0.5072** | **0.3807** |

- EM绝对增益：**+12.00个百分点**；图像簇Bootstrap 95%置信区间：**[+4.14, +19.85]个百分点**。
- Token-F1增益：**+0.1265**；图像簇Bootstrap 95%区间：**[+0.0497, +0.2008]**。
- 配对表现：**26题**由错变对，**8题**由对变错；**78/150**次回答文字发生变化。
- 150/150问答完成配对，预测缺失数为0。原始汇总字段保存在 [`results/day4_validation_summary.json`](results/day4_validation_summary.json)。

![150条问答的Gradio评测结果看板](assets/02_validation_dashboard.png)

![验证集EM对比](figures/day5_validation_metrics.png)

**解释边界**：该实验表明，在当前SLAKE验证集和字面评测指标下，图像输入提供了正向预测增益；不等于具备临床诊断可靠性。`axial` 与 `Transverse Plane` 这类同义答案可能被严格EM误判。尚未完成正式独立测试集和跨医院验证；统计区间为探索性结果。方法及限制详见 [`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md)。

## 3. 技术栈

| 技术 | 用途 |
|---|---|
| Python / PyTorch | 模型推理计算（部署已在RTX 4090 D验证） |
| Hugging Face Transformers | 加载Qwen3-VL-2B模型与多模态处理器 |
| Hugging Face Datasets | 获取并读取SLAKE英文版影像问答数据 |
| pandas / numpy | 整理评测结果 |
| Bootstrap（自助抽样） | 图像级聚类置信区间估计 |
| Gradio | 实时问答、仅文本消融、结果展示 |
| Git / GitHub | 代码版本管理与项目展示 |

## 4. 目录结构

```text
MedVQA-Evidence/
├── medvqa.py                 # 数据读取、模型推理、答案规范化与指标
├── preflight.py              # 数据下载与图像内容哈希审计
├── evaluate.py               # 早期50条三组探索实验（wrong_image有可追溯性局限）
├── day4_evaluate.py          # 正式的图像/文本配对评测与图像聚类Bootstrap
├── day5_app.py               # 最终Gradio演示，内置SLAKE示例
├── app.py                    # 早期基础演示入口（保留供版本对照）
├── requirements.txt
├── results/
│   └── day4_validation_summary.json  # 已验证的聚合结果，允许公开
├── figures/                  # 结果图
├── assets/                   # 项目演示截图
├── docs/                     # 方法报告、复现与简历说明
└── tests/                    # 自动化单元测试
```

## 5. 快速运行

已在AutoDL的 **RTX 4090 D、Python 3.12.3、PyTorch 2.5.1+cu124、CUDA可用** 环境完成实际推理与Gradio验证。

```bash
# 1. 进入工程（不要修改已正常工作的GPU版torch）
cd MedVQA-Evidence

# 2. 安装Python依赖，首次会额外下载数据与模型权重
python -m pip install -r requirements.txt

# 3. 预检查：支持检查三个split之间的相同图像
python preflight.py --exhaustive

# 4. 离线检查演示程序能读取汇总结果
python day5_app.py --check

# 5. 启动网页（需GPU运行模型）
python day5_app.py
```

在AutoDL平台使用实例的**7860端口映射**打开网页。内置SLAKE示例可直接加载影像，无需自行寻找PNG文件；点击“开始真实模型推理”，查看有图像及无图像两种回答。模型首次加载或云端网络下载可能较慢。

### 复现150条验证集对照

```bash
python day4_evaluate.py --limit 150
# 若推理文件已存在，可只根据现有记录重建报告
python day4_evaluate.py --limit 150 --report-only
```

**注意**：本代码仓库仅公开汇总指标，不随仓库分发原始逐题模型预测。运行完整评测需要你重新在相同模型/数据/环境条件下计算。验证集结果不能写成测试集结果。

### 本地离线单元测试

```bash
python -m unittest discover -s tests -v
```

## 6. 一张图的演示示例

示例问题：`What modality is used to take this image?`  
标准答案：`X-Ray`  
本次实际演示结果：**图像输入 → `X-ray`**，**仅文本 → `CT`**。

这说明在**这一道题**上图像信息改变了预测并使答案与标签一致；单题不能代表整体性能，也不能证明临床医学推理能力。

## 7. 结果可信度与局限

1. SLAKE同一医学影像可被多道问题复用：按图像分簇进行Bootstrap，不能把150条问题误称为150名患者。
2. 跨train/validation/test的图像哈希交叉均为0，是用户在AutoDL上的审计输出；这一结果不排除**相似影像、模型预训练污染或语义重复**。
3. 当前**严格EM是标准化字面匹配，不等同医学语义正确率**；尚未系统性人工复核医学同义词及可能错误标签。
4. Day2随机`wrong_image`条件缺少可溯源图像编号，故不作为Day4核心统计证据。
5. 未进行疾病筛查临床验证，未开发目标检测或像素级分割功能，不可作为医疗建议。

## 8. 公开数据与模型引用 / 版权说明

- 模型：[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) / [源码](https://github.com/QwenLM/Qwen3-VL)，遵守模型仓库的授权条款。
- 实际评测使用的SLAKE英文整理版：[mdwiratathya/SLAKE-vqa-english](https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english)。
- SLAKE原始数据由 **Bo Liu、Xiao-Ming Wu 等**发布，[官方项目](https://www.med-vqa.com/slake/)与[BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE)。原始数据集标示为 **CC BY 4.0**，项目演示截图中的SLAKE影像为公开数据的教学科研使用，引用和改编应遵守原始作者署名要求。
- 不提交完整模型权重、SLAKE原始数据集、私有医疗影像、缓存文件或未筛选的原始实验日志。

**免责声明**：本项目仅供方法研究、技术展示和求职作品集使用，不提供诊断或治疗建议。
